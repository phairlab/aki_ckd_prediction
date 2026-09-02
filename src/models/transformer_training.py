"""
Transformer training and prediction.

Changes for the resubmission
----------------------------
* Every architecture and optimisation hyperparameter is now passed in rather
  than hardcoded, so the model can be tuned inside nested cross-validation
  (editor's point 1).
* Optional class weighting in the loss. The outcome rate is 6.1%; leaving this
  fixed at "off" was one of the untuned choices the editor objected to, so it
  is now part of the search space rather than an assumption.
* Seeding is explicit and threaded through, so a tuned configuration can be
  refitted and reproduced.
* An optional per-epoch callback supports Optuna pruning, which is what makes
  a 10-outer-fold nested search affordable.

BUG FIX retained from the original: the line saving the best weights had
`bes_model_state` (typo), so `best_model_state` was never assigned and the
final-epoch model was returned regardless of early stopping.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, StratifiedKFold

from src.models.transformer_model import build_transformer, count_parameters


# ---------------------------------------------------------------------------
# Defaults reproducing the originally submitted configuration
# ---------------------------------------------------------------------------

DEFAULT_PARAMS = {
    "architecture": "row_token",
    "embedding_dim": 64,
    "num_heads": 4,
    "num_layers": 2,
    "dropout": 0.1,
    "ff_multiplier": 4,
    "activation": "relu",
    "learning_rate": 5e-5,
    "weight_decay": 1e-5,
    "batch_size": 32,
    "epochs": 100,
    "early_stopping": 10,
    "validation_split": 0.15,
    "class_weight": None,        # None | "balanced"
    "grad_clip": 1.0,
}


def resolve_params(params=None, **overrides):
    """Merge caller parameters over the submitted-configuration defaults.

    Accepts the legacy keyword names used by the original code so existing
    call sites keep working:
        model_size="small"/"large"  ->  architecture
        early_stopping_patience     ->  early_stopping
    """
    merged = dict(DEFAULT_PARAMS)
    for source in (params or {}, overrides):
        for key, value in source.items():
            if value is None and key in ("params",):
                continue
            merged[key] = value

    if "model_size" in merged:
        size = merged.pop("model_size")
        if size in ("small", "large"):
            merged.setdefault("architecture", size)
            merged["architecture"] = size if merged.get("architecture") in (None, "row_token") else merged["architecture"]
    if "early_stopping_patience" in merged:
        merged["early_stopping"] = merged.pop("early_stopping_patience")

    # d_model is the FeatureTokenTransformer's name for the embedding width.
    merged.setdefault("d_model", merged.get("embedding_dim", 64))
    return merged


def set_seed(seed):
    """Seed python/numpy/torch so a refit of a tuned configuration reproduces."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _class_weight_tensor(y, mode, device):
    """Inverse-frequency weights for CrossEntropyLoss, or None."""
    if mode not in ("balanced", True):
        return None
    y = np.asarray(y)
    counts = np.bincount(y.astype(int), minlength=2).astype(float)
    counts[counts == 0] = 1.0
    weights = len(y) / (2.0 * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def train_with_validation(X_train, y_train, device,
                          params=None, seed=1202, epoch_callback=None,
                          verbose=True, **overrides):
    """Train a transformer with validation-based early stopping.

    Parameters
    ----------
    X_train, y_train : training fold, already scaled and imputed.
    device : torch device.
    params : dict of hyperparameters; missing keys fall back to the originally
        submitted configuration (see DEFAULT_PARAMS).
    seed : seeds weight init, the train/validation split and shuffling.
    epoch_callback : called as ``fn(epoch, val_loss)`` after each epoch. Raise
        from inside it to stop training early (Optuna pruning does this).
    **overrides : convenience keyword form of `params`, and the legacy names.

    Returns
    -------
    (model, train_losses, val_losses, best_epoch)
    """
    p = resolve_params(params, **overrides)
    set_seed(seed)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train,
        test_size=p["validation_split"],
        random_state=seed,
        stratify=y_train,
    )

    to_x = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32).to(device)
    to_y = lambda a: torch.tensor(np.asarray(a), dtype=torch.long).to(device)

    train_loader = DataLoader(
        TensorDataset(to_x(X_tr), to_y(y_tr)),
        batch_size=int(p["batch_size"]), shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    val_loader = DataLoader(
        TensorDataset(to_x(X_val), to_y(y_val)),
        batch_size=int(p["batch_size"]), shuffle=False,
    )

    model = build_transformer(
        input_dim=X_train.shape[1],
        architecture=p["architecture"],
        embedding_dim=int(p.get("embedding_dim", 64)),
        d_model=int(p.get("d_model", 64)),
        num_heads=int(p["num_heads"]),
        num_layers=int(p["num_layers"]),
        dropout=float(p["dropout"]),
        ff_multiplier=p.get("ff_multiplier", 4),
        activation=p.get("activation", "relu"),
    ).to(device)

    criterion = nn.CrossEntropyLoss(
        weight=_class_weight_tensor(y_tr, p.get("class_weight"), device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(p["learning_rate"]),
        weight_decay=float(p["weight_decay"]),
    )

    best_val_loss = float("inf")
    best_model_state = None
    best_epoch = 0
    counter = 0
    train_losses, val_losses = [], []

    for epoch in range(int(p["epochs"])):
        model.train()
        total = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            if p.get("grad_clip"):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(p["grad_clip"]))
            optimizer.step()
            total += loss.item()
        train_loss = total / max(len(train_loader), 1)
        train_losses.append(train_loss)

        model.eval()
        total = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                total += criterion(model(inputs), labels).item()
        val_loss = total / max(len(val_loader), 1)
        val_losses.append(val_loss)

        if verbose:
            print(f"    epoch {epoch+1}/{int(p['epochs'])}  "
                  f"train {train_loss:.4f}  val {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            counter = 0
        else:
            counter += 1

        if epoch_callback is not None:
            epoch_callback(epoch, val_loss)

        if counter >= int(p["early_stopping"]):
            if verbose:
                print(f"    early stopping after epoch {epoch+1}")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, train_losses, val_losses, best_epoch


def predict_transformer(model, X_test, device, batch_size=4096):
    """Run inference and return (y_pred, probability of the positive class).

    Batched so that a feature-tokenised model on a large fold does not exhaust
    GPU memory (its attention cost scales with n_features squared).
    """
    model.eval()
    probs_all = []
    X = np.asarray(X_test, dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            chunk = torch.tensor(X[start:start + batch_size], dtype=torch.float32).to(device)
            probs_all.append(torch.softmax(model(chunk), dim=1).cpu().numpy())
    probs = np.concatenate(probs_all, axis=0) if probs_all else np.zeros((0, 2))
    return probs.argmax(axis=1), probs[:, 1]


def oof_train_probas(X_train, y_train, device, n_splits=5, seed=1202,
                     params=None, verbose=True, **overrides):
    """Inner-CV out-of-fold probabilities for the training set.

    Each row gets a prediction from a model that never trained on it. These are
    the honest predictions the logistic recalibration map must be fitted on --
    in-sample predictions are memorised and produce the wrong map. The
    evaluation suite consumes these as train_*_predictions.json.

    The inner models are configured identically to the final model, so `params`
    must be the same dict used for the outer fit.
    """
    oof = np.full(len(y_train), np.nan, dtype=np.float64)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for k, (fit_idx, held_idx) in enumerate(skf.split(X_train, y_train), 1):
        if verbose:
            print(f"  recalibration inner fold {k}/{n_splits} "
                  f"(fit {len(fit_idx)}, held-out {len(held_idx)}, "
                  f"held-out events {int(np.asarray(y_train)[held_idx].sum())})")

        inner_model, _, _, _ = train_with_validation(
            X_train[fit_idx], np.asarray(y_train)[fit_idx], device=device,
            params=params, seed=seed + k, verbose=False, **overrides,
        )
        _, p = predict_transformer(inner_model, X_train[held_idx], device=device)
        oof[held_idx] = p

        del inner_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    assert not np.isnan(oof).any(), "some rows never received an out-of-fold prediction"
    return oof


def describe_model(X, params=None, **overrides):
    """One-line capacity description, for the methods section."""
    p = resolve_params(params, **overrides)
    model = build_transformer(
        input_dim=X.shape[1], architecture=p["architecture"],
        embedding_dim=int(p.get("embedding_dim", 64)), d_model=int(p.get("d_model", 64)),
        num_heads=int(p["num_heads"]), num_layers=int(p["num_layers"]),
        dropout=float(p["dropout"]),
    )
    return (f"{p['architecture']} | d_model={p.get('d_model')} heads={p['num_heads']} "
            f"layers={p['num_layers']} dropout={p['dropout']} | "
            f"{count_parameters(model):,} trainable parameters")
