"""
Transformer training and prediction functions.

train_with_validation — trains with early stopping on a validation split.
predict_transformer   — inference on held-out data.

BUG FIX: The original code had `bes_model_state` (typo) on the line that
saves the best weights, so best_model_state was never updated and the model
from the *last* epoch was always returned.  Fixed here.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

from src.models.transformer_model import TabularTransformer, LargeTabularTransformer


def train_with_validation(X_train, y_train, device,
                          epochs=20, batch_size=32,
                          validation_split=0.15, early_stopping=5,
                          learning_rate=1e-4, model_size="small"):
    """Train a transformer with validation-based early stopping.

    Returns (model, train_losses, val_losses, best_epoch).
    """
    # Train / validation split
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train,
        test_size=validation_split,
        random_state=1202,
        stratify=y_train,
    )

    X_tr_t = torch.tensor(X_tr, dtype=torch.float32).to(device)
    y_tr_t = torch.tensor(y_tr, dtype=torch.long).to(device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.long).to(device)

    train_loader = DataLoader(
        TensorDataset(X_tr_t, y_tr_t), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(X_val_t, y_val_t), batch_size=batch_size, shuffle=False
    )

    # Pick architecture
    if model_size == "large":
        model = LargeTabularTransformer(input_dim=X_train.shape[1]).to(device)
    else:
        model = TabularTransformer(input_dim=X_train.shape[1]).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    best_val_loss = float("inf")
    best_model_state = None
    counter = 0
    train_losses = []
    val_losses = []
    best_epoch = 0

    for epoch in range(epochs):
        # --- train ---
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # --- validate ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                val_loss += criterion(model(inputs), labels).item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()  # FIX: was `bes_model_state`
            best_epoch = epoch
            counter = 0
        else:
            counter += 1

        if counter >= early_stopping:
            print(f"Early stopping triggered after epoch {epoch+1}")
            break

    # Restore best weights
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, train_losses, val_losses, best_epoch


def oof_train_probas(X_train, y_train, device, n_splits=5, seed=1202, **train_kwargs):
    """Inner-CV out-of-fold probabilities for the training set.

    Each row gets a prediction from a model that never trained on it. These are
    the honest predictions the calibration map must be fitted on -- in-sample
    predictions are memorised and produce the wrong map.

    train_kwargs are passed straight to train_with_validation, so the inner
    models must be configured identically to the final model.
    """
    oof = np.full(len(y_train), np.nan, dtype=np.float64)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for k, (fit_idx, held_idx) in enumerate(skf.split(X_train, y_train), 1):
        print(f"  inner fold {k}/{n_splits} "
              f"(fit {len(fit_idx)}, held-out {len(held_idx)}, "
              f"held-out events {int(y_train[held_idx].sum())})")

        inner_model, _, _, _ = train_with_validation(
            X_train[fit_idx], y_train[fit_idx], device=device, **train_kwargs
        )
        _, p = predict_transformer(inner_model, X_train[held_idx], device=device)
        oof[held_idx] = p

        del inner_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    assert not np.isnan(oof).any(), "some rows never received an out-of-fold prediction"
    return oof


def predict_transformer(model, X_test, device):
    """Run inference and return (y_pred, probas_positive_class)."""
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tensor)
        probs = torch.softmax(outputs, dim=1)
        predictions = torch.argmax(probs, dim=1)
    return predictions.cpu().numpy(), probs[:, 1].cpu().numpy()
