"""
Hyperparameter tuning inside nested cross-validation.

Addresses the editor's primary concern: the submitted models used XGBoost
package defaults and a priori transformer settings, so the null result could
not establish that complex models fail to extract additional signal.

Design
------
Nested cross-validation. The outer loop is the existing 10-fold split that
produces the reported predictions. Inside each outer training fold an
independent inner StratifiedKFold search runs, and the configuration selected
there is refitted on the whole outer training fold. The outer test fold is
never seen by the search, so the reported performance stays an honest estimate
of a *tuned* model rather than of a model tuned on its own test data.

Search is Optuna's TPE sampler (Bayesian, the editor's named alternative),
with pruning at inner-fold granularity: a trial whose running mean AUROC is
already worse than the median of completed trials at the same inner fold is
abandoned. Without that, a 10-outer-fold transformer search is not affordable.

If Optuna is unavailable the module degrades to seeded random search over the
same distributions, so the pipeline still runs; the objective, the inner-fold
protocol and the returned artefacts are identical.

Everything about a search is recorded -- every trial, its parameters and its
inner-fold scores -- and written next to the fold results, so the supplement
can report what was searched rather than only what won.
"""

from __future__ import annotations

import json
import os
import sys
import time
import warnings
from typing import Callable, Optional

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import config

try:
    import optuna
    from optuna.samplers import TPESampler
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:                                            # pragma: no cover
    HAS_OPTUNA = False


# ---------------------------------------------------------------------------
# Search spaces
# ---------------------------------------------------------------------------

def _suggest_xgboost(trial, pos_weight_max: float):
    """XGBoost search space.

    `scale_pos_weight` is included deliberately. The outcome rate is 6.1% and
    the submitted models applied no class weighting at all -- that was an
    untuned choice, not a finding, so it belongs in the search rather than in
    the methods as an assumption.
    """
    return {
        "n_estimators":      trial.suggest_int("n_estimators", 100, 2000, log=True),
        "max_depth":         trial.suggest_int("max_depth", 2, 10),
        "learning_rate":     trial.suggest_float("learning_rate", 5e-3, 0.3, log=True),
        "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "min_child_weight":  trial.suggest_float("min_child_weight", 1.0, 30.0, log=True),
        "gamma":             trial.suggest_float("gamma", 1e-8, 5.0, log=True),
        "reg_alpha":         trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda", 1e-3, 100.0, log=True),
        "scale_pos_weight":  trial.suggest_float("scale_pos_weight", 1.0, pos_weight_max, log=True),
    }


def _suggest_transformer(trial, n_features: int):
    """Transformer search space, spanning both architectures.

    `architecture` is itself searched. The submitted model embeds the whole row
    as a single token, which makes self-attention a no-op (see
    models/transformer_model.py); `feature_token` is a genuine FT-Transformer
    where attention runs over one token per clinical variable. Searching across
    both is what lets the resubmission claim an upper bound on transformer
    performance rather than on one arbitrary configuration.

    d_model is sampled as a multiple of the head count so the two are always
    compatible.
    """
    architecture = trial.suggest_categorical("architecture", ["row_token", "feature_token"])
    num_heads = trial.suggest_categorical("num_heads", [2, 4, 8])
    head_dim = trial.suggest_categorical("head_dim", [8, 16, 32])
    d_model = num_heads * head_dim

    params = {
        "architecture":     architecture,
        "num_heads":        num_heads,
        "d_model":          d_model,
        "embedding_dim":    d_model,
        "num_layers":       trial.suggest_int("num_layers", 1, 4),
        "dropout":          trial.suggest_float("dropout", 0.0, 0.5),
        "ff_multiplier":    trial.suggest_categorical("ff_multiplier", [1, 2, 4]),
        "learning_rate":    trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
        "weight_decay":     trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        "batch_size":       trial.suggest_categorical("batch_size", [16, 32, 64, 128]),
        "class_weight":     trial.suggest_categorical("class_weight", [None, "balanced"]),
        "activation":       trial.suggest_categorical("activation", ["relu", "gelu"]),
    }

    # Feature-token attention is quadratic in the number of features. Cap the
    # width for wide inputs so a single trial cannot exhaust GPU memory.
    if architecture == "feature_token" and n_features > 150 and d_model > 128:
        params["d_model"] = params["embedding_dim"] = 128
    return params


# Distributions for the no-Optuna fallback, mirroring the spaces above.
_RANDOM_SPACE_XGB = {
    "n_estimators":     ("int_log", 100, 2000),
    "max_depth":        ("int", 2, 10),
    "learning_rate":    ("float_log", 5e-3, 0.3),
    "subsample":        ("float", 0.5, 1.0),
    "colsample_bytree": ("float", 0.3, 1.0),
    "min_child_weight": ("float_log", 1.0, 30.0),
    "gamma":            ("float_log", 1e-8, 5.0),
    "reg_alpha":        ("float_log", 1e-8, 10.0),
    "reg_lambda":       ("float_log", 1e-3, 100.0),
}


class _RandomTrial:
    """Minimal stand-in for an Optuna trial, for the no-Optuna fallback."""

    def __init__(self, rng):
        self.rng = rng
        self.params = {}

    def suggest_int(self, name, low, high, log=False):
        v = (int(round(np.exp(self.rng.uniform(np.log(low), np.log(high))))) if log
             else int(self.rng.integers(low, high + 1)))
        self.params[name] = v
        return v

    def suggest_float(self, name, low, high, log=False):
        v = (float(np.exp(self.rng.uniform(np.log(low), np.log(high)))) if log
             else float(self.rng.uniform(low, high)))
        self.params[name] = v
        return v

    def suggest_categorical(self, name, choices):
        v = choices[int(self.rng.integers(0, len(choices)))]
        self.params[name] = v
        return v

    def report(self, value, step):
        pass

    def should_prune(self):
        return False


# ---------------------------------------------------------------------------
# Objectives
# ---------------------------------------------------------------------------

def _inner_cv_score(fit_predict: Callable, X, y, inner_folds: int, seed: int,
                    trial=None) -> float:
    """Mean inner-fold AUROC for one candidate configuration.

    `fit_predict(X_fit, y_fit, X_held)` returns held-out positive-class
    probabilities. Intermediate means are reported to the trial after each
    inner fold so a clearly poor configuration can be pruned early.
    """
    skf = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed)
    scores = []
    for step, (fit_idx, held_idx) in enumerate(skf.split(X, y)):
        y_held = np.asarray(y)[held_idx]
        if len(np.unique(y_held)) < 2:
            continue
        proba = fit_predict(X[fit_idx], np.asarray(y)[fit_idx], X[held_idx])
        scores.append(roc_auc_score(y_held, proba))

        if trial is not None:
            trial.report(float(np.mean(scores)), step)
            if trial.should_prune():
                import optuna as _o
                raise _o.TrialPruned()

    return float(np.mean(scores)) if scores else 0.5


def _run_search(objective: Callable, n_trials: int, seed: int, label: str,
                use_pruning: bool = True):
    """Run the search with Optuna if present, else seeded random search.

    Returns (best_params, record) where record documents every trial.
    """
    t0 = time.time()

    if HAS_OPTUNA:
        pruner = (optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=1)
                  if use_pruning else optuna.pruners.NopPruner())
        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=seed, n_startup_trials=min(10, max(5, n_trials // 5))),
            pruner=pruner,
            study_name=label,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # gc_after_trial releases the trial's tensors before the next one
            # starts, which matters when several folds share a GPU.
            study.optimize(objective, n_trials=n_trials,
                           gc_after_trial=True, show_progress_bar=False)

        completed = [t for t in study.trials if t.value is not None]
        record = {
            "search": "optuna_tpe",
            "n_trials_requested": n_trials,
            "n_trials_completed": len(completed),
            "n_trials_pruned": sum(1 for t in study.trials
                                   if t.state == optuna.trial.TrialState.PRUNED),
            "best_value": float(study.best_value),
            "best_params": dict(study.best_params),
            "elapsed_sec": round(time.time() - t0, 1),
            "trials": [
                {"number": t.number, "value": (float(t.value) if t.value is not None else None),
                 "state": str(t.state).split(".")[-1], "params": dict(t.params)}
                for t in study.trials
            ],
        }
        return study.best_trial, record

    # ---- fallback: seeded random search -----------------------------------
    print(f"    [tuning] optuna not installed; using seeded random search "
          f"({n_trials} trials). pip install optuna for TPE.")
    rng = np.random.default_rng(seed)
    best_value, best_trial, trials = -np.inf, None, []
    for i in range(n_trials):
        trial = _RandomTrial(rng)
        value = objective(trial)
        trials.append({"number": i, "value": float(value), "state": "COMPLETE",
                       "params": dict(trial.params)})
        if value > best_value:
            best_value, best_trial = value, trial

    record = {
        "search": "random",
        "n_trials_requested": n_trials,
        "n_trials_completed": n_trials,
        "n_trials_pruned": 0,
        "best_value": float(best_value),
        "best_params": dict(best_trial.params),
        "elapsed_sec": round(time.time() - t0, 1),
        "trials": trials,
    }
    return best_trial, record


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def tune_xgboost(X, y, n_trials: int = 60, inner_folds: int = 5, seed: int = 1202,
                 label: str = "xgb") -> tuple[dict, dict]:
    """Search XGBoost hyperparameters on one outer training fold.

    X must carry NaN where values are missing -- XGBoost's native missing
    handling is part of what is being tuned, so the caller must not have
    imputed.

    Returns (best_params, search_record).
    """
    from xgboost import XGBClassifier

    y = np.asarray(y).astype(int)
    n_pos = max(int(y.sum()), 1)
    n_neg = max(int(len(y) - n_pos), 1)
    pos_weight_max = float(np.clip(n_neg / n_pos, 2.0, 50.0))

    def objective(trial):
        params = _suggest_xgboost(trial, pos_weight_max)

        def fit_predict(X_fit, y_fit, X_held):
            clf = XGBClassifier(
                **params, random_state=seed,
                # Pinned, not -1: hist is not reproducible across thread
                # counts, and n_jobs=-1 makes the search depend on how many
                # workers happen to share the machine.
                n_jobs=getattr(config, "XGBOOST_N_JOBS", 4),
                eval_metric="logloss", tree_method="hist",
            )
            clf.fit(X_fit, y_fit, verbose=False)
            return clf.predict_proba(X_held)[:, 1]

        return _inner_cv_score(fit_predict, X, y, inner_folds, seed, trial)

    best_trial, record = _run_search(objective, n_trials, seed, f"{label}_xgb")

    best = dict(record["best_params"])
    best.update({"random_state": seed,
                 "n_jobs": getattr(config, "XGBOOST_N_JOBS", 4),
                 "eval_metric": "logloss", "tree_method": "hist"})
    return best, record


def tune_transformer(X, y, device, n_trials: int = 30, inner_folds: int = 3,
                     seed: int = 1202, max_epochs: int = 100,
                     early_stopping: int = 10, label: str = "tf") -> tuple[dict, dict]:
    """Search transformer hyperparameters on one outer training fold.

    Fewer inner folds than XGBoost by default: each evaluation trains a network,
    so the inner loop dominates runtime. Three folds keeps the estimate usable
    while making a 10-outer-fold nested search finish overnight on a GPU.

    X must already be imputed and scaled (fold-local statistics only).
    """
    from src.models.transformer_training import train_with_validation, predict_transformer

    y = np.asarray(y).astype(int)
    n_features = X.shape[1]

    def objective(trial):
        params = _suggest_transformer(trial, n_features)
        params.update({"epochs": max_epochs, "early_stopping": early_stopping})

        def fit_predict(X_fit, y_fit, X_held):
            model, _, _, _ = train_with_validation(
                X_fit, y_fit, device=device, params=params, seed=seed, verbose=False,
            )
            _, proba = predict_transformer(model, X_held, device=device)
            del model
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            return proba

        return _inner_cv_score(fit_predict, X, y, inner_folds, seed, trial)

    best_trial, record = _run_search(objective, n_trials, seed, f"{label}_tf")

    # Reconstruct the full parameter dict: the search space derives d_model
    # from num_heads x head_dim, so the raw best_params are not directly usable.
    raw = record["best_params"]
    d_model = raw["num_heads"] * raw["head_dim"]
    if raw["architecture"] == "feature_token" and n_features > 150 and d_model > 128:
        d_model = 128

    best = {
        "architecture":  raw["architecture"],
        "num_heads":     raw["num_heads"],
        "d_model":       d_model,
        "embedding_dim": d_model,
        "num_layers":    raw["num_layers"],
        "dropout":       raw["dropout"],
        "ff_multiplier": raw["ff_multiplier"],
        "learning_rate": raw["learning_rate"],
        "weight_decay":  raw["weight_decay"],
        "batch_size":    raw["batch_size"],
        "class_weight":  raw["class_weight"],
        "activation":    raw["activation"],
        "epochs":        max_epochs,
        "early_stopping": early_stopping,
    }
    return best, record


def tune_logreg(X, y, n_trials: int = 30, inner_folds: int = 5, seed: int = 1202,
                label: str = "logreg") -> tuple[dict, dict]:
    """Search regularisation strength for the logistic-regression baseline.

    Kept deliberately small: the parametric baseline exists to be a clean,
    interpretable reference, not to win. Tuning only C and the penalty keeps it
    honest without turning it into another black box.
    """
    from sklearn.linear_model import LogisticRegression

    y = np.asarray(y).astype(int)

    def objective(trial):
        penalty = trial.suggest_categorical("penalty", ["l2", "none_"])
        C = trial.suggest_float("C", 1e-3, 1e3, log=True)
        class_weight = trial.suggest_categorical("class_weight", [None, "balanced"])

        def fit_predict(X_fit, y_fit, X_held):
            kwargs = dict(max_iter=5000, random_state=seed, class_weight=class_weight)
            if penalty == "none_":
                kwargs["C"] = np.inf          # unregularised; penalty=None deprecated in sklearn 1.8
            else:
                kwargs["C"] = C
            clf = LogisticRegression(**kwargs)
            clf.fit(X_fit, y_fit)
            return clf.predict_proba(X_held)[:, 1]

        return _inner_cv_score(fit_predict, X, y, inner_folds, seed, trial)

    best_trial, record = _run_search(objective, n_trials, seed, f"{label}_lr",
                                     use_pruning=False)

    raw = record["best_params"]
    best = {
        "C": np.inf if raw["penalty"] == "none_" else raw["C"],
        "class_weight": raw["class_weight"],
        "max_iter": 5000,
        "random_state": seed,
    }
    return best, record


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_search_record(record: dict, output_dir: str, fold_num, model_type: str) -> str:
    """Write one fold's search record, and append a one-line summary."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"tuning_fold_{fold_num}_{model_type}.json")
    with open(path, "w") as f:
        json.dump(record, f, indent=2, default=str)
    return path


def summarise_tuning(output_dir: str) -> Optional[dict]:
    """Collapse every per-fold search in a directory into a supplement table.

    Reports the selected value of each hyperparameter across the 10 outer
    folds. Stability here is itself a result worth reporting: if the search
    picks wildly different configurations fold to fold, that is evidence the
    data does not determine an optimum, which supports the paper's argument
    more directly than any single tuned score.
    """
    import glob
    import pandas as pd

    paths = sorted(glob.glob(os.path.join(output_dir, "tuning_fold_*_*.json")))
    if not paths:
        return None

    rows = []
    for path in paths:
        with open(path) as f:
            rec = json.load(f)
        stem = os.path.basename(path).replace(".json", "").split("_")
        row = {"fold": stem[2], "model": "_".join(stem[3:]),
               "best_inner_auroc": rec.get("best_value"),
               "n_completed": rec.get("n_trials_completed"),
               "n_pruned": rec.get("n_trials_pruned"),
               "elapsed_sec": rec.get("elapsed_sec"),
               "search": rec.get("search")}
        row.update({f"param_{k}": v for k, v in rec.get("best_params", {}).items()})
        rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, "tuning_summary.csv")
    df.to_csv(csv_path, index=False)

    param_cols = [c for c in df.columns if c.startswith("param_")]
    stability = {}
    for col in param_cols:
        vals = df[col].dropna()
        if vals.empty:
            continue
        stability[col.replace("param_", "")] = {
            "n_distinct_across_folds": int(vals.nunique()),
            "modal_value": str(vals.mode().iloc[0]) if len(vals.mode()) else None,
            "values": [str(v) for v in vals.tolist()],
        }

    summary = {
        "n_searches": len(df),
        "mean_best_inner_auroc": float(df["best_inner_auroc"].mean()),
        "total_tuning_hours": round(float(df["elapsed_sec"].sum()) / 3600, 2),
        "hyperparameter_stability": stability,
        "csv": csv_path,
    }
    with open(os.path.join(output_dir, "tuning_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[Tuning] Summary across {len(df)} searches -> {csv_path}")
    print(f"[Tuning] Mean best inner AUROC: {summary['mean_best_inner_auroc']:.4f}  "
          f"| total search time {summary['total_tuning_hours']} h")
    return summary
