"""
Shared loader for cross-validation fold predictions.

Every downstream analysis in this package -- competing risk, outcome
ascertainment, the threshold sweep, equivalence testing -- operates on the
pooled out-of-fold predictions rather than refitting anything, so they all read
through this module and all land on the same probability scale.

Recalibration
-------------
`fit_apply_recalibration` is deliberately byte-for-byte the same procedure as
`core_eval_functions.fit_apply_recalibration` in the lancet-digital-health-eval-suite:
fit an unregularised logistic regression of the outcome on the logit of the
training-fold predictions, then apply it to the held-out fold.

The duplication is intentional -- this repository must stay runnable without
the evaluation suite on the path -- but the two must not drift. `verify_against_eval_suite()`
imports the suite when it is available and asserts the two agree, so a change
in one is caught rather than silently producing two tables on different scales.
"""

from __future__ import annotations

import os
import json
import glob

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


# ---------------------------------------------------------------------------
# Recalibration
# ---------------------------------------------------------------------------

def fit_apply_recalibration(y_train_true, y_train_prob, y_test_prob):
    """Logistic recalibration fitted on TRAIN predictions, applied to TEST.

    Monotone within a fold, so that fold's AUROC is unchanged, but it moves
    patients across a fixed absolute-risk threshold -- which is why every
    threshold-dependent analysis must use the same scale.
    """
    y_train_prob = np.clip(np.asarray(y_train_prob, dtype=float), 1e-7, 1 - 1e-7)
    train_logit = np.log(y_train_prob / (1 - y_train_prob))

    # C=inf is the unregularised fit; penalty=None was deprecated in sklearn 1.8.
    lr = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=1000)
    lr.fit(train_logit.reshape(-1, 1), np.asarray(y_train_true))

    y_test_prob = np.clip(np.asarray(y_test_prob, dtype=float), 1e-7, 1 - 1e-7)
    test_logit = np.log(y_test_prob / (1 - y_test_prob))
    return lr.predict_proba(test_logit.reshape(-1, 1))[:, 1]


def verify_against_eval_suite(seed: int = 0, n: int = 500) -> bool:
    """Assert this recalibration matches the evaluation suite's, if importable.

    Returns True when verified, False when the suite is not on the path.
    """
    try:
        from core_eval_functions import fit_apply_recalibration as suite_fn
    except ImportError:
        return False

    rng = np.random.default_rng(seed)
    y_train = rng.integers(0, 2, n)
    p_train = rng.uniform(0.01, 0.99, n)
    p_test = rng.uniform(0.01, 0.99, n // 2)

    mine = fit_apply_recalibration(y_train, p_train, p_test)
    theirs = suite_fn(y_train, p_train, p_test)
    if not np.allclose(mine, theirs, atol=1e-10):
        raise AssertionError(
            "Recalibration in src/analysis/predictions.py has drifted from "
            "core_eval_functions.fit_apply_recalibration. The two MUST agree or "
            "the threshold tables and the NRI table will sit on different scales."
        )
    print("[Predictions] Recalibration verified against the evaluation suite.")
    return True


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _fold_files(experiment_dir):
    files = sorted(glob.glob(os.path.join(experiment_dir, "fold_*_predictions.json")))
    if not files:
        raise FileNotFoundError(
            f"No fold_*_predictions.json in {experiment_dir}. "
            f"Run the experiment before this analysis."
        )
    return files


def load_experiment_predictions(experiment_dir, recalibrate=True):
    """Pooled out-of-fold predictions for one experiment.

    Returns a DataFrame with columns:
        patient_id, y_true, y_proba, y_proba_raw, fold

    `y_proba` is recalibrated when `recalibrate=True` (the scale the manuscript
    reports); `y_proba_raw` is always the model's untransformed output, so a
    pre-recalibration sensitivity analysis needs no second load.

    Each patient appears exactly once, since every row is in exactly one test
    fold of the outer split.
    """
    frames = []
    for path in _fold_files(experiment_dir):
        with open(path) as f:
            test = json.load(f)

        fold_num = int(os.path.basename(path).split("_")[1])
        y_true = np.asarray(test["y_true"], dtype=int)
        y_raw = np.asarray(test["y_proba"], dtype=float)
        y_cal = y_raw

        if recalibrate:
            # Rename the FILE, not the path. The experiment directory is itself
            # named "..._fold_results", so a replace over the whole path
            # rewrites the directory and then fails to find a file that is
            # sitting right there.
            directory, filename = os.path.split(path)
            train_path = os.path.join(
                directory, filename.replace("fold_", "train_", 1))
            if not os.path.exists(train_path):
                raise FileNotFoundError(
                    f"recalibrate=True needs {os.path.basename(train_path)} in "
                    f"{directory}, alongside {os.path.basename(path)}. Re-run the "
                    f"experiment, or pass recalibrate=False."
                )
            with open(train_path) as f:
                train = json.load(f)
            y_cal = fit_apply_recalibration(
                np.asarray(train["y_true"], dtype=int),
                np.asarray(train["y_proba"], dtype=float),
                y_raw,
            )

        ids = test.get("patient_ids")
        if ids is None:
            # Pre-rename runs did not save patient ids; fall back to the row
            # index within the full matrix, which is still a valid join key
            # within one CV design.
            ids = [f"row_{i}" for i in test.get("test_indices", range(len(y_true)))]

        frames.append(pd.DataFrame({
            "patient_id": [str(p) for p in ids],
            "y_true": y_true,
            "y_proba": y_cal,
            "y_proba_raw": y_raw,
            "fold": fold_num,
        }))

    pooled = pd.concat(frames, ignore_index=True)

    duplicated = pooled["patient_id"].duplicated().sum()
    if duplicated:
        print(f"[Predictions] WARNING: {duplicated} duplicate patient_id(s) across "
              f"folds in {os.path.basename(experiment_dir)}. Each patient should "
              f"appear in exactly one test fold.")
    return pooled


def load_many(experiment_dirs: dict, recalibrate=True) -> dict:
    """Load several experiments at once. `experiment_dirs` is {label: path}."""
    out = {}
    for label, path in experiment_dirs.items():
        try:
            out[label] = load_experiment_predictions(path, recalibrate=recalibrate)
        except FileNotFoundError as exc:
            print(f"[Predictions] Skipping {label}: {exc}")
    return out


def find_experiment_dirs(results_root, names=None):
    """Map experiment name -> most recent results directory under `results_root`.

    Directories are named `<timestamp>_<experiment>_fold_results`, so the latest
    run of each experiment is the lexicographically greatest match.
    """
    if not os.path.isdir(results_root):
        return {}

    found = {}
    for entry in sorted(os.listdir(results_root)):
        if not entry.endswith("_fold_results"):
            continue
        path = os.path.join(results_root, entry)
        if not os.path.isdir(path):
            continue
        parts = entry.split("_")
        if len(parts) < 4:
            continue
        name = "_".join(parts[2:-2])
        if names is not None and name not in names:
            continue
        # Later timestamps sort later, so a plain assignment keeps the newest.
        found[name] = path
    return found


# ---------------------------------------------------------------------------
# Threshold metrics, computed identically everywhere
# ---------------------------------------------------------------------------

def threshold_metrics(y_true, y_proba, threshold):
    """Confusion-matrix metrics at one absolute-risk threshold.

    Degenerate denominators return NaN rather than 0, so a bootstrap resample
    with no events is dropped from the percentiles instead of biasing them
    toward zero.
    """
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba, dtype=float)
    flagged = y_proba >= threshold

    tp = int(np.sum(flagged & (y_true == 1)))
    fp = int(np.sum(flagged & (y_true == 0)))
    tn = int(np.sum(~flagged & (y_true == 0)))
    fn = int(np.sum(~flagged & (y_true == 1)))
    n = int(y_true.size)

    def ratio(num, den):
        return float(num / den) if den > 0 else np.nan

    return {
        "threshold": float(threshold),
        "n": n,
        "n_events": int(y_true.sum()),
        "prevalence": ratio(int(y_true.sum()), n),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "alert_rate": ratio(tp + fp, n),
        "sensitivity": ratio(tp, tp + fn),
        "specificity": ratio(tn, tn + fp),
        "ppv": ratio(tp, tp + fp),
        "npv": ratio(tn, tn + fn),
        "net_benefit": (
            float(tp / n - (fp / n) * (threshold / (1 - threshold)))
            if n > 0 and threshold < 1 else np.nan
        ),
    }
