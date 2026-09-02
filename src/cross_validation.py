"""
Nested cross-validation engine.

Outer loop: 10-fold stratified CV producing the predictions that are reported.
Inner loop: an independent hyperparameter search on each outer training fold.

What changed for the resubmission
---------------------------------
* StratifiedKFold, not KFold.  The original passed labels to `KFold.split()`,
  which ignores them; at a 6.1% event rate that put avoidable variation in the
  per-fold event count, and the manuscript's argument leans on fold-to-fold SD.
* Hyperparameters are searched inside each outer training fold (editor point 1)
  rather than left at package defaults.
* Imputation is fitted on the training fold only (see data_preprocessing).
* XGBoost genuinely receives NaN.  The original applied
  `np.nan_to_num(..., nan=0.0)` after StandardScaler to every model, which is
  mean imputation in standardized space -- so the Methods claim that "XGBoost
  retained missing values directly" did not describe the code.  It now does.
* Feature selection uses the SAME model-agnostic selector for every
  architecture.  Previously XGBoost used RFE (driven by XGBoost's own
  importances) and the transformer used SelectKBest, so a difference between
  the two confounded architecture with selection method.
* `mutual_info_classif` is seeded, so selection is reproducible.
* SHAP is computed out-of-fold (editor point 6).
* patient_id is written into every prediction file.
* Folds are dispatched across GPUs (see parallel.py).
"""

from __future__ import annotations

import os
import sys
import json
import time
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.feature_selection import RFE, SelectKBest, mutual_info_classif
from sklearn.base import clone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

import config
from src.evaluation.metrics import compute_fold_metrics, aggregate_fold_metrics
from src.data_preprocessing import FoldImputer
from src import parallel


def _seeded_mutual_info(X, y):
    """Mutual information with a fixed seed.

    Module-level rather than a lambda for two reasons: `mutual_info_classif` is
    stochastic and the original code left it unseeded, so feature selection was
    not reproducible; and a lambda cannot be pickled, so a SelectKBest holding
    one cannot be saved with the fold artifacts.
    """
    return mutual_info_classif(X, y, random_state=config.RANDOM_SEED)


# ===========================================================================
# Per-fold worker
# ===========================================================================
# Must be a module-level function: the process pool uses the 'spawn' start
# method (required for CUDA) and therefore pickles it by qualified name.

def run_one_fold(device: str, n_threads: int, fold_num: int,
                 payload_path: str, train_idx, test_idx,
                 exp: dict, output_dir: str) -> dict:
    """Fit, tune, predict and explain one outer fold. Returns fold metrics."""
    t0 = time.time()

    with open(payload_path, "rb") as f:
        payload = pickle.load(f)

    features = payload["features"]
    labels = payload["labels"]
    feature_names = np.asarray(payload["feature_names"])
    patient_ids = payload["patient_ids"]
    imputation_plan = payload["imputation_plan"]

    train_idx = np.asarray(train_idx)
    test_idx = np.asarray(test_idx)

    X_train_raw = features[train_idx].astype(float)
    X_test_raw = features[test_idx].astype(float)
    y_train = labels[train_idx].astype(int)
    y_test = labels[test_idx].astype(int)

    model_type = exp["model_type"]
    print(f"\n[Fold {fold_num}] {exp['name']} on {device} "
          f"({len(train_idx)} train / {len(test_idx)} test, "
          f"{int(y_train.sum())} train events)", flush=True)

    # ---------------------------------------------------------------- scale
    # StandardScaler ignores NaN when computing statistics and preserves NaN
    # on transform, so this is safe to run before deciding whether to impute.
    scaler = StandardScaler().fit(X_train_raw)
    X_train_scaled = scaler.transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    # --------------------------------------------------------------- impute
    imputer = FoldImputer(imputation_plan).fit(X_train_scaled)
    X_train_imputed = imputer.transform(X_train_scaled)
    X_test_imputed = imputer.transform(X_test_scaled)

    if model_type == "xgboost":
        # Preserve missingness: XGBoost learns a default split direction for
        # NaN, which is the behaviour the manuscript describes.
        X_train_model, X_test_model = X_train_scaled, X_test_scaled
    else:
        X_train_model, X_test_model = X_train_imputed, X_test_imputed

    # ------------------------------------------------------------ selection
    selector = None
    selected = np.arange(X_train_model.shape[1])
    n_select = exp.get("n_features")
    method = exp.get("feature_selection_method")

    if method and n_select and n_select < X_train_model.shape[1]:
        # Selection always runs on the imputed matrix, whatever the model
        # consumes, so that every architecture is handed the same columns and
        # the comparison isolates the model.
        if method == "rfe":
            from xgboost import XGBClassifier
            estimator = XGBClassifier(random_state=config.RANDOM_SEED,
                                      n_jobs=n_threads, tree_method="hist",
                                      eval_metric="logloss")
            selector = RFE(estimator=estimator, n_features_to_select=n_select,
                           step=exp.get("rfe_step", 0.1))
            selector.fit(X_train_imputed, y_train)
            selected = np.where(selector.support_)[0]
        elif method == "selectkbest":
            selector = SelectKBest(score_func=_seeded_mutual_info, k=n_select)
            selector.fit(X_train_imputed, y_train)
            selected = np.where(selector.get_support())[0]
        else:
            raise ValueError(f"Unknown feature_selection_method: {method!r}")

        X_train_model = X_train_model[:, selected]
        X_test_model = X_test_model[:, selected]
        print(f"[Fold {fold_num}] {method}: {len(selected)} of "
              f"{features.shape[1]} features", flush=True)

    feature_names_fold = feature_names[selected]

    # --------------------------------------------------------------- tuning
    best_params, search_record = _tune_fold(
        exp, X_train_model, y_train, device, n_threads, fold_num, output_dir)

    # ------------------------------------------------------ fit and predict
    model, y_pred, probas, train_probas_oof = _fit_and_predict(
        model_type, best_params, X_train_model, y_train, X_test_model,
        device, n_threads, exp)

    fold_metrics = compute_fold_metrics(y_test, y_pred, probas)
    fold_metrics["fold"] = fold_num
    # `n` and `n_train` are what the Nadeau-Bengio correction needs
    # (equivalence.py reads them back from fold_N.json); without them the
    # correction silently falls back to assuming n_train = (k-1) * n_test.
    fold_metrics["n"] = int(len(test_idx))
    fold_metrics["n_events"] = int(y_test.sum())
    fold_metrics["n_train"] = int(len(train_idx))
    fold_metrics["n_features_used"] = int(X_train_model.shape[1])
    if search_record:
        fold_metrics["tuning_best_inner_auroc"] = search_record.get("best_value")

    # ----------------------------------------------------------- out-of-fold SHAP
    shap_summary = None
    if exp.get("perform_shap"):
        shap_summary = _fold_shap(
            model_type, model, X_test_model, feature_names_fold,
            device, fold_num, output_dir,
            max_samples=exp.get("shap_max_samples", 200))

    # --------------------------------------------------------------- persist
    _save_fold_artifacts(
        output_dir, fold_num, exp, model, scaler, imputer, selector,
        y_test, y_pred, probas, test_idx, patient_ids[test_idx],
        y_train, train_probas_oof, train_idx, patient_ids[train_idx],
        feature_names_fold, fold_metrics, best_params,
        payload.get("sex_values"))

    elapsed = time.time() - t0
    print(f"[Fold {fold_num}] done in {elapsed / 60:.1f} min — "
          f"ROC AUC {fold_metrics['roc_auc']:.4f}", flush=True)

    return {"metrics": fold_metrics, "shap": shap_summary,
            "best_params": best_params, "elapsed_sec": elapsed,
            "n_features_used": int(X_train_model.shape[1])}


# ---------------------------------------------------------------------------
# Fold sub-steps
# ---------------------------------------------------------------------------

def _tune_fold(exp, X_train, y_train, device, n_threads, fold_num, output_dir):
    """Run the inner hyperparameter search, or return the fixed defaults."""
    from src import tuning

    if not exp.get("tune"):
        # Reproduces the originally submitted configuration exactly, so a
        # tuned-vs-untuned comparison is available in the response letter.
        defaults = {
            "xgboost": {"random_state": config.RANDOM_SEED, "n_jobs": n_threads,
                        "eval_metric": "logloss", "tree_method": "hist"},
            "transformer": dict(exp.get("transformer_params") or {}),
            "logreg": {"max_iter": 5000, "random_state": config.RANDOM_SEED},
        }[exp["model_type"]]
        return defaults, None

    model_type = exp["model_type"]
    print(f"[Fold {fold_num}] tuning {model_type} "
          f"({exp.get('n_trials')} trials, {exp.get('inner_folds')} inner folds)...",
          flush=True)

    if model_type == "xgboost":
        best, record = tuning.tune_xgboost(
            X_train, y_train, n_trials=exp["n_trials"],
            inner_folds=exp["inner_folds"], seed=config.RANDOM_SEED,
            label=f"{exp['name']}_f{fold_num}")
        best["n_jobs"] = n_threads
    elif model_type == "transformer":
        best, record = tuning.tune_transformer(
            X_train, y_train, device=device, n_trials=exp["n_trials"],
            inner_folds=exp["inner_folds"], seed=config.RANDOM_SEED,
            max_epochs=exp.get("epochs", 100),
            early_stopping=exp.get("early_stopping_patience", 10),
            label=f"{exp['name']}_f{fold_num}")
    elif model_type == "logreg":
        best, record = tuning.tune_logreg(
            X_train, y_train, n_trials=exp["n_trials"],
            inner_folds=exp["inner_folds"], seed=config.RANDOM_SEED,
            label=f"{exp['name']}_f{fold_num}")
    else:
        raise ValueError(f"Unknown model_type: {model_type!r}")

    tuning.save_search_record(record, output_dir, fold_num, model_type)
    print(f"[Fold {fold_num}] best inner AUROC {record['best_value']:.4f} "
          f"after {record['n_trials_completed']} trials "
          f"({record['n_trials_pruned']} pruned, {record['elapsed_sec'] / 60:.1f} min)",
          flush=True)
    return best, record


def _fit_and_predict(model_type, params, X_train, y_train, X_test,
                     device, n_threads, exp):
    """Refit on the whole outer training fold and predict.

    Also returns honest out-of-fold probabilities for the TRAINING rows. The
    evaluation suite fits its logistic recalibration map on those; in-sample
    training predictions are memorised and produce the wrong map.
    """
    inner_splits = exp.get("recalibration_inner_folds", 5)

    if model_type == "xgboost":
        from xgboost import XGBClassifier
        model = XGBClassifier(**params)
        model.fit(X_train, y_train, verbose=False)
        probas = model.predict_proba(X_test)[:, 1]
        y_pred = (probas >= 0.5).astype(int)
        train_oof = cross_val_predict(
            clone(model), X_train, y_train,
            cv=StratifiedKFold(inner_splits, shuffle=True,
                               random_state=config.RANDOM_SEED),
            method="predict_proba")[:, 1]

    elif model_type == "transformer":
        from src.models.transformer_training import (
            train_with_validation, predict_transformer, oof_train_probas)
        model, _, _, _ = train_with_validation(
            X_train, y_train, device=device, params=params,
            seed=config.RANDOM_SEED, verbose=False)
        y_pred, probas = predict_transformer(model, X_test, device=device)
        train_oof = oof_train_probas(
            X_train, y_train, device=device, n_splits=inner_splits,
            seed=config.RANDOM_SEED, params=params, verbose=False)

    elif model_type == "logreg":
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(**params)
        model.fit(X_train, y_train)
        probas = model.predict_proba(X_test)[:, 1]
        y_pred = (probas >= 0.5).astype(int)
        train_oof = cross_val_predict(
            clone(model), X_train, y_train,
            cv=StratifiedKFold(inner_splits, shuffle=True,
                               random_state=config.RANDOM_SEED),
            method="predict_proba")[:, 1]

    else:
        raise ValueError(f"Unknown model_type: {model_type!r}")

    return model, np.asarray(y_pred), np.asarray(probas), np.asarray(train_oof)


def _fold_shap(model_type, model, X_test, feature_names, device,
               fold_num, output_dir, max_samples=200):
    """Mean |SHAP| per feature on this fold's HELD-OUT rows.

    Editor point 6: the submitted attributions came from one model fitted to
    all 4,687 patients and explained on those same patients. Computing them on
    held-out rows of each fold makes them out-of-sample, and the spread across
    the ten folds gives the fold-wise variability the editor asked for.

    Explainer choice matters for feasibility. TreeExplainer and LinearExplainer
    are exact and effectively free. For the transformer, KernelExplainer costs
    roughly (2 * n_features + 2048) forward passes PER explained row -- about
    450,000 passes for 200 rows at 100 features, per fold, times ten folds.
    GradientExplainer (expected gradients) is used instead: it is the standard
    choice for differentiable models, runs on the same device as the model, and
    turns hours into seconds. KernelExplainer remains the fallback, with a hard
    cap, for the case where autograd is unavailable.
    """
    try:
        import shap
    except ImportError:
        print(f"[Fold {fold_num}] shap not installed; skipping attribution.")
        return None

    try:
        if model_type == "xgboost":
            values = shap.TreeExplainer(model).shap_values(X_test)

        elif model_type == "logreg":
            values = shap.LinearExplainer(model, X_test).shap_values(X_test)

        else:
            values = _transformer_shap(model, X_test, device, max_samples, fold_num)
            if values is None:
                return None

        values = np.asarray(values)
        # Newer shap returns (n, features, classes) for a multiclass output;
        # older versions return a list per class. Reduce either to the positive
        # class so the fold summaries are comparable.
        if values.ndim == 3:
            values = values[:, :, 1] if values.shape[2] == 2 else values[..., -1]
        elif values.ndim == 1:
            values = values.reshape(1, -1)

        mean_abs = np.abs(values).mean(axis=0)
        summary = {"fold": fold_num,
                   "feature": [str(f) for f in feature_names],
                   "mean_abs_shap": mean_abs.tolist(),
                   "n_explained": int(values.shape[0])}

        with open(os.path.join(output_dir, f"shap_fold_{fold_num}.json"), "w") as f:
            json.dump(summary, f, indent=2)
        np.save(os.path.join(output_dir, f"shap_values_fold_{fold_num}.npy"), values)
        return summary

    except Exception as exc:                                   # noqa: BLE001
        # A failed attribution must not lose the fold's predictions, which are
        # the primary output. Report and carry on.
        print(f"[Fold {fold_num}] SHAP failed ({type(exc).__name__}: {exc}); continuing.")
        return None


def _transformer_shap(model, X_test, device, max_samples, fold_num):
    """SHAP values for a torch model: expected gradients, with a Kernel fallback."""
    import shap
    import torch

    n_explain = int(min(max_samples, len(X_test)))
    n_background = int(min(100, max(10, len(X_test) // 4)))

    X = np.asarray(X_test, dtype=np.float32)
    background = torch.tensor(X[:n_background], dtype=torch.float32).to(device)
    to_explain = torch.tensor(X[:n_explain], dtype=torch.float32).to(device)

    model.eval()
    try:
        explainer = shap.GradientExplainer(model, background)
        values = explainer.shap_values(to_explain)
        if isinstance(values, list):
            values = np.stack(values, axis=-1)
        return np.asarray(values)
    except Exception as exc:                                   # noqa: BLE001
        print(f"[Fold {fold_num}] GradientExplainer unavailable "
              f"({type(exc).__name__}: {exc}); falling back to KernelExplainer "
              f"on a reduced sample.")

    # Fallback. Both the sample size and nsamples are capped: left on 'auto',
    # KernelExplainer would need ~450k forward passes per fold at 100 features.
    n_explain = min(n_explain, 50)
    model_cpu = model.to("cpu")
    model_cpu.eval()

    def predict_fn(batch):
        with torch.no_grad():
            out = model_cpu(torch.tensor(np.asarray(batch, dtype=np.float32)))
            return torch.softmax(out, dim=1).numpy()

    summarised = shap.kmeans(X[:n_background], min(10, max(2, n_background // 5)))
    values = shap.KernelExplainer(predict_fn, summarised).shap_values(
        X[:n_explain], nsamples=200, silent=True)
    model.to(device)
    return values


def _save_fold_artifacts(output_dir, fold_num, exp, model, scaler, imputer, selector,
                         y_test, y_pred, probas, test_idx, test_patient_ids,
                         y_train, train_probas_oof, train_idx, train_patient_ids,
                         feature_names_fold, fold_metrics, best_params,
                         sex_values=None):
    """Write everything the evaluation suite and the analyses need."""
    test_prefix = os.path.join(output_dir, f"fold_{fold_num}")
    train_prefix = os.path.join(output_dir, f"train_{fold_num}")

    serializable = {k: (v.item() if isinstance(v, np.generic) else v)
                    for k, v in fold_metrics.items() if not isinstance(v, np.ndarray)}
    with open(f"{test_prefix}.json", "w") as f:
        json.dump(serializable, f, indent=4)

    if exp["model_type"] == "transformer":
        import torch
        torch.save(model.state_dict(), f"{test_prefix}_model.pt")
        with open(f"{test_prefix}_model_params.json", "w") as f:
            json.dump(best_params, f, indent=2, default=str)
    else:
        with open(f"{test_prefix}_model.pkl", "wb") as f:
            pickle.dump(model, f)

    for name, obj in (("scaler", scaler), ("imputer", imputer), ("selector", selector)):
        if obj is not None:
            with open(f"{test_prefix}_{name}.pkl", "wb") as f:
                pickle.dump(obj, f)

    with open(f"{test_prefix}_best_params.json", "w") as f:
        json.dump(best_params, f, indent=2, default=str)

    # patient_ids let the evaluation suite verify that two experiments scored
    # the same patients in the same order before computing a paired NRI.
    test_payload = {
        "y_true": np.asarray(y_test).tolist(),
        "y_pred": np.asarray(y_pred).tolist(),
        "y_proba": np.asarray(probas).tolist(),
        "test_indices": np.asarray(test_idx).tolist(),
        "patient_ids": [str(p) for p in np.asarray(test_patient_ids)],
    }
    train_payload = {
        "y_true": np.asarray(y_train).tolist(),
        "y_proba": np.asarray(train_probas_oof).tolist(),
        "train_indices": np.asarray(train_idx).tolist(),
        "patient_ids": [str(p) for p in np.asarray(train_patient_ids)],
    }
    if sex_values is not None:
        test_payload["sex"] = np.asarray(sex_values)[np.asarray(test_idx)].tolist()
        train_payload["sex"] = np.asarray(sex_values)[np.asarray(train_idx)].tolist()

    with open(f"{test_prefix}_predictions.json", "w") as f:
        json.dump(test_payload, f, indent=4)
    with open(f"{train_prefix}_predictions.json", "w") as f:
        json.dump(train_payload, f, indent=4)

    if exp["model_type"] == "xgboost" and hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        order = np.argsort(importances)[::-1]
        with open(f"{test_prefix}_feature_importances.txt", "w") as f:
            for rank, i in enumerate(order):
                f.write(f"{rank}\t{feature_names_fold[i]}\t{float(importances[i]):.6f}\n")


# ===========================================================================
# Orchestration
# ===========================================================================

def run_cross_validation(exp_config, data, devices=None, sequential=False,
                         output_dir=None):
    """Run nested 10-fold stratified CV for one experiment.

    Returns the path to the experiment output directory.
    """
    features = data["features"]
    labels = np.asarray(data["labels"]).astype(int)
    feature_names = np.asarray(data["feature_names"])
    patient_ids = np.asarray(data["patient_ids"])
    features_df = data.get("features_df")

    sex_values = None
    if features_df is not None and "sex" in features_df.columns:
        sex_values = pd.to_numeric(features_df["sex"], errors="coerce").to_numpy()
        if np.all(np.isnan(sex_values)):
            sex_values = None

    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        output_dir = os.path.join(config.get_experiments_dir(),
                                  f"{timestamp}_{exp_config.name}_fold_results")
    os.makedirs(output_dir, exist_ok=True)

    exp = _experiment_to_dict(exp_config, features, labels)
    with open(os.path.join(output_dir, "args.json"), "w") as f:
        json.dump(exp, f, indent=4, default=str)

    print(f"\n{'=' * 70}")
    print(f"Experiment : {exp_config.name}")
    print(f"Model      : {exp_config.model_type}   Features: {exp_config.feature_set}")
    print(f"Samples    : {features.shape[0]}   Input features: {features.shape[1]}")
    print(f"Events     : {int(labels.sum())} ({labels.mean() * 100:.1f}%)")
    print(f"Tuning     : {'ON — ' + str(exp['n_trials']) + ' trials/fold, '
                          + str(exp['inner_folds']) + ' inner folds'
                          if exp['tune'] else 'OFF (submitted defaults)'}")
    print(f"Output     : {output_dir}")
    print(f"{'=' * 70}")

    # Write the shared arrays once; workers memory-map them rather than
    # receiving a pickled copy of the matrix per task.
    payload_path = os.path.join(output_dir, "_fold_payload.pkl")
    with open(payload_path, "wb") as f:
        pickle.dump({"features": features, "labels": labels,
                     "feature_names": feature_names, "patient_ids": patient_ids,
                     "imputation_plan": data["imputation_plan"],
                     "sex_values": sex_values}, f)

    cv = StratifiedKFold(n_splits=config.N_OUTER_FOLDS, shuffle=True,
                         random_state=config.RANDOM_SEED)
    splits = list(cv.split(features, labels))

    event_counts = [int(labels[test].sum()) for _, test in splits]
    print(f"Fold event counts: {event_counts}  "
          f"(StratifiedKFold: range {min(event_counts)}-{max(event_counts)})")

    tasks = [{"fold_num": i + 1, "payload_path": payload_path,
              "train_idx": train, "test_idx": test,
              "exp": exp, "output_dir": output_dir}
             for i, (train, test) in enumerate(splits)]

    devices = devices or ["cpu"]
    results = parallel.run_tasks(run_one_fold, tasks, devices,
                                 sequential=sequential, label="fold")

    fold_metrics = [r["metrics"] for r in results]
    agg = aggregate_fold_metrics(fold_metrics)
    agg["total_fold_hours"] = round(sum(r["elapsed_sec"] for r in results) / 3600, 2)
    with open(os.path.join(output_dir, "aggregated_results.json"), "w") as f:
        json.dump(agg, f, indent=4, default=str)

    if exp["tune"]:
        from src import tuning
        tuning.summarise_tuning(output_dir)

    shap_summaries = [r["shap"] for r in results if r.get("shap")]
    if shap_summaries:
        summarise_out_of_fold_shap(shap_summaries, output_dir, exp_config.name)

    _save_selected_params(results, output_dir)

    try:
        os.remove(payload_path)
    except OSError:
        pass

    print(f"\nExperiment {exp_config.name} complete -> {output_dir}")
    return output_dir


def _experiment_to_dict(exp_config, features, labels):
    """Flatten the experiment config into a picklable dict for the workers."""
    d = {
        "name": exp_config.name,
        "model_type": exp_config.model_type,
        "feature_set": exp_config.feature_set,
        "target": exp_config.target,
        "n_features": exp_config.n_features,
        "feature_selection_method": exp_config.feature_selection_method,
        "rfe_step": getattr(exp_config, "rfe_step", 0.1),
        "perform_shap": getattr(exp_config, "perform_shap", True),
        "shap_max_samples": getattr(exp_config, "shap_max_samples", 200),
        "tune": getattr(exp_config, "tune", False),
        "n_trials": getattr(exp_config, "n_trials", 0),
        "inner_folds": getattr(exp_config, "inner_folds", 3),
        "recalibration_inner_folds": getattr(exp_config, "recalibration_inner_folds", 5),
        "epochs": getattr(exp_config, "epochs", 100),
        "early_stopping_patience": getattr(exp_config, "early_stopping_patience", 10),
        "n_samples": int(features.shape[0]),
        "n_features_input": int(features.shape[1]),
        "n_events": int(labels.sum()),
        "event_rate": float(labels.mean()),
    }
    if exp_config.model_type == "transformer":
        d["transformer_params"] = {
            "architecture": getattr(exp_config, "architecture", "row_token"),
            "embedding_dim": getattr(exp_config, "embedding_dim", 64),
            "d_model": getattr(exp_config, "embedding_dim", 64),
            "num_heads": getattr(exp_config, "num_heads", 4),
            "num_layers": getattr(exp_config, "num_layers", 2),
            "dropout": getattr(exp_config, "dropout", 0.1),
            "learning_rate": exp_config.learning_rate,
            "batch_size": exp_config.batch_size,
            "epochs": exp_config.epochs,
            "early_stopping": exp_config.early_stopping_patience,
            "validation_split": exp_config.validation_split,
        }
    return d


def _save_selected_params(results, output_dir):
    """Table of the configuration chosen in each outer fold.

    Worth reporting in the supplement: if the search lands on very different
    configurations fold to fold, that is itself evidence that the data does not
    determine an optimum, which supports the manuscript's argument more
    directly than any single tuned score does.
    """
    rows = []
    for i, r in enumerate(results, 1):
        row = {"fold": i, "n_features_used": r.get("n_features_used"),
               "elapsed_min": round(r["elapsed_sec"] / 60, 1)}
        row.update({f"param_{k}": v for k, v in (r.get("best_params") or {}).items()})
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(output_dir, "selected_hyperparameters.csv"), index=False)


def summarise_out_of_fold_shap(shap_summaries, output_dir, experiment_name,
                               top_n=25):
    """Aggregate per-fold mean |SHAP| into a mean and SD across folds.

    This is what the editor asked for in point 6: attributions computed on
    held-out data, with the fold-to-fold variability shown rather than a single
    in-sample number.
    """
    frames = [pd.DataFrame({"feature": s["feature"],
                            "mean_abs_shap": s["mean_abs_shap"],
                            "fold": s["fold"]})
              for s in shap_summaries]
    long = pd.concat(frames, ignore_index=True)

    summary = (long.groupby("feature")["mean_abs_shap"]
               .agg(mean="mean", sd="std", n_folds="count", min="min", max="max")
               .reset_index()
               .sort_values("mean", ascending=False))
    # A feature selected in only some folds is itself informative; report it.
    summary["selected_in_n_folds"] = summary["n_folds"]
    summary["rank"] = range(1, len(summary) + 1)

    csv_path = os.path.join(output_dir, "shap_out_of_fold_summary.csv")
    summary.to_csv(csv_path, index=False)

    top = summary.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, max(5, 0.32 * len(top))))
    ax.barh(top["feature"], top["mean"],
            xerr=top["sd"].fillna(0), color="#1f77b4",
            error_kw={"ecolor": "#444444", "elinewidth": 1, "capsize": 3})
    ax.set_xlabel("Mean |SHAP| on held-out folds (error bars: SD across folds)")
    ax.set_title(f"Out-of-fold feature attribution — {experiment_name}",
                 fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "shap_out_of_fold.png"),
                dpi=config.PLOT_CONFIG["figure_dpi"])
    plt.close(fig)

    print(f"[SHAP] Out-of-fold attribution across {len(shap_summaries)} folds "
          f"-> {csv_path}")
    return summary
