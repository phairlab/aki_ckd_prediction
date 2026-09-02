"""
Shared cross-validation engine for all model types.

Runs 10-fold stratified CV, dispatching to the correct model trainer,
computing metrics, saving fold artifacts, and optionally running SHAP
and UMAP analyses.
"""

import os
import json
import time
import pickle
from glob import glob
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.feature_selection import RFE, SelectKBest, mutual_info_classif
from xgboost import XGBClassifier
import torch

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

from src.evaluation.metrics import compute_fold_metrics, aggregate_fold_metrics
from src.models.xgboost_model import train_xgboost, predict_xgboost, get_feature_importances
from src.models.transformer_training import train_with_validation, predict_transformer, oof_train_probas
from src.models.logistic_regression import train_logreg, predict_logreg


def _print_feature_table(feature_names, title):
    """Print a simple text table of feature names."""
    feature_names = list(feature_names)
    if not feature_names:
        return

    print(f"\n{title} ({len(feature_names)} features)")
    print("-" * 110)
    print(f"{'Index':<8} {'Feature'}")
    print("-" * 110)
    for idx, name in enumerate(feature_names):
        print(f"{idx:<8} {name}")
    print("-" * 110)


def run_cross_validation(exp_config, data):
    """Run 10-fold CV for a single experiment.

    Parameters
    ----------
    exp_config : config.ExperimentConfig
    data : dict from preprocess_data()

    Returns
    -------
    str : path to the experiment output directory
    """
    features = data["features"]
    labels = data["labels"]
    feature_names = data["feature_names"]
    features_df = data.get("features_df")
    sex_values = None

    # Preferred source: raw metadata column carried through preprocessing.
    if features_df is not None and "sex" in features_df.columns:
        sex_values = pd.to_numeric(features_df["sex"], errors="coerce").to_numpy()
        if np.all(np.isnan(sex_values)):
            sex_values = None

    # Fallback source: derive from model feature matrix if a sex feature exists.
    if sex_values is None:
        try:
            feature_names_list = [str(x) for x in feature_names.tolist()]
        except AttributeError:
            feature_names_list = [str(x) for x in feature_names]

        if "sex" in feature_names_list:
            sex_idx = feature_names_list.index("sex")
            sex_values = pd.to_numeric(features[:, sex_idx], errors="coerce")
        elif "sex_1" in feature_names_list:
            sex_idx = feature_names_list.index("sex_1")
            sex_values = (np.asarray(features[:, sex_idx], dtype=float) >= 0.5).astype(int)

    elif getattr(exp_config, "sex_subgroups", False):
        print("[Subgroups] Warning: 'sex' column not found in features_df; subgroup reports will be skipped.")

    if getattr(exp_config, "sex_subgroups", False):
        if sex_values is None:
            print("[Subgroups] Warning: could not infer sex labels from metadata or feature matrix; subgroup reports will be skipped.")
        else:
            sex_values = np.asarray(sex_values)
            unique_vals = np.unique(sex_values[~np.isnan(sex_values)]) if sex_values.dtype.kind in {"f"} else np.unique(sex_values)
            print(f"[Subgroups] Sex labels available for {sex_values.shape[0]} rows. Unique values: {unique_vals.tolist()}")

    # Create output directory
    experiments_dir = config.get_experiments_dir()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    folder_name = f"{timestamp}_{exp_config.name}_fold_results"
    output_dir = os.path.join(experiments_dir, folder_name)
    os.makedirs(output_dir, exist_ok=True)

    # Save experiment config
    with open(os.path.join(output_dir, "args.json"), "w") as f:
        json.dump({
            "name": exp_config.name,
            "model_type": exp_config.model_type,
            "feature_set": exp_config.feature_set,
            "target": exp_config.target,
            "n_features": exp_config.n_features,
            "feature_selection_method": exp_config.feature_selection_method,
            "sex_subgroups": getattr(exp_config, "sex_subgroups", False),
            "n_samples": features.shape[0],
            "n_features_input": features.shape[1],
        }, f, indent=4)

    print(f"\n{'='*60}")
    print(f"Experiment: {exp_config.name}")
    print(f"Model: {exp_config.model_type} | Features: {exp_config.feature_set}")
    print(f"Samples: {features.shape[0]} | Input features: {features.shape[1]}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}")

    if exp_config.feature_set == "expanded":
        _print_feature_table(feature_names, "Expanded feature set before RFE")

    # Torch device (for transformer)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 10-fold CV
    cv = KFold(n_splits=10, shuffle=True, random_state=config.RANDOM_SEED)
    all_fold_metrics = []
    all_train_losses = []  # transformer only
    all_val_losses = []

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(features, labels)):
        t0 = time.time()
        fold_num = fold_idx + 1
        print(f"\n--- Fold {fold_num}/10 ---")

        X_train = features[train_idx].astype(float)
        y_train = labels[train_idx].astype("int8")
        X_test = features[test_idx].astype(float)
        y_test = labels[test_idx].astype("int8")

        # Scale
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Handle NaN/inf from scaling (e.g., zero-variance columns)
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

        feature_names_fold = feature_names.copy()
        selector = None

        # Feature selection
        if exp_config.feature_selection_method == "rfe" and exp_config.n_features:
            print(f"  RFE: selecting {exp_config.n_features} features...")
            rfe_estimator = XGBClassifier(random_state=config.RANDOM_SEED)
            selector = RFE(
                estimator=rfe_estimator,
                n_features_to_select=exp_config.n_features,
                step=exp_config.rfe_step,
                verbose=0,
            )
            selector.fit(X_train, y_train)
            X_train = selector.transform(X_train)
            X_test = selector.transform(X_test)
            feature_names_fold = feature_names[selector.support_]

        elif exp_config.feature_selection_method == "selectkbest" and exp_config.n_features:
            print(f"  SelectKBest: selecting {exp_config.n_features} features...")
            selector = SelectKBest(
                score_func=mutual_info_classif,
                k=exp_config.n_features,
            )
            X_train = selector.fit_transform(X_train, y_train)
            X_test = selector.transform(X_test)
            feature_names_fold = feature_names[selector.get_support()]

        # Train + predict
        train_losses, val_losses = [], []

        if exp_config.model_type == "xgboost":
            model = train_xgboost(X_train, y_train, random_state=config.RANDOM_SEED)
            y_pred, probas_1 = predict_xgboost(model, X_test)
            y_train_pred, train_probas_1 = predict_xgboost(model, X_train)

            ######## SANDBOX FOR FIXING RECAL PROBLEMS ########
            # if this is here, then y_train_pred might not match up
            from sklearn.model_selection import cross_val_predict
            from sklearn.base import clone

            train_probas_1 = cross_val_predict(clone(model), X_train, y_train,
                                            cv=5, method='predict_proba')[:, 1]
            ######## SANDBOX FOR FIXING RECAL PROBLEMS ########

        elif exp_config.model_type == "transformer":

            train_kwargs = dict(
                epochs=exp_config.epochs,
                batch_size=exp_config.batch_size,
                validation_split=exp_config.validation_split,
                early_stopping=exp_config.early_stopping_patience,
                learning_rate=exp_config.learning_rate,
                model_size=exp_config.model_size,
            )

            model, train_losses, val_losses, best_epoch = train_with_validation(
                X_train, y_train, device=device, **train_kwargs
            )
            y_pred, probas_1 = predict_transformer(model, X_test, device=device)
            y_train_pred, train_probas_1 = predict_transformer(model, X_train, device=device)

            # NEW: honest training predictions, for the calibration map only
            train_probas_1 = oof_train_probas(X_train, y_train, device=device,
                                            n_splits=5, **train_kwargs)


        elif exp_config.model_type == "logreg":
            model = train_logreg(X_train, y_train, random_state=config.RANDOM_SEED)
            y_pred, probas_1 = predict_logreg(model, X_test)
            y_train_pred, train_probas_1 = predict_logreg(model, X_train)

        else:
            raise ValueError(f"Unknown model_type: {exp_config.model_type}")

        all_train_losses.append(train_losses)
        all_val_losses.append(val_losses)

        # Metrics
        fold_metrics = compute_fold_metrics(y_test, y_pred, probas_1)
        all_fold_metrics.append(fold_metrics)

        # Save fold artifacts
        _save_fold_artifacts(
            output_dir, fold_num, exp_config,
            model, scaler, selector, device,
            y_test, y_pred, probas_1, test_idx,
            y_train, y_train_pred, train_probas_1, train_idx,
            feature_names_fold, fold_metrics,
            None if sex_values is None else sex_values[test_idx],
            None if sex_values is None else sex_values[train_idx],
        )

        elapsed = time.time() - t0
        print(f"  Fold {fold_num} done in {elapsed:.1f}s — ROC AUC: {fold_metrics['roc_auc']:.4f}")

    # Aggregated metrics
    agg = aggregate_fold_metrics(all_fold_metrics)
    with open(os.path.join(output_dir, "aggregated_results.json"), "w") as f:
        json.dump(agg, f, indent=4)

    if getattr(exp_config, "sex_subgroups", False):
        _generate_sex_subgroup_reports(output_dir)

    # Learning curves (transformer only)
    if exp_config.model_type == "transformer":
        _plot_learning_curves(all_train_losses, all_val_losses, output_dir)

    # Full model training for SHAP / feature importances
    if exp_config.model_type == "xgboost":
        _train_full_xgboost(features, labels, feature_names, exp_config, output_dir)

    print(f"\nExperiment {exp_config.name} complete. Results in {output_dir}")
    return output_dir


# ---------------------------------------------------------------------------
# Artifact saving
# ---------------------------------------------------------------------------

def _save_fold_artifacts(output_dir, fold_num, exp_config,
                         model, scaler, selector, device,
                         y_test, y_pred, probas_1, test_idx,
                         y_train, y_train_pred, train_probas_1, train_idx,
                         feature_names_fold, fold_metrics,
                         test_sex=None, train_sex=None):
    """Save model, scaler, selector, predictions, metrics for one fold."""
    test_prefix = os.path.join(output_dir, f"fold_{fold_num}")
    train_prefix = os.path.join(output_dir, f"train_{fold_num}")

    # Metrics JSON
    serializable = {k: v for k, v in fold_metrics.items()
                    if not isinstance(v, np.ndarray)}
    serializable = {k: (v.item() if isinstance(v, np.generic) else v)
                    for k, v in serializable.items()}
    # Test predictions
    with open(f"{test_prefix}.json", "w") as f:
        json.dump(serializable, f, indent=4)

    # Model
    if exp_config.model_type == "transformer":
        torch.save(model.state_dict(), f"{test_prefix}_model.pt")
    else:
        with open(f"{test_prefix}_model.pkl", "wb") as f:
            pickle.dump(model, f)

    # Scaler
    with open(f"{test_prefix}_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    # Feature selector
    if selector is not None:
        with open(f"{test_prefix}_selector.pkl", "wb") as f:
            pickle.dump(selector, f)

    # Predictions
    preds = {
        "y_true": y_test.tolist(),
        "y_pred": y_pred.tolist(),
        "y_proba": probas_1.tolist(),
        "test_indices": test_idx.tolist(),
    }
    if test_sex is not None:
        preds["sex"] = np.asarray(test_sex).tolist()
    with open(f"{test_prefix}_predictions.json", "w") as f:
        json.dump(preds, f, indent=4)

    # Training Predictions
    preds = {
        "y_true": y_train.tolist(),
        "y_pred": y_train_pred.tolist(),
        "y_proba": train_probas_1.tolist(),
        "train_indices": train_idx.tolist(),
    }
    if train_sex is not None:
        preds["sex"] = np.asarray(train_sex).tolist()
    with open(f"{train_prefix}_predictions.json", "w") as f:
        json.dump(preds, f, indent=4)


    # Feature importances (XGBoost only)
    if exp_config.model_type == "xgboost":
        importances = get_feature_importances(model, feature_names_fold)
        with open(f"{test_prefix}_feature_importances.txt", "w") as f:
            for rank, (name, imp) in enumerate(importances):
                f.write(f"{rank}\t{name}\t{imp:.6f}\n")


def _generate_sex_subgroup_reports(output_dir):
    """Generate fold and aggregate metrics for female/male subgroups."""
    group_map = {
        "female": 0,
        "male": 1,
    }
    subgroup_root = os.path.join(output_dir, "sex_subgroups")
    os.makedirs(subgroup_root, exist_ok=True)

    fold_files = sorted(glob(os.path.join(output_dir, "fold_*_predictions.json")))
    if not fold_files:
        print("[Subgroups] No fold prediction files found; skipping subgroup reports.")
        return

    subgroup_fold_metrics = {name: [] for name in group_map}
    subgroup_summary = {
        "n_folds": len(fold_files),
        "groups": {name: {"label_value": val, "n_total": 0, "n_positive": 0, "n_negative": 0,
                           "folds_with_metrics": 0, "folds_skipped": 0}
                   for name, val in group_map.items()}
    }

    for fold_file in fold_files:
        with open(fold_file, "r") as f:
            preds = json.load(f)

        train_file = fold_file.replace("fold_", "train_", 1)
        train_preds = None
        if os.path.exists(train_file):
            with open(train_file, "r") as f:
                train_preds = json.load(f)

        fold_name = os.path.basename(fold_file).replace("_predictions.json", "")
        y_true_all = np.asarray(preds.get("y_true", []))
        y_pred_all = np.asarray(preds.get("y_pred", []))
        y_proba_all = np.asarray(preds.get("y_proba", []))
        sex_all = np.asarray(preds.get("sex", []))

        if sex_all.size == 0:
            print(f"[Subgroups] Missing sex labels in {fold_name}; skipping subgroup reports.")
            return

        for group_name, sex_value in group_map.items():
            group_dir = os.path.join(subgroup_root, group_name)
            os.makedirs(group_dir, exist_ok=True)

            mask = sex_all == sex_value
            y_true = y_true_all[mask]
            y_pred = y_pred_all[mask]
            y_proba = y_proba_all[mask]

            subgroup_summary["groups"][group_name]["n_total"] += int(y_true.size)
            subgroup_summary["groups"][group_name]["n_positive"] += int(np.sum(y_true == 1))
            subgroup_summary["groups"][group_name]["n_negative"] += int(np.sum(y_true == 0))

            filtered_preds = {
                "y_true": y_true.tolist(),
                "y_pred": y_pred.tolist(),
                "y_proba": y_proba.tolist(),
                "sex": np.asarray(sex_all[mask]).tolist(),
                "source_fold": fold_name,
            }
            with open(os.path.join(group_dir, f"{fold_name}_predictions.json"), "w") as f:
                json.dump(filtered_preds, f, indent=4)

            if train_preds is not None and "sex" in train_preds:
                train_y_true_all = np.asarray(train_preds.get("y_true", []))
                train_y_pred_all = np.asarray(train_preds.get("y_pred", []))
                train_y_proba_all = np.asarray(train_preds.get("y_proba", []))
                train_sex_all = np.asarray(train_preds.get("sex", []))
                train_mask = train_sex_all == sex_value

                filtered_train_preds = {
                    "y_true": train_y_true_all[train_mask].tolist(),
                    "y_pred": train_y_pred_all[train_mask].tolist(),
                    "y_proba": train_y_proba_all[train_mask].tolist(),
                    "sex": train_sex_all[train_mask].tolist(),
                    "source_train_file": os.path.basename(train_file),
                }
                train_name = os.path.basename(train_file)
                with open(os.path.join(group_dir, train_name), "w") as f:
                    json.dump(filtered_train_preds, f, indent=4)

            fold_output_path = os.path.join(group_dir, f"{fold_name}.json")
            # Metrics requiring ROC/PR need both classes represented in the subgroup fold.
            if y_true.size < 2 or len(np.unique(y_true)) < 2:
                subgroup_summary["groups"][group_name]["folds_skipped"] += 1
                skipped = {
                    "status": "skipped",
                    "reason": "insufficient class diversity in subgroup fold",
                    "n": int(y_true.size),
                    "n_positive": int(np.sum(y_true == 1)),
                    "n_negative": int(np.sum(y_true == 0)),
                }
                with open(fold_output_path, "w") as f:
                    json.dump(skipped, f, indent=4)
                continue

            metrics = compute_fold_metrics(y_true, y_pred, y_proba)
            subgroup_fold_metrics[group_name].append(metrics)
            subgroup_summary["groups"][group_name]["folds_with_metrics"] += 1

            serializable = {k: v for k, v in metrics.items() if not isinstance(v, np.ndarray)}
            serializable = {
                k: (v.item() if isinstance(v, np.generic) else v)
                for k, v in serializable.items()
            }
            with open(fold_output_path, "w") as f:
                json.dump(serializable, f, indent=4)

    for group_name, fold_metrics in subgroup_fold_metrics.items():
        group_dir = os.path.join(subgroup_root, group_name)
        if fold_metrics:
            agg = aggregate_fold_metrics(fold_metrics)
            with open(os.path.join(group_dir, "aggregated_results.json"), "w") as f:
                json.dump(agg, f, indent=4)
        else:
            with open(os.path.join(group_dir, "aggregated_results.json"), "w") as f:
                json.dump(
                    {
                        "status": "skipped",
                        "reason": "no fold had enough class diversity for metric computation",
                    },
                    f,
                    indent=4,
                )

    with open(os.path.join(subgroup_root, "summary.json"), "w") as f:
        json.dump(subgroup_summary, f, indent=4)
    print(f"[Subgroups] Sex-specific reports saved to {subgroup_root}")


# ---------------------------------------------------------------------------
# Full-dataset model (for SHAP)
# ---------------------------------------------------------------------------

def _train_full_xgboost(features, labels, feature_names, exp_config, output_dir):
    """Train XGBoost on entire dataset and save feature importances."""
    print("\nTraining full XGBoost model on entire dataset...")
    X_full = features.astype(float)
    y_full = labels.astype("int8")

    scaler = StandardScaler()
    X_full_scaled = scaler.fit_transform(X_full)

    feature_names_full = feature_names.copy()

    if exp_config.feature_selection_method == "rfe" and exp_config.n_features:
        rfe_est = XGBClassifier(random_state=config.RANDOM_SEED)
        rfe = RFE(estimator=rfe_est, n_features_to_select=exp_config.n_features,
                   step=exp_config.rfe_step, verbose=0)
        rfe.fit(X_full_scaled, y_full)
        X_full_scaled = rfe.transform(X_full_scaled)
        feature_names_full = feature_names[rfe.support_]
        with open(os.path.join(output_dir, "full_rfe.pkl"), "wb") as f:
            pickle.dump(rfe, f)

    model = train_xgboost(X_full_scaled, y_full, random_state=config.RANDOM_SEED)

    with open(os.path.join(output_dir, "full_model.pkl"), "wb") as f:
        pickle.dump(model, f)
    with open(os.path.join(output_dir, "full_scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    importances = get_feature_importances(model, feature_names_full)
    with open(os.path.join(output_dir, "full_feature_importances.txt"), "w") as f:
        for rank, (name, imp) in enumerate(importances):
            f.write(f"{rank}\t{name}\t{imp:.6f}\n")

    print(f"  Full model saved. Top 5 features:")
    for name, imp in importances[:5]:
        print(f"    {name}: {imp:.4f}")


# ---------------------------------------------------------------------------
# Learning curves plot
# ---------------------------------------------------------------------------

def _plot_learning_curves(all_train_losses, all_val_losses, output_dir):
    """Plot training/validation loss curves across all folds."""
    valid_train = [l for l in all_train_losses if l]
    valid_val = [l for l in all_val_losses if l]
    if not valid_train:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (tl, vl) in enumerate(zip(valid_train, valid_val)):
        epochs = range(1, len(tl) + 1)
        ax.plot(epochs, tl, "b-", alpha=0.2)
        ax.plot(epochs, vl, "r-", alpha=0.2)

    # Mean curves
    min_len = min(len(l) for l in valid_train)
    mean_train = np.mean([l[:min_len] for l in valid_train], axis=0)
    mean_val = np.mean([l[:min_len] for l in valid_val], axis=0)
    ax.plot(range(1, min_len + 1), mean_train, "b-", lw=2, label="Mean Train")
    ax.plot(range(1, min_len + 1), mean_val, "r-", lw=2, label="Mean Val")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Learning Curves Across Folds")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "learning_curves.png"), dpi=config.PLOT_CONFIG["figure_dpi"])
    plt.close(fig)
