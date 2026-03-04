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
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.feature_selection import RFE, SelectKBest, mutual_info_classif
from xgboost import XGBClassifier
import torch

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

from src.evaluation.metrics import compute_fold_metrics, aggregate_fold_metrics
from src.models.xgboost_model import train_xgboost, predict_xgboost, get_feature_importances
from src.models.transformer_training import train_with_validation, predict_transformer
from src.models.logistic_regression import train_logreg, predict_logreg


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
            "n_samples": features.shape[0],
            "n_features_input": features.shape[1],
        }, f, indent=4)

    print(f"\n{'='*60}")
    print(f"Experiment: {exp_config.name}")
    print(f"Model: {exp_config.model_type} | Features: {exp_config.feature_set}")
    print(f"Samples: {features.shape[0]} | Input features: {features.shape[1]}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}")

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

        elif exp_config.model_type == "transformer":
            model, train_losses, val_losses, best_epoch = train_with_validation(
                X_train, y_train, device=device,
                epochs=exp_config.epochs,
                batch_size=exp_config.batch_size,
                validation_split=exp_config.validation_split,
                early_stopping=exp_config.early_stopping_patience,
                learning_rate=exp_config.learning_rate,
                model_size=exp_config.model_size,
            )
            y_pred, probas_1 = predict_transformer(model, X_test, device=device)

        elif exp_config.model_type == "logreg":
            model = train_logreg(X_train, y_train, random_state=config.RANDOM_SEED)
            y_pred, probas_1 = predict_logreg(model, X_test)

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
            feature_names_fold, fold_metrics,
        )

        elapsed = time.time() - t0
        print(f"  Fold {fold_num} done in {elapsed:.1f}s — ROC AUC: {fold_metrics['roc_auc']:.4f}")

    # Aggregated metrics
    agg = aggregate_fold_metrics(all_fold_metrics)
    with open(os.path.join(output_dir, "aggregated_results.json"), "w") as f:
        json.dump(agg, f, indent=4)

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
                         feature_names_fold, fold_metrics):
    """Save model, scaler, selector, predictions, metrics for one fold."""
    prefix = os.path.join(output_dir, f"fold_{fold_num}")

    # Metrics JSON
    serializable = {k: v for k, v in fold_metrics.items()
                    if not isinstance(v, np.ndarray)}
    serializable = {k: (v.item() if isinstance(v, np.generic) else v)
                    for k, v in serializable.items()}
    with open(f"{prefix}.json", "w") as f:
        json.dump(serializable, f, indent=4)

    # Model
    if exp_config.model_type == "transformer":
        torch.save(model.state_dict(), f"{prefix}_model.pt")
    else:
        with open(f"{prefix}_model.pkl", "wb") as f:
            pickle.dump(model, f)

    # Scaler
    with open(f"{prefix}_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    # Feature selector
    if selector is not None:
        with open(f"{prefix}_selector.pkl", "wb") as f:
            pickle.dump(selector, f)

    # Predictions
    preds = {
        "y_true": y_test.tolist(),
        "y_pred": y_pred.tolist(),
        "y_proba": probas_1.tolist(),
        "test_indices": test_idx.tolist(),
    }
    with open(f"{prefix}_predictions.json", "w") as f:
        json.dump(preds, f, indent=4)

    # Feature importances (XGBoost only)
    if exp_config.model_type == "xgboost":
        importances = get_feature_importances(model, feature_names_fold)
        with open(f"{prefix}_feature_importances.txt", "w") as f:
            for rank, (name, imp) in enumerate(importances):
                f.write(f"{rank}\t{name}\t{imp:.6f}\n")


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
