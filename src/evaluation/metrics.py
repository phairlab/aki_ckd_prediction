"""
Shared metric computation for cross-validation folds.

Computes confusion-matrix-derived metrics, ROC/PRC curves, and
optimal thresholds.  Used by the CV engine for all model types.
"""

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    roc_curve,
    auc,
    precision_recall_curve,
)


def compute_fold_metrics(y_test, y_pred, probas_1):
    """Compute all metrics for a single CV fold.

    Parameters
    ----------
    y_test : array-like
        True binary labels.
    y_pred : array-like
        Predicted binary labels.
    probas_1 : array-like
        Predicted probability of the positive class.

    Returns
    -------
    dict with keys: accuracy, sensitivity, specificity, ppv, npv, f1,
    roc_auc, prc_auc, best_threshold_roc, best_threshold_prc,
    fpr, tpr, precision_curve, recall_curve.
    """
    cm = confusion_matrix(y_test, y_pred)
    TN, FP, FN, TP = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    P = TP + FN
    N = TN + FP

    accuracy = (TP + TN) / (P + N)
    sensitivity = TP / P if P > 0 else 0.0
    specificity = TN / N if N > 0 else 0.0
    ppv = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    npv = TN / (TN + FN) if (TN + FN) > 0 else 0.0
    f1 = f1_score(y_test, y_pred)

    # ROC curve + best threshold (maximize geometric mean)
    fpr, tpr, roc_thresholds = roc_curve(y_test, probas_1)
    gmeans = tpr * (1 - fpr)
    ix_roc = np.argmax(gmeans)
    best_threshold_roc = float(roc_thresholds[ix_roc])
    roc_auc = auc(fpr, tpr)

    # Precision-Recall curve + best threshold (maximize F-score)
    precision_curve, recall_curve, prc_thresholds = precision_recall_curve(
        y_test, probas_1
    )
    fscore = (2 * precision_curve * recall_curve) / (
        precision_curve + recall_curve + 1e-10
    )
    ix_prc = np.argmax(fscore)
    best_threshold_prc = (
        float(prc_thresholds[ix_prc]) if ix_prc < len(prc_thresholds) else 0.5
    )
    prc_auc = auc(recall_curve, precision_curve)

    return {
        "accuracy": float(accuracy),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "ppv": float(ppv),
        "npv": float(npv),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "prc_auc": float(prc_auc),
        "best_threshold_roc": best_threshold_roc,
        "best_threshold_prc": best_threshold_prc,
        # Curves (for interpolation / averaging across folds)
        "fpr": fpr,
        "tpr": tpr,
        "precision_curve": precision_curve,
        "recall_curve": recall_curve,
    }


def aggregate_fold_metrics(fold_metrics_list):
    """Compute mean +/- std across folds and print summary.

    Parameters
    ----------
    fold_metrics_list : list of dict
        Each dict is the output of compute_fold_metrics().

    Returns
    -------
    dict with 'mean_<metric>' and 'std_<metric>' keys.
    """
    scalar_keys = [
        "accuracy", "sensitivity", "specificity", "ppv", "npv",
        "f1", "roc_auc", "prc_auc",
        "best_threshold_roc", "best_threshold_prc",
    ]
    agg = {}
    for key in scalar_keys:
        values = [m[key] for m in fold_metrics_list]
        agg[f"mean_{key}"] = float(np.mean(values))
        agg[f"std_{key}"] = float(np.std(values))

    print("\nAggregated Results Over All Folds:")
    for key in scalar_keys:
        print(f"  {key:>22s}: {agg[f'mean_{key}']:.4f} +/- {agg[f'std_{key}']:.4f}")

    return agg
