"""
SHAP analysis for AKI-CKD prediction models.

Provides TreeExplainer (XGBoost) and KernelExplainer (transformer/neural net)
analyses with publication-quality plots.
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config

from src.plot_style import setup_global_style, get_shap_colormap, style_axis


# Feature name mapping for cleaner display in plots
FEATURE_NAME_MAPPING = {
    # Creatinine
    "discharge_creatinine_raw": "Discharge Creatinine",
    "baseline_creatinine_raw": "Baseline Creatinine",
    "discharge_creatinine_points": "Discharge Creatinine (points)",
    "baseline_creatinine_points": "Baseline Creatinine (points)",
    # Sex
    "sex_0": "Sex: Male",
    "sex_1": "Sex: Female",
    "sex": "Sex",
    "sex_points": "Sex (points)",
    # Age
    "age_admit": "Age at Admission",
    "age_admit_points": "Age (points)",
    # Albuminuria
    "albuminuria_status_raw_unmeasured": "Albuminuria: Unmeasured",
    "albuminuria_status_raw_mild": "Albuminuria: Mild",
    "albuminuria_status_raw_normal": "Albuminuria: Normal",
    "albuminuria_status_raw_heavy": "Albuminuria: Heavy",
    "albuminuria_status_raw": "Albuminuria Status",
    "albuminuria_status_points": "Albuminuria (points)",
    # AKI stage
    "highest_stage": "Highest AKI Stage",
    "highest_stage_1": "AKI Stage 1",
    "highest_stage_2": "AKI Stage 2",
    "highest_stage_3": "AKI Stage 3",
    "highest_stage_points": "AKI Stage (points)",
    # Alberta score
    "alberta_score": "Alberta Score",
}


def clean_feature_names(feature_names):
    """Apply the feature name mapping to make names more readable."""
    if hasattr(feature_names, "tolist"):
        feature_names = feature_names.tolist()
    return [FEATURE_NAME_MAPPING.get(name, name) for name in feature_names]


def perform_shap_analysis_tree(classifier, X_data, feature_names, output_dir,
                               max_display=20,
                               beeswarm_title="SHAP Feature Impact",
                               bar_title="SHAP Feature Importance"):
    """SHAP analysis for tree-based models using TreeExplainer.

    Parameters
    ----------
    classifier : trained tree model (XGBoost, etc.)
    X_data : np.ndarray
    feature_names : array-like
    output_dir : str
    max_display : int
    beeswarm_title, bar_title : str

    Returns
    -------
    shap.Explanation
    """
    print("[SHAP] Running TreeExplainer...")
    setup_global_style()
    dpi = config.PLOT_CONFIG["figure_dpi"]

    cleaned_names = clean_feature_names(feature_names)
    shap_df = pd.DataFrame(X_data, columns=cleaned_names)

    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer(shap_df)

    cmap = get_shap_colormap()

    # Beeswarm plot
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.plots.beeswarm(shap_values, max_display=max_display,
                        color=cmap, show=False, plot_size=None)
    ax = plt.gca()
    style_axis(ax, title=beeswarm_title,
               xlabel="SHAP value (impact on model output)", ylabel="")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "shap_beeswarm.png"),
                dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # Bar plot
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.plots.bar(shap_values, max_display=max_display, show=False)
    ax = plt.gca()
    style_axis(ax, title=bar_title, xlabel="mean(|SHAP value|)", ylabel="")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "shap_bar.png"),
                dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # Save values
    with open(os.path.join(output_dir, "shap_values.pkl"), "wb") as f:
        pickle.dump(shap_values, f)

    print(f"[SHAP] TreeExplainer done. Plots saved to {output_dir}")
    return shap_values


def perform_shap_analysis_kernel(model_predict_fn, X_data, feature_names,
                                 output_dir, background_samples=100,
                                 samples_to_explain=100, max_display=20,
                                 beeswarm_title="SHAP Feature Impact",
                                 bar_title="SHAP Feature Importance"):
    """SHAP analysis using KernelExplainer (for transformers / neural nets).

    Parameters
    ----------
    model_predict_fn : callable — takes X, returns (n_samples, n_classes) probabilities
    X_data : np.ndarray (scaled/preprocessed)
    feature_names : array-like
    output_dir : str
    background_samples : int — number of background samples for the explainer
    samples_to_explain : int — number of samples to compute SHAP values for
    max_display : int
    beeswarm_title, bar_title : str

    Returns
    -------
    np.ndarray — SHAP values
    """
    print("[SHAP] Running KernelExplainer...")
    setup_global_style()
    dpi = config.PLOT_CONFIG["figure_dpi"]

    background = X_data[:background_samples]
    sample_to_explain = X_data[:samples_to_explain]

    print(f"  Background: {background.shape}, Explain: {sample_to_explain.shape}")

    explainer = shap.KernelExplainer(model_predict_fn, background)
    print("  Computing SHAP values (this may take a while)...")
    shap_values = explainer.shap_values(sample_to_explain)

    cleaned_names = clean_feature_names(feature_names)

    # Beeswarm (positive class)
    fig = plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values[1], sample_to_explain,
                      feature_names=cleaned_names, max_display=max_display,
                      show=False)
    ax = plt.gca()
    style_axis(ax, title=beeswarm_title, xlabel="", ylabel="")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "shap_beeswarm.png"),
                dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # Bar plot (positive class)
    fig = plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values[1], sample_to_explain,
                      feature_names=cleaned_names, plot_type="bar",
                      max_display=max_display, show=False)
    ax = plt.gca()
    style_axis(ax, title=bar_title, xlabel="", ylabel="")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "shap_bar.png"),
                dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # Save values + data for regeneration
    for name, obj in [("shap_values.pkl", shap_values),
                      ("shap_X_data.pkl", sample_to_explain),
                      ("shap_feature_names.pkl", cleaned_names)]:
        with open(os.path.join(output_dir, name), "wb") as f:
            pickle.dump(obj, f)

    print(f"[SHAP] KernelExplainer done. Plots saved to {output_dir}")
    return shap_values


def load_shap_values(filepath):
    """Load previously computed SHAP values from a pickle file."""
    with open(filepath, "rb") as f:
        return pickle.load(f)
