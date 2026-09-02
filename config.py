"""
Central configuration for the AKI-CKD prediction pipeline.

Toggle USE_NONSENSE_DATA to switch between the 1000-row test dataset
and the real server data.  All path helpers, experiment definitions,
and plot settings live here so nothing is hardcoded elsewhere.
"""

import os
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Base paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

USE_NONSENSE_DATA = False          # flip to False on the secure server

# Server paths (only used when USE_NONSENSE_DATA is False)
SERVER_DATA_DIR = "/data/kidney/Sacha/newdata"
SERVER_RAW_DATA_DIR = "/data/kidney/Hing"

RANDOM_SEED = 1202


def get_features_path():
    """Path to the merged features CSV."""
    if USE_NONSENSE_DATA:
        return os.path.join(PROJECT_ROOT, "nonsense_data", "features.csv")
    return os.path.join(SERVER_DATA_DIR, "features.csv")


def get_cohort_path():
    """Path to the cohort CSV."""
    if USE_NONSENSE_DATA:
        return os.path.join(PROJECT_ROOT, "nonsense_data", "cohort.csv")
    return os.path.join(SERVER_DATA_DIR, "cohort.csv")


def get_labs_paths():
    """Return (in_hosp_labs_path, pre_hosp_labs_path) for Alberta score computation.

    pre_hosp_labs_path may be None if the file doesn't exist (e.g. nonsense data).
    """
    if USE_NONSENSE_DATA:
        base = os.path.join(PROJECT_ROOT, "nonsense_data")
        in_hosp = os.path.join(base, "in-hosp_labs.csv")
        pre_hosp = os.path.join(base, "pre-hosp_labs.csv")
        return (in_hosp, pre_hosp if os.path.exists(pre_hosp) else None)
    return (
        os.path.join(SERVER_RAW_DATA_DIR, "in-hosp labs.csv"),
        os.path.join(SERVER_RAW_DATA_DIR, "pre-hosp labs.csv"),
    )


def get_raw_data_dir():
    """Directory containing the raw Hing CSVs (ETL input)."""
    if USE_NONSENSE_DATA:
        return os.path.join(PROJECT_ROOT, "nonsense_data")
    return SERVER_RAW_DATA_DIR


def get_etl_output_dir():
    """Directory where ETL writes intermediate + final CSVs."""
    if USE_NONSENSE_DATA:
        return os.path.join(PROJECT_ROOT, "nonsense_data")
    return SERVER_DATA_DIR


def get_experiments_dir():
    """Root directory for experiment outputs."""
    return os.path.join(PROJECT_ROOT, "experiments", "results", "paper")


# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    name: str
    model_type: str                    # 'xgboost', 'transformer', 'logreg'
    feature_set: str                   # 'alberta_score', 'alberta_raw', 'expanded', 'egfr'
    target: str = "ckd"               # 'ckd' or 'ckdordeath'
    n_features: Optional[int] = None  # None = use all features in the set
    feature_selection_method: Optional[str] = None  # 'rfe' or 'selectkbest'
    rfe_step: float = 0.1
    # Transformer-specific
    model_size: str = "small"          # 'small' or 'large'
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 5e-5
    early_stopping_patience: int = 10
    validation_split: float = 0.15
    # Flags
    perform_shap: bool = True
    perform_umap: bool = True
    sex_subgroups: bool = False


EXPERIMENTS = {
    "xgb_alberta_score": ExperimentConfig(
        name="xgb_alberta_score",
        model_type="xgboost",
        feature_set="alberta_score",
    ),
    "xgb_alberta_raw": ExperimentConfig(
        name="xgb_alberta_raw",
        model_type="xgboost",
        feature_set="alberta_raw",
    ),
    "transformer_alberta_raw": ExperimentConfig(
        name="transformer_alberta_raw",
        model_type="transformer",
        feature_set="alberta_raw",
    ),
    "xgb_expanded": ExperimentConfig(
        name="xgb_expanded",
        model_type="xgboost",
        feature_set="expanded",
        n_features=100,
        feature_selection_method="rfe",
    ),
    "transformer_expanded": ExperimentConfig(
        name="transformer_expanded",
        model_type="transformer",
        feature_set="expanded",
        n_features=100,
        feature_selection_method="selectkbest",
    ),
    # "transformer_egfr": ExperimentConfig(
    #     name="transformer_egfr",
    #     model_type="transformer",
    #     feature_set="egfr",
    # ),
}


# NRI comparison pairs: (baseline_experiment, new_experiment)
NRI_PAIRS = [
    ("xgb_alberta_score", "xgb_alberta_raw"),
    ("xgb_alberta_score", "transformer_alberta_raw"),
    ("xgb_alberta_score", "xgb_expanded"),
    ("xgb_alberta_score", "transformer_expanded"),
    # ("xgb_alberta_raw", "transformer_alberta_raw"),
    # ("xgb_alberta_raw", "xgb_expanded"),
    # ("xgb_alberta_raw", "transformer_expanded"),
    # ("transformer_alberta_raw", "xgb_expanded"),
    # ("transformer_alberta_raw", "transformer_expanded"),
    # ("xgb_expanded", "transformer_expanded"),
]

NRI_THRESHOLDS = [0.2]


# ---------------------------------------------------------------------------
# Plot configuration
# ---------------------------------------------------------------------------

PLOT_CONFIG = {
    "figure_facecolor": "white",
    "axes_facecolor": "white",
    "axes_edgecolor": "black",
    "axes_linewidth": 1.0,
    "font_family": "sans-serif",
    "font_size": 11,
    "axes_label_size": 12,
    "axes_title_size": 14,
    "tick_label_size": 10,
    "legend_font_size": 10,
    "figure_dpi": 300,
}
