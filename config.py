"""
Central configuration for the AKI-CKD prediction pipeline.

Everything that varies between the local smoke test and the secure-server run
lives here: data paths, experiment definitions, tuning budgets, decision
thresholds and plot settings.
"""

import os
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Base paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Data source. Exactly one of these is true at a time; run_pipeline.py sets
# them from --smoke / --nonsense / --server.
#
#   smoke    : coherent synthetic data from src/make_smoke_data.py. Patient ids
#             link across files, so the WHOLE pipeline runs end to end. Use this
#             to verify a change before spending GPU hours.
#   nonsense : the legacy column-shuffled data. Every column was permuted
#             independently, so the join key is destroyed and preprocessing
#             drops almost every patient. Kept only for backwards compatibility;
#             prefer --smoke.
#   server   : the real data.
USE_NONSENSE_DATA = False
USE_SMOKE_DATA = True

SMOKE_DATA_DIR = os.path.join(PROJECT_ROOT, "smoke_data")

SERVER_DATA_DIR = "/data/kidney/Sacha/newdata"
SERVER_RAW_DATA_DIR = "/data/kidney/Hing"

# Post-discharge outpatient labs, needed for the outcome-ascertainment analysis
# (editor point 3).  None until `python src/probe_server_data.py --server`
# confirms such an extract exists; set it to that file's path once it does.
# The analysis reports clearly that it was skipped rather than failing silently.
FOLLOWUP_LABS_PATH: Optional[str] = None

RANDOM_SEED = 1202
N_OUTER_FOLDS = 10


def get_features_path():
    if USE_SMOKE_DATA:
        return os.path.join(SMOKE_DATA_DIR, "features.csv")
    if USE_NONSENSE_DATA:
        return os.path.join(PROJECT_ROOT, "nonsense_data", "features.csv")
    return os.path.join(SERVER_DATA_DIR, "features.csv")


def get_cohort_path():
    if USE_SMOKE_DATA:
        return os.path.join(SMOKE_DATA_DIR, "cohort.csv")
    if USE_NONSENSE_DATA:
        return os.path.join(PROJECT_ROOT, "nonsense_data", "cohort.csv")
    return os.path.join(SERVER_DATA_DIR, "cohort.csv")


def get_labs_paths():
    """(in_hospital_labs, pre_hospital_labs). The second may be None."""
    if USE_SMOKE_DATA:
        return (os.path.join(SMOKE_DATA_DIR, "in-hosp labs.csv"),
                os.path.join(SMOKE_DATA_DIR, "pre-hosp labs.csv"))
    if USE_NONSENSE_DATA:
        base = os.path.join(PROJECT_ROOT, "nonsense_data")
        pre = os.path.join(base, "pre-hosp_labs.csv")
        return os.path.join(base, "in-hosp_labs.csv"), (pre if os.path.exists(pre) else None)
    return (os.path.join(SERVER_RAW_DATA_DIR, "in-hosp labs.csv"),
            os.path.join(SERVER_RAW_DATA_DIR, "pre-hosp labs.csv"))


def get_followup_labs_path():
    """Post-discharge labs, or None if no such extract is configured."""
    if USE_SMOKE_DATA:
        path = os.path.join(SMOKE_DATA_DIR, "post-discharge labs.csv")
        return path if os.path.exists(path) else None
    if FOLLOWUP_LABS_PATH and os.path.exists(FOLLOWUP_LABS_PATH):
        return FOLLOWUP_LABS_PATH
    return None


def get_raw_data_dir():
    if USE_SMOKE_DATA:
        return SMOKE_DATA_DIR
    if USE_NONSENSE_DATA:
        return os.path.join(PROJECT_ROOT, "nonsense_data")
    return SERVER_RAW_DATA_DIR


def get_etl_output_dir():
    if USE_SMOKE_DATA:
        return SMOKE_DATA_DIR
    if USE_NONSENSE_DATA:
        return os.path.join(PROJECT_ROOT, "nonsense_data")
    return SERVER_DATA_DIR


def get_experiments_dir():
    # Smoke runs are kept out of the paper results tree so a test run can never
    # be mistaken for, or picked up alongside, the reported experiments.
    if USE_SMOKE_DATA:
        return os.path.join(PROJECT_ROOT, "experiments", "results", "smoke")
    return os.path.join(PROJECT_ROOT, "experiments", "results", "paper")


def get_reports_dir():
    """Manuscript-ready tables and figures land here."""
    if USE_SMOKE_DATA:
        return os.path.join(PROJECT_ROOT, "reports", "smoke")
    return os.path.join(PROJECT_ROOT, "reports")


# ---------------------------------------------------------------------------
# ETL
# ---------------------------------------------------------------------------

# Group laboratory measurements by clinical entity rather than by raw TEST_NM.
# Addresses editor point 5: random glucose entered the candidate set under six
# in-hospital and ten pre-index name strings, each becoming four feature
# columns.  See src/lab_normalization.py; audit the mapping with
# `python src/probe_server_data.py --server` before trusting a refit.
NORMALIZE_LAB_NAMES = True

# Albuminuria component: include urine protein:creatinine as a last-resort
# fallback when neither ACR nor dipstick is available.
#
# DO NOT ENABLE THIS FOR THE RESUBMISSION. It is dormant insurance, not part of
# the planned analysis.
#
# James et al. define the albuminuria component on ACR or urine dipstick only.
# Multimedia Appendix 2, section A1 states it plainly: "albuminuria status is
# determined using either albumin:creatinine ratio or urine dipstick
# measurements". Substituting a different assay means the model being compared
# against is no longer the James score, which dissolves the like-for-like
# comparison the whole paper rests on.
#
# The manuscript also argues against it directly. From the Discussion:
#
#     "unmeasured albuminuria carries a defined one-point weight, which is how
#      76.0% of this cohort was scored, creating minimal implementation and
#      interpretation barriers: no complex EHR integration, NO IMPUTATION
#      PIPELINES for hundreds of potentially missing features..."
#
# A uPCR fallback is exactly such a pipeline. "Unmeasured" is not missing data
# in this score; it is a modelled state with a deliberate weight, and its
# computability without extra inputs is presented as a REASON to prefer the
# score. Filling it in undermines that argument.
#
# What is worth keeping is the NUMBER, not the arm. Section 5 of
# src/audit_james_inputs.py reports that 229 patients (4.9% of the cohort) hold
# a uPCR result while scored "unmeasured", and where they would land. That
# answers a reviewer who asks why data on hand went unused, without changing
# the score. Enable the arm only if a reviewer explicitly asks to see it.
ALBUMINURIA_INCLUDE_UPCR = False

# Number of distinct lab entities retained per source before crosstabbing.
# Raised from the original 34/50 because normalization collapses variants, so
# the same cut on raw names would now discard genuinely distinct analytes.
TOP_LABS_IN_HOSPITAL = 40
TOP_LABS_PRE_INDEX = 60


# ---------------------------------------------------------------------------
# Decision thresholds
# ---------------------------------------------------------------------------

# 20% is where nephrology referral is considered (Acharya et al. 2025) and
# remains the primary reporting threshold.
PRIMARY_THRESHOLD = 0.20

# Editor point 7b: report threshold metrics and reclassification across a range
# rather than at a single cutoff.
THRESHOLD_SWEEP = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]

# Editor point 7c: a pre-specified equivalence margin, declared here BEFORE the
# analysis runs so the claim is not fitted to the result.
#
# 0.02 AUROC is the margin below which a difference in discrimination is taken
# to be clinically unimportant for this decision. Rationale for the response
# letter: the externally validated James score spans AUC 0.81-0.87 across its
# derivation, validation and Grampian cohorts, so a 0.02 difference is well
# inside the between-cohort variation of the score's own established
# performance and could not change a referral policy.
EQUIVALENCE_MARGIN_AUROC = 0.02

# NRI margin, on the same logic: a net reclassification of fewer than 5 patients
# per 100 at the referral threshold does not change nephrology capacity planning.
EQUIVALENCE_MARGIN_NRI = 0.05


# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    name: str
    model_type: str                    # 'xgboost' | 'transformer' | 'logreg'
    feature_set: str                   # 'james_score' | 'james_raw' | 'expanded' | 'egfr'
    target: str = "ckd"                # 'ckd' | 'ckdordeath'

    # Feature selection
    n_features: Optional[int] = None
    feature_selection_method: Optional[str] = None   # 'selectkbest' | 'rfe'
    rfe_step: float = 0.1

    # Nested hyperparameter search (editor point 1)
    tune: bool = True
    n_trials: int = 60
    inner_folds: int = 5
    recalibration_inner_folds: int = 5

    # Transformer architecture defaults; overridden by the search when tune=True
    architecture: str = "row_token"
    embedding_dim: int = 64
    num_heads: int = 4
    num_layers: int = 2
    dropout: float = 0.1
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 5e-5
    early_stopping_patience: int = 10
    validation_split: float = 0.15

    # Analyses
    perform_shap: bool = True
    perform_umap: bool = False        # editor: UMAP need not be recomputed
    shap_max_samples: int = 200
    sex_subgroups: bool = False


# ---------------------------------------------------------------------------
# Tuning budget profiles
# ---------------------------------------------------------------------------
# Applied by run_pipeline.py via --tuning. Transformer trials are far more
# expensive than XGBoost trials (each is a network fit x inner_folds), so the
# two are budgeted separately.

TUNING_PROFILES = {
    # Smoke test: proves the plumbing works, produces no reportable numbers.
    # Epochs and recalibration folds are cut hard so a full seven-experiment
    # run finishes in a couple of minutes on a laptop CPU.
    "smoke":  {"xgb_trials": 4,   "tf_trials": 2,  "xgb_inner": 2, "tf_inner": 2,
               "epochs": 6,   "recal_inner": 2},
    # A few hours on 4 GPUs.
    "fast":   {"xgb_trials": 30,  "tf_trials": 15, "xgb_inner": 5, "tf_inner": 3,
               "epochs": 100, "recal_inner": 5},
    # Overnight on 4 GPUs. This is the profile intended for the resubmission.
    "full":   {"xgb_trials": 100, "tf_trials": 40, "xgb_inner": 5, "tf_inner": 3,
               "epochs": 100, "recal_inner": 5},
    # Maximum defensibility if there is time to spare.
    "deep":   {"xgb_trials": 200, "tf_trials": 80, "xgb_inner": 5, "tf_inner": 5,
               "epochs": 150, "recal_inner": 5},
}

DEFAULT_TUNING_PROFILE = "full"


def apply_tuning_profile(exp: ExperimentConfig, profile: str) -> ExperimentConfig:
    """Return a copy of `exp` with the profile's trial budget applied."""
    from dataclasses import replace
    if profile in ("off", "none"):
        return replace(exp, tune=False)
    if profile not in TUNING_PROFILES:
        raise ValueError(f"Unknown tuning profile {profile!r}. "
                         f"Choose from {sorted(TUNING_PROFILES) + ['off']}")
    p = TUNING_PROFILES[profile]
    common = {"tune": True,
              "epochs": p.get("epochs", exp.epochs),
              "recalibration_inner_folds": p.get("recal_inner",
                                                 exp.recalibration_inner_folds)}
    if exp.model_type == "transformer":
        return replace(exp, n_trials=p["tf_trials"], inner_folds=p["tf_inner"], **common)
    return replace(exp, n_trials=p["xgb_trials"], inner_folds=p["xgb_inner"], **common)


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------
# Naming follows the manuscript: the score is the JAMES score. The repository
# previously called it the Alberta score; LEGACY_EXPERIMENT_ALIASES below keeps
# the old names working.

EXPERIMENTS = {
    # -- Baselines (editor point 4) -------------------------------------------
    # Primary baseline for the resubmission. Fitting an untuned XGBoost to a
    # single scalar produced a step function whose step edges relative to 0.20
    # determined every threshold metric and every NRI in the paper. A logistic
    # regression on the same scalar is monotone and smooth, so the reference
    # model is no longer an artifact of tree quantisation.
    "logreg_james_score": ExperimentConfig(
        name="logreg_james_score",
        model_type="logreg",
        feature_set="james_score",
        n_trials=30,
    ),
    # Retained so the resubmission can be compared like-for-like with the
    # submitted results, and so the response letter can quantify how much of
    # the original finding was baseline specification.
    "xgb_james_score": ExperimentConfig(
        name="xgb_james_score",
        model_type="xgboost",
        feature_set="james_score",
    ),

    # -- Simple features ------------------------------------------------------
    "logreg_james_raw": ExperimentConfig(
        name="logreg_james_raw",
        model_type="logreg",
        feature_set="james_raw",
        n_trials=30,
    ),
    "xgb_james_raw": ExperimentConfig(
        name="xgb_james_raw",
        model_type="xgboost",
        feature_set="james_raw",
    ),
    "transformer_james_raw": ExperimentConfig(
        name="transformer_james_raw",
        model_type="transformer",
        feature_set="james_raw",
    ),

    # -- Expanded features ----------------------------------------------------
    # Both architectures now use the SAME model-agnostic selector, so a
    # difference between them is attributable to the model rather than to RFE
    # versus SelectKBest.
    "xgb_expanded": ExperimentConfig(
        name="xgb_expanded",
        model_type="xgboost",
        feature_set="expanded",
        n_features=100,
        feature_selection_method="selectkbest",
    ),
    "transformer_expanded": ExperimentConfig(
        name="transformer_expanded",
        model_type="transformer",
        feature_set="expanded",
        n_features=100,
        feature_selection_method="selectkbest",
    ),
}


# Sensitivity to the number of selected features. k=100 was unjustified in the
# submission and a reviewer is likely to ask why. Enable with
# `--experiments-set sensitivity_k`.
SENSITIVITY_K_EXPERIMENTS = {
    f"xgb_expanded_k{k}": ExperimentConfig(
        name=f"xgb_expanded_k{k}",
        model_type="xgboost",
        feature_set="expanded",
        n_features=k,
        feature_selection_method="selectkbest",
        perform_shap=False,
    )
    for k in (10, 25, 50, 200)
}

# Retains the original RFE selector for XGBoost, so the response letter can say
# what changing the selector cost or gained.
SENSITIVITY_SELECTOR_EXPERIMENTS = {
    "xgb_expanded_rfe": ExperimentConfig(
        name="xgb_expanded_rfe",
        model_type="xgboost",
        feature_set="expanded",
        n_features=100,
        feature_selection_method="rfe",
        perform_shap=False,
    ),
}

ALL_EXPERIMENTS = {**EXPERIMENTS, **SENSITIVITY_K_EXPERIMENTS,
                   **SENSITIVITY_SELECTOR_EXPERIMENTS}

EXPERIMENT_SETS = {
    "primary": list(EXPERIMENTS),
    "sensitivity_k": list(SENSITIVITY_K_EXPERIMENTS),
    "sensitivity_selector": list(SENSITIVITY_SELECTOR_EXPERIMENTS),
    "all": list(ALL_EXPERIMENTS),
}

LEGACY_EXPERIMENT_ALIASES = {
    "xgb_alberta_score": "xgb_james_score",
    "xgb_alberta_raw": "xgb_james_raw",
    "transformer_alberta_raw": "transformer_james_raw",
}


def resolve_experiment_name(name: str) -> str:
    """Map a pre-rename experiment name onto its current one."""
    return LEGACY_EXPERIMENT_ALIASES.get(name, name)


# Display labels used in overlay plots and manuscript tables.
EXPERIMENT_LABELS = {
    "logreg_james_score":    "Logistic Regression + James Score",
    "xgb_james_score":       "XGBoost + James Score",
    "logreg_james_raw":      "Logistic Regression + James Features",
    "xgb_james_raw":         "XGBoost + James Features",
    "transformer_james_raw": "Transformer + James Features",
    "xgb_expanded":          "XGBoost + Top 100 Expanded Features",
    "transformer_expanded":  "Transformer + Top 100 Expanded Features",
}


# ---------------------------------------------------------------------------
# Model comparisons
# ---------------------------------------------------------------------------
# The primary baseline is now the logistic-regression James score. The XGBoost
# James score comparisons are retained so the resubmission can show how the
# original conclusion moves when the baseline artifact is removed.

NRI_BASELINE = "logreg_james_score"

NRI_PAIRS = [
    ("logreg_james_score", "xgb_james_score"),
    ("logreg_james_score", "logreg_james_raw"),
    ("logreg_james_score", "xgb_james_raw"),
    ("logreg_james_score", "transformer_james_raw"),
    ("logreg_james_score", "xgb_expanded"),
    ("logreg_james_score", "transformer_expanded"),
    # Continuity with the submitted Table 5
    ("xgb_james_score", "xgb_james_raw"),
    ("xgb_james_score", "transformer_james_raw"),
    ("xgb_james_score", "xgb_expanded"),
    ("xgb_james_score", "transformer_expanded"),
]

NRI_THRESHOLDS = [PRIMARY_THRESHOLD]


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

N_BOOTSTRAP = 2000
CI_LEVEL = 0.95


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
