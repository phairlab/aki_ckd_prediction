"""
Data preprocessing for the AKI-CKD prediction pipeline.

Loads features.csv and the lab extracts, computes the James score components,
applies the cohort filters, selects features for the requested experiment, and
returns everything the cross-validation engine needs.

Changes for the resubmission
----------------------------
1. NO IMPUTATION HAPPENS HERE.  The original module filled missing values with
   column medians computed over all 4,687 patients and then handed the result
   to the CV split, so every test fold's values had contributed to its own
   imputation.  This module now returns the feature matrix with NaN intact
   plus an *imputation plan* -- a per-column rule -- which cross_validation.py
   fits on each training fold and applies to the held-out fold.

2. The feature matrix is now IDENTICAL for every model type.  Previously the
   transformer path one-hot encoded low-cardinality columns and the XGBoost
   path did not, so the two models were compared on different inputs and any
   difference confounded architecture with encoding.  Encoding is now shared.

3. Cohort attrition is returned as structured data (`cohort_log`) rather than
   only printed, so the flow diagram and the response letter can be generated
   from the run rather than transcribed by hand.

4. `patient_id` is carried through to the fold prediction files, which lets the
   evaluation suite verify that two experiments scored the same patients in the
   same order, and lets the ascertainment and competing-risk analyses join back
   to the cohort.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from james_score_helpers import (
    age_mapping, stage_mapping,
    baseline_creatinine_mapping, discharge_creatinine_mapping,
    albuminuria_status_mapping,
    get_baseline_creatinine, get_discharge_creatinine, get_albuminuria_status,
    report_unknown_units, report_unknown_pcr_units,
)

import config


# ---------------------------------------------------------------------------
# Cohort logging
# ---------------------------------------------------------------------------

class CohortLog:
    """Record cohort size after each filtering step.

    Feeds both the console trace and `reports/cohort_flow.csv`, so Figure 1 and
    the response letter can be regenerated from an actual run.
    """

    def __init__(self):
        self.steps = []

    def record(self, description, df):
        n = len(df)
        removed = (self.steps[-1]["n_remaining"] - n) if self.steps else 0
        self.steps.append({
            "step": len(self.steps) + 1,
            "description": description,
            "n_remaining": n,
            "n_removed": removed,
        })
        suffix = f"  (-{removed})" if removed else ""
        print(f"[Cohort] {description} -> {n} patients remaining{suffix}")
        return df

    def to_frame(self):
        return pd.DataFrame(self.steps)

    def save(self, output_dir, filename="cohort_flow.csv"):
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, filename)
        self.to_frame().to_csv(path, index=False)
        print(f"[Cohort] Attrition table written to {path}")
        return path


# ---------------------------------------------------------------------------
# Fold-local imputation
# ---------------------------------------------------------------------------

# Column-name patterns -> imputation rule.  The rule is chosen from the column
# name (a count of something absent is genuinely 0; a mean lab value is not),
# but the VALUE used is always computed from the training fold only.

ZERO_FILL_PATTERNS = (
    "records_count", "consult_count", "labs_count",
    "pre-index_labs_count", "pre-index_medication",
    "pre-index_vars", "pre-index_count",
    "stage1", "stage2", "stage3",
)

MEDIAN_FILL_PATTERNS = (
    "records_mean", "records_min", "records_max",
    "labs_mean", "labs_min", "labs_max",
    "pre-index_labs_mean", "pre-index_labs_min", "pre-index_labs_max",
    "pre-index_mean", "pre-index_min", "pre-index_max",
)


def build_imputation_plan(feature_names):
    """Map each column index to 'zero' or 'median'.

    Columns matching neither pattern default to 'median', which is the safe
    choice for an unrecognised continuous measurement.
    """
    plan = {}
    for idx, name in enumerate(feature_names):
        name = str(name)
        if any(p in name for p in ZERO_FILL_PATTERNS):
            plan[idx] = "zero"
        elif any(p in name for p in MEDIAN_FILL_PATTERNS):
            plan[idx] = "median"
        else:
            plan[idx] = "median"
    return plan


class FoldImputer:
    """Impute using training-fold statistics only.

    Deliberately minimal and explicit rather than an sklearn SimpleImputer, so
    that the zero-vs-median distinction the manuscript describes is visible in
    one place and testable.
    """

    def __init__(self, plan):
        self.plan = plan
        self.fill_values_ = None

    def fit(self, X_train):
        X_train = np.asarray(X_train, dtype=float)
        fill = np.zeros(X_train.shape[1], dtype=float)
        for idx in range(X_train.shape[1]):
            if self.plan.get(idx, "median") == "zero":
                fill[idx] = 0.0
            else:
                column = X_train[:, idx]
                finite = column[np.isfinite(column)]
                # A column entirely missing in this training fold has no median
                # to borrow; 0 is the only defensible constant, and it is
                # recorded rather than silently substituted.
                fill[idx] = float(np.median(finite)) if finite.size else 0.0
        self.fill_values_ = fill
        return self

    def transform(self, X):
        if self.fill_values_ is None:
            raise RuntimeError("FoldImputer.transform called before fit")
        X = np.array(X, dtype=float, copy=True)
        missing = ~np.isfinite(X)
        if missing.any():
            cols = np.where(missing)[1]
            X[missing] = self.fill_values_[cols]
        return X

    def fit_transform(self, X_train):
        return self.fit(X_train).transform(X_train)


# ---------------------------------------------------------------------------
# Cohort cache
# ---------------------------------------------------------------------------
# Computing the James score components is the slowest part of preprocessing by
# a wide margin: get_baseline_creatinine, get_discharge_creatinine and
# get_albuminuria_status each run per patient via DataFrame.apply, and each call
# filters the full 710,000-row lab table. That is ~1.5 minutes for 4,694
# patients -- and it was being repeated for every experiment, seven times per
# run, for an identical result.
#
# The scored cohort depends only on the input files and the scoring logic, never
# on which model or feature set is being fitted, so it is cached.
#
# The key includes a hash of james_score_helpers.py itself, so editing the
# scoring logic invalidates the cache automatically rather than relying on
# anyone remembering to bump a version number.

CACHE_SUBDIR = ".cache"


def _cache_dir():
    path = os.path.join(config.PROJECT_ROOT, CACHE_SUBDIR)
    os.makedirs(path, exist_ok=True)
    return path


def cohort_cache_key():
    """Fingerprint of every input the scored cohort depends on."""
    import hashlib

    parts = []
    for path in [config.get_features_path(), *config.get_labs_paths()]:
        if path and os.path.exists(path):
            stat = os.stat(path)
            parts.append(f"{os.path.basename(path)}:{stat.st_size}:{int(stat.st_mtime)}")
        else:
            parts.append(f"{path}:missing")

    helper = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "james_score_helpers.py")
    if os.path.exists(helper):
        with open(helper, "rb") as f:
            parts.append("logic:" + hashlib.sha256(f.read()).hexdigest()[:16])

    parts.append(f"upcr:{getattr(config, 'ALBUMINURIA_INCLUDE_UPCR', False)}")
    parts.append("schema:2")          # bump if the cached frame's shape changes

    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:24]


def clear_cohort_cache():
    """Delete every cached cohort. Returns the number of files removed."""
    import glob as _glob

    removed = 0
    for path in _glob.glob(os.path.join(_cache_dir(), "cohort_*.pkl")):
        os.remove(path)
        removed += 1
    print(f"[Cache] Removed {removed} cached cohort file(s) from {_cache_dir()}")
    return removed
def build_scored_cohort(cohort_log=None, verbose=True, use_cache=True):
    """Load the cohort, filter it, and compute the James score components.

    The expensive, experiment-INDEPENDENT half of preprocessing, split out so
    it can be cached. Returns (features_df, CohortLog).
    """
    import pickle

    log = cohort_log or CohortLog()
    key = cohort_cache_key()
    cache_path = os.path.join(_cache_dir(), f"cohort_{key}.pkl")

    if use_cache and os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            payload = pickle.load(f)
        log.steps = payload["log_steps"]
        if verbose:
            print(f"[Cache] HIT cohort_{key}.pkl "
                  f"({os.path.getsize(cache_path) / 1e6:.1f} MB) — skipping the "
                  f"James score recomputation")
            for step in log.steps:
                print(f"[Cohort] {step['description']} -> "
                      f"{step['n_remaining']} patients remaining")
        return payload["features_df"], log

    if verbose and use_cache:
        print("[Cache] MISS — computing the scored cohort "
              "(the James score components take a minute or two)")

    features_df = pd.read_csv(config.get_features_path())
    log.record("Loaded features.csv", features_df)

    features_df["admit_date"] = pd.to_datetime(features_df["admit_date"])
    features_df["discharge_date"] = pd.to_datetime(features_df["discharge_date"])
    features_df["death_date"] = pd.to_datetime(features_df["death_date"])

    features_df = features_df.drop(
        features_df[features_df["death_date"] <= features_df["discharge_date"]].index
    )
    log.record("Dropped patients who died before discharge", features_df)

    # Derived time features. Computed unconditionally: they are cheap, and
    # making them conditional on the feature set would make the cached cohort
    # experiment-specific for no benefit. Feature sets that do not list them
    # simply ignore them.
    if True:
        features_df["length_of_stay"] = (
            features_df["discharge_date"] - features_df["admit_date"]
        ).dt.days
        for stage_num in (1, 2, 3):
            col = f"stage{stage_num}_date"
            features_df[f"admit_to_stage{stage_num}_or_discharge"] = features_df.apply(
                lambda x, c=col: (pd.to_datetime(x[c]) - x["admit_date"]).days
                if pd.notna(x[c]) else (x["discharge_date"] - x["admit_date"]).days,
                axis=1,
            )

    # Mortality-derived columns.  Retained on features_df for the competing-risk
    # analysis; excluded from the model inputs below.
    features_df["days_to_death"] = (
        features_df["death_date"] - features_df["discharge_date"]
    ).dt.days
    features_df["death_within_1yr"] = (
        (features_df["days_to_death"] <= 365) & features_df["death_date"].notna()
    ).fillna(False)
    features_df["ckd_or_death"] = (
        features_df["ckd_stage45"].astype(bool) | features_df["death_within_1yr"]
    )

    # ------------------------------------------------------------------
    # Labs and James score components
    # ------------------------------------------------------------------
    in_hosp_labs_path, pre_hosp_labs_path = config.get_labs_paths()
    all_labs_df = pd.read_csv(in_hosp_labs_path)
    if pre_hosp_labs_path is not None and os.path.exists(pre_hosp_labs_path):
        all_labs_df = pd.concat(
            [all_labs_df, pd.read_csv(pre_hosp_labs_path)], ignore_index=True
        )

    for col in ("test_date", "AdmitDt", "DischDt"):
        all_labs_df[col] = pd.to_datetime(all_labs_df[col], errors="coerce")
    if verbose:
        print(f"[Data] Loaded {len(all_labs_df):,} lab rows")

    features_df["sex_points"] = features_df.sex.map({1: 3, 0: 0})
    features_df["age_admit_points"] = features_df.age_admit.map(age_mapping)
    features_df["highest_stage_points"] = features_df.highest_stage.map(stage_mapping)

    features_df["baseline_creatinine_raw"] = features_df.apply(
        lambda row: get_baseline_creatinine(
            row["patient_id"], all_labs_df, row["admit_date"], row["discharge_date"]),
        axis=1)
    features_df["baseline_creatinine_points"] = \
        features_df.baseline_creatinine_raw.map(baseline_creatinine_mapping)

    features_df["discharge_creatinine_raw"] = features_df.apply(
        lambda row: get_discharge_creatinine(
            row["patient_id"], all_labs_df, row["admit_date"], row["discharge_date"]),
        axis=1)
    features_df["discharge_creatinine_points"] = \
        features_df.discharge_creatinine_raw.map(discharge_creatinine_mapping)

    include_upcr = getattr(config, "ALBUMINURIA_INCLUDE_UPCR", False)
    features_df["albuminuria_status_raw"] = features_df.apply(
        lambda row: get_albuminuria_status(
            row["patient_id"], all_labs_df, row["admit_date"], row["discharge_date"],
            include_upcr=include_upcr)[0],
        axis=1)

    if include_upcr:
        # Recompute without the fallback so the impact is a number in the log
        # rather than something to be taken on trust.
        published = features_df.apply(
            lambda row: get_albuminuria_status(
                row["patient_id"], all_labs_df, row["admit_date"],
                row["discharge_date"], include_upcr=False)[0],
            axis=1)
        changed = published != features_df["albuminuria_status_raw"]
        print(f"[Albuminuria] SENSITIVITY ARM: uPCR fallback enabled.")
        print(f"[Albuminuria] band changed for {int(changed.sum()):,} of "
              f"{len(features_df):,} patients "
              f"({changed.mean() * 100:.2f}%)")
        if changed.any():
            moves = (pd.DataFrame({"from": published[changed],
                                   "to": features_df.loc[changed,
                                                         "albuminuria_status_raw"]})
                     .groupby(["from", "to"]).size().sort_values(ascending=False))
            for (src_band, dst_band), n in moves.items():
                print(f"    {src_band:>11s} -> {dst_band:<8s} {n:>6,}")
        report_unknown_pcr_units()
    features_df["albuminuria_status_points"] = \
        features_df.albuminuria_status_raw.map(albuminuria_status_mapping)

    if verbose:
        print("[Data] Computed James score components")
        report_unknown_units()

    features_df = features_df.dropna(subset=["baseline_creatinine_raw"])
    log.record("Dropped missing baseline creatinine", features_df)
    features_df = features_df.dropna(subset=["discharge_creatinine_raw"])
    log.record("Dropped missing discharge creatinine", features_df)

    point_cols = ["sex_points", "age_admit_points", "highest_stage_points",
                  "baseline_creatinine_points", "discharge_creatinine_points",
                  "albuminuria_status_points"]
    features_df["james_score"] = features_df[point_cols].sum(axis=1, skipna=False)

    # A NaN here means a component failed to map.  The original code let those
    # patients through with a NaN score; dropping them explicitly is what the
    # cohort diagram's "no valid creatinine measurement" step describes.
    n_bad = int(features_df["james_score"].isna().sum())
    if n_bad:
        print(f"[Data] WARNING: {n_bad} patient(s) had an unmappable James score "
              f"component and are being dropped. Component NaN counts:")
        for col in point_cols:
            n_col = int(features_df[col].isna().sum())
            if n_col:
                print(f"    {col}: {n_col}")
        features_df = features_df.dropna(subset=["james_score"])
        log.record("Dropped unmappable James score component", features_df)

    # Backwards-compatible alias for any downstream code still using the old name
    features_df["alberta_score"] = features_df["james_score"]
    if verbose:
        print("[Data] Computed James score")

    # Row-wise deterministic fills.  These use no cross-patient statistics, so
    # they are safe to apply before the CV split -- unlike the median fills,
    # which are now handled per fold by FoldImputer.
    features_df = _apply_rowwise_stage_fills(features_df)

    if use_cache:
        with open(cache_path, "wb") as f:
            pickle.dump({"features_df": features_df, "log_steps": log.steps},
                        f, protocol=4)
        if verbose:
            print(f"[Cache] Wrote cohort_{key}.pkl "
                  f"({os.path.getsize(cache_path) / 1e6:.1f} MB); the remaining "
                  f"experiments in this run reuse it")

    return features_df, log




# ---------------------------------------------------------------------------
# Main preprocessing entry point
# ---------------------------------------------------------------------------

def preprocess_data(exp_config, cohort_log=None, verbose=True):
    """Preprocess data for a single experiment.

    Returns a dict with:
        features        (n_patients, n_features) float array, NaN PRESERVED
        labels          (n_patients,) int array
        feature_names   array of column names
        patient_ids     array of patient identifiers, aligned to rows
        features_df     the full DataFrame (population tables, subgroups, UMAP)
        imputation_plan {column_index: 'zero'|'median'} for FoldImputer
        cohort_log      CohortLog instance
        missingness     per-column missing fraction (reported, not acted on)
    """
    feature_set = exp_config.feature_set
    target = exp_config.target
    features_df, log = build_scored_cohort(
        cohort_log, verbose=verbose,
        use_cache=getattr(config, "USE_COHORT_CACHE", True))


    # ------------------------------------------------------------------
    # Feature selection for this experiment
    # ------------------------------------------------------------------
    features_df, feature_columns = _select_feature_columns(features_df, feature_set)

    labels = (features_df["ckd_or_death"] if target == "ckdordeath"
              else features_df["ckd_stage45"]).astype(int).values

    selected = features_df[feature_columns]
    non_numeric = [c for c in feature_columns
                   if not pd.api.types.is_numeric_dtype(selected[c])
                   and not pd.api.types.is_bool_dtype(selected[c])]
    if non_numeric:
        raise TypeError(
            f"{len(non_numeric)} feature column(s) are not numeric and were not "
            f"encoded: {non_numeric[:10]}"
            f"{' ...' if len(non_numeric) > 10 else ''}\n"
            f"Add them to the encoder in _one_hot_encode(), or exclude them via "
            f"EXCLUDED_PATTERNS."
        )
    features_used = selected.astype(float).values
    feature_names = np.array(feature_columns)
    patient_ids = features_df["patient_id"].values

    missingness = features_df[feature_columns].isnull().mean()
    if verbose:
        print(f"[Data] Missingness: median {missingness.median():.4f}, "
              f"mean {missingness.mean():.4f}, "
              f"{int((missingness > 0.5).sum())} column(s) >50% missing")
        print(f"[Data] Final matrix: {features_used.shape[0]} patients "
              f"x {features_used.shape[1]} features "
              f"| {int(labels.sum())} events ({labels.mean() * 100:.1f}%)")

    return {
        "features": features_used,
        "labels": labels,
        "feature_names": feature_names,
        "patient_ids": patient_ids,
        "features_df": features_df,
        "imputation_plan": build_imputation_plan(feature_names),
        "cohort_log": log,
        "missingness": missingness,
    }


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _apply_rowwise_stage_fills(features_df):
    """Fill stage flags and stage-specific creatinine from the same patient's data.

    Every rule here reads only other columns of the same row, so no information
    crosses between patients and nothing leaks across a CV split.
    """
    for stage in ("stage1", "stage2", "stage3"):
        if stage in features_df.columns:
            features_df[stage] = features_df[stage].fillna(0)

    if {"stage1_creatinine", "discharge_creatinine_raw"} <= set(features_df.columns):
        mask = features_df["stage1"] == 0
        features_df.loc[mask, "stage1_creatinine"] = (
            features_df.loc[mask, "discharge_creatinine_raw"] * 88.4)
    if {"stage2_creatinine", "stage1_creatinine"} <= set(features_df.columns):
        mask = features_df["stage2"] == 0
        features_df.loc[mask, "stage2_creatinine"] = features_df.loc[mask, "stage1_creatinine"]
    if {"stage3_creatinine", "stage2_creatinine"} <= set(features_df.columns):
        mask = features_df["stage3"] == 0
        features_df.loc[mask, "stage3_creatinine"] = features_df.loc[mask, "stage2_creatinine"]

    return features_df


EXCLUDED_PATTERNS = (
    "patient_id", "_raw", "_date", "after",
    "ckd_or_death", "ckd_stage45", "_points",
    "alberta_score", "james_score",
    "days_to_death", "death_within_1yr",
)

# Already 0/1; one-hot encoding them would create a redundant column pair.
BINARY_COLUMNS = {"sex"}


def _select_feature_columns(features_df, feature_set):
    """Choose model input columns and apply encoding.

    Encoding is applied identically regardless of model type. In the original
    code only the transformer path one-hot encoded categoricals, so XGBoost and
    the transformer were compared on different input matrices and a performance
    difference could not be attributed to architecture.
    """
    features_df = features_df.drop(
        columns=["index_vars:admission_services"], errors="ignore")

    if feature_set == "james_score":
        return features_df, ["james_score"]

    if feature_set == "james_raw":
        columns = ["sex", "age_admit", "highest_stage",
                   "baseline_creatinine_raw", "discharge_creatinine_raw",
                   "albuminuria_status_raw"]
        return _one_hot_encode(features_df, columns)

    if feature_set in ("expanded", "egfr"):
        candidates = [c for c in features_df.columns
                      if not any(p in c for p in EXCLUDED_PATTERNS)]
        return _one_hot_encode(features_df, candidates)

    raise ValueError(f"Unknown feature_set: {feature_set!r}")


def _one_hot_encode(features_df, feature_columns):
    """One-hot encode categorical columns in the given list.

    A column is categorical if it is object/bool dtype, or numeric with fewer
    than 10 distinct values (and is not a known binary indicator).
    """
    categorical = []
    for col in feature_columns:
        if col not in features_df.columns or col in BINARY_COLUMNS:
            continue
        series = features_df[col]
        # dtype-agnostic: pandas 3 gives string columns a StringDtype rather
        # than object, so an `== "object"` test silently misses them and the
        # column reaches .astype(float) as a string.
        is_numeric = (pd.api.types.is_numeric_dtype(series)
                      and not pd.api.types.is_bool_dtype(series))
        if not is_numeric or series.nunique(dropna=True) < 10:
            categorical.append(col)

    if not categorical:
        return features_df, list(feature_columns)

    before = set(features_df.columns)
    features_df = pd.get_dummies(features_df, columns=categorical, dummy_na=False)
    created = [c for c in features_df.columns if c not in before]

    kept = [c for c in feature_columns if c not in categorical]
    return features_df, kept + created


# ---------------------------------------------------------------------------
# Backwards-compatible aliases for the pre-rename feature-set names
# ---------------------------------------------------------------------------

FEATURE_SET_ALIASES = {
    "alberta_score": "james_score",
    "alberta_raw": "james_raw",
}


def canonical_feature_set(name):
    """Translate a legacy feature-set name to its current one."""
    return FEATURE_SET_ALIASES.get(name, name)
