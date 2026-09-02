"""
Data preprocessing for the AKI-CKD prediction pipeline.

Loads features.csv and lab CSVs, computes Alberta score components,
applies cohort filters, selects features based on the experiment config,
and returns everything the CV engine needs.
"""

import numpy as np
import pandas as pd

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from alberta_score_helpers import (
    age_mapping, stage_mapping,
    baseline_creatinine_mapping, discharge_creatinine_mapping,
    albuminuria_status_mapping,
    get_baseline_creatinine, get_discharge_creatinine, get_albuminuria_status,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


# ---------------------------------------------------------------------------
# Cohort logging
# ---------------------------------------------------------------------------

def log_cohort_change(description, df):
    """Print cohort size after a filtering step."""
    print(f"[Cohort] {description} -> {len(df)} patients remaining")


# ---------------------------------------------------------------------------
# Main preprocessing entry point
# ---------------------------------------------------------------------------

def preprocess_data(exp_config):
    """Preprocess data for a single experiment.

    Parameters
    ----------
    exp_config : config.ExperimentConfig

    Returns
    -------
    dict with keys:
        features    — numpy array (n_patients, n_features)
        labels      — numpy array (n_patients,)
        feature_names — numpy array of column names
        features_df — the full DataFrame (for UMAP, population tables, etc.)
        eGFR_features — numpy array or empty list
    """
    feature_set = exp_config.feature_set
    target = exp_config.target
    model_type = exp_config.model_type

    # ------------------------------------------------------------------
    # Load features.csv
    # ------------------------------------------------------------------
    features_df = pd.read_csv(config.get_features_path())
    log_cohort_change("Loaded features.csv", features_df)

    features_df["admit_date"] = pd.to_datetime(features_df["admit_date"])
    features_df["discharge_date"] = pd.to_datetime(features_df["discharge_date"])

    # Drop patients who died before discharge
    features_df.drop(
        features_df[features_df["death_date"] <= features_df["discharge_date"]].index,
        inplace=True,
    )
    log_cohort_change("Dropped patients who died before discharge", features_df)

    # Compute derived time features (needed for expanded feature set)
    if feature_set == "expanded":
        features_df["length_of_stay"] = (
            features_df["discharge_date"] - features_df["admit_date"]
        ).dt.days

        for stage_num in [1, 2, 3]:
            col = f"stage{stage_num}_date"
            features_df[f"admit_to_stage{stage_num}_or_discharge"] = features_df.apply(
                lambda x, c=col: (pd.to_datetime(x[c]) - x["admit_date"]).days
                if pd.notna(x[c])
                else (x["discharge_date"] - x["admit_date"]).days,
                axis=1,
            )

    # Death-within-1yr and composite outcome
    features_df["death_date"] = pd.to_datetime(features_df["death_date"])
    features_df["days_to_death"] = (
        features_df["death_date"] - features_df["discharge_date"]
    ).dt.days
    features_df["death_within_1yr"] = (
        (features_df["days_to_death"] <= 365) & (~pd.isna(features_df["death_date"]))
    ).fillna(False)
    features_df["ckd_or_death"] = (
        features_df["ckd_stage45"] | features_df["death_within_1yr"]
    )

    # Drop columns we don't want leaking into features
    features_df = features_df.drop(
        ["death_date", "days_to_death", "death_within_1yr",
         "index_vars:admission_services"],
        axis=1,
        errors="ignore",
    )

    # ------------------------------------------------------------------
    # Load lab CSVs (for Alberta score computation)
    # ------------------------------------------------------------------
    in_hosp_labs_path, pre_hosp_labs_path = config.get_labs_paths()
    index_labs_df = pd.read_csv(in_hosp_labs_path)

    if pre_hosp_labs_path is not None:
        prehosp_labs_df = pd.read_csv(pre_hosp_labs_path)
        all_labs_df = pd.concat([index_labs_df, prehosp_labs_df], ignore_index=True)
    else:
        all_labs_df = index_labs_df

    all_labs_df["test_date"] = pd.to_datetime(all_labs_df["test_date"])
    all_labs_df["AdmitDt"] = pd.to_datetime(all_labs_df["AdmitDt"])
    all_labs_df["DischDt"] = pd.to_datetime(all_labs_df["DischDt"])
    print("[Data] Loaded lab data")

    # ------------------------------------------------------------------
    # Compute Alberta score components
    # ------------------------------------------------------------------
    features_df["sex_points"] = features_df.sex.map({1: 3, 0: 0})
    features_df["age_admit_points"] = features_df.age_admit.map(age_mapping)
    features_df["highest_stage_points"] = features_df.highest_stage.map(stage_mapping)

    features_df["baseline_creatinine_raw"] = features_df.apply(
        lambda row: get_baseline_creatinine(
            row["patient_id"], all_labs_df, row["admit_date"], row["discharge_date"]
        ),
        axis=1,
    )
    features_df["baseline_creatinine_points"] = features_df.baseline_creatinine_raw.map(
        baseline_creatinine_mapping
    )

    features_df["discharge_creatinine_raw"] = features_df.apply(
        lambda row: get_discharge_creatinine(
            row["patient_id"], all_labs_df, row["admit_date"], row["discharge_date"]
        ),
        axis=1,
    )
    features_df["discharge_creatinine_points"] = features_df.discharge_creatinine_raw.map(
        discharge_creatinine_mapping
    )

    features_df["albuminuria_status_raw"] = features_df.apply(
        lambda row: get_albuminuria_status(
            row["patient_id"], all_labs_df, row["admit_date"], row["discharge_date"]
        )[0],
        axis=1,
    )
    features_df["albuminuria_status_points"] = features_df.albuminuria_status_raw.map(
        albuminuria_status_mapping
    )

    print("[Data] Computed Alberta score components")

    # ------------------------------------------------------------------
    # Drop patients missing baseline or discharge creatinine
    # ------------------------------------------------------------------
    features_df = features_df.dropna(subset=["baseline_creatinine_raw"])
    log_cohort_change("Dropped missing baseline creatinine", features_df)

    features_df = features_df.dropna(subset=["discharge_creatinine_raw"])
    log_cohort_change("Dropped missing discharge creatinine", features_df)

    # Compute total Alberta score
    features_df["alberta_score"] = (
        features_df["sex_points"]
        + features_df["age_admit_points"]
        + features_df["highest_stage_points"]
        + features_df["baseline_creatinine_points"]
        + features_df["discharge_creatinine_points"]
        + features_df["albuminuria_status_points"]
    )
    print("[Data] Computed Alberta score")

    # ------------------------------------------------------------------
    # eGFR temporal features (only for egfr experiment)
    # ------------------------------------------------------------------
    eGFR_feature_set = []
    if feature_set == "egfr":
        eGFR_feature_set = _compute_egfr_features(features_df, all_labs_df)

    # ------------------------------------------------------------------
    # Imputation for transformer models (expanded feature set)
    # ------------------------------------------------------------------
    if model_type == "transformer" and feature_set == "expanded":
        features_df = _impute_for_transformer(features_df)

    # ------------------------------------------------------------------
    # Select feature columns based on experiment config
    # ------------------------------------------------------------------
    if feature_set == "alberta_score":
        feature_columns = ["alberta_score"]

    elif feature_set == "alberta_raw":
        feature_columns = [
            "sex", "age_admit", "highest_stage",
            "baseline_creatinine_raw", "discharge_creatinine_raw",
            "albuminuria_status_raw",
        ]
        # One-hot encode categorical columns
        feature_columns, features_df = _one_hot_encode(features_df, feature_columns)

    elif feature_set in ("expanded", "egfr"):
        # Use all features except excluded patterns
        excluded_patterns = [
            "patient_id", "_raw", "_date", "after",
            "ckd_or_death", "ckd_stage45", "_points", "alberta_score",
        ]

        if model_type == "transformer":
            # One-hot encode categoricals first
            # Binary columns should NOT be one-hot encoded (already 0/1)
            binary_cols = {"sex"}
            cat_features = []
            for col in features_df.columns:
                if col in binary_cols:
                    continue
                if not any(p in col for p in excluded_patterns):
                    if (features_df[col].dtype == "object"
                            or features_df[col].dtype == "bool"
                            or (features_df[col].dtype in ["int64", "float64"]
                                and len(features_df[col].unique()) < 10)):
                        cat_features.append(col)
            if cat_features:
                features_df = pd.get_dummies(features_df, columns=cat_features)

        feature_columns = [
            col for col in features_df.columns
            if not any(p in col for p in excluded_patterns)
        ]

    else:
        raise ValueError(f"Unknown feature_set: {feature_set}")

    # ------------------------------------------------------------------
    # Build output arrays
    # ------------------------------------------------------------------
    if target == "ckdordeath":
        labels = features_df["ckd_or_death"].values
    else:
        labels = features_df["ckd_stage45"].values

    features_used = features_df[feature_columns].values
    feature_names = np.array(feature_columns)

    # Compute the median missingness of the features
    missingness = features_df[feature_columns].isnull().sum() / len(features_df)
    median_missingness = missingness.median()
    print(sorted(missingness.tolist()))
    print(f"Median missingness of features: {median_missingness:.4f}")

    # # Handle any remaining NaN/inf values in features
    # features_used = np.nan_to_num(features_used, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"[Data] Final feature matrix: {features_used.shape[0]} patients x {features_used.shape[1]} features")

    return {
        "features": features_used,
        "labels": labels,
        "feature_names": feature_names,
        "features_df": features_df,
        "eGFR_features": eGFR_feature_set,
    }


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _compute_egfr_features(features_df, all_labs_df, max_weeks=100):
    """Build 100-week eGFR temporal feature matrix."""
    eGFR_labs = all_labs_df[all_labs_df.TEST_NM == "eGFR"].copy()
    eGFR_labs = eGFR_labs.dropna(subset=["TEST_RSLT"])
    eGFR_labs = eGFR_labs[pd.to_numeric(eGFR_labs["TEST_RSLT"], errors="coerce").notna()]
    eGFR_labs["TEST_RSLT"] = pd.to_numeric(eGFR_labs["TEST_RSLT"], errors="coerce")
    eGFR_labs["id"] = eGFR_labs["id"].astype(str)
    eGFR_labs = eGFR_labs[eGFR_labs["test_date"] <= eGFR_labs["DischDt"]]
    eGFR_labs = eGFR_labs.sort_values(by=["id", "test_date"])

    median_val = eGFR_labs.TEST_RSLT.median()
    patient_vectors = {}

    for (patient_id, disch_date), group in eGFR_labs.groupby(["id", "DischDt"]):
        vector = [np.nan] * max_weeks
        for week in range(1, max_weeks + 1):
            start = disch_date - pd.Timedelta(weeks=week)
            end = disch_date - pd.Timedelta(weeks=week - 1)
            weekly = group[(group["test_date"] >= start) & (group["test_date"] < end)]
            if not weekly.empty:
                vector[week - 1] = weekly["TEST_RSLT"].astype(float).mean()
        vector = pd.Series(vector).ffill().bfill().tolist()
        patient_vectors[patient_id] = vector[::-1]

    pvdf = pd.DataFrame(patient_vectors).T
    pvdf.index.name = "patient_id"
    pvdf.reset_index(inplace=True)

    features_df2 = features_df.copy()
    features_df2["patient_id"] = features_df2["patient_id"].astype(str)
    pvdf["patient_id"] = pvdf["patient_id"].astype(str)

    merged = pd.merge(features_df2, pvdf, on="patient_id", how="left")
    for col in range(max_weeks):
        merged[col] = merged[col].fillna(median_val)

    return merged.loc[:, 0 : max_weeks - 1].to_numpy()


def _one_hot_encode(features_df, feature_columns):
    """One-hot encode categorical columns in the feature list.

    Returns (updated_feature_columns, updated_features_df).
    """
    # Binary columns should NOT be one-hot encoded (already 0/1)
    binary_cols = {"sex"}

    cat_features = []
    for col in feature_columns:
        if col not in features_df.columns:
            continue
        if col in binary_cols:
            continue
        if (features_df[col].dtype == "object"
                or features_df[col].dtype == "bool"
                or (features_df[col].dtype in ["int64", "float64"]
                    and len(features_df[col].unique()) < 10)):
            cat_features.append(col)

    if not cat_features:
        return feature_columns, features_df

    old_cols = set(features_df.columns)
    features_df = pd.get_dummies(features_df, columns=cat_features)
    new_cols = [c for c in features_df.columns if c not in old_cols]

    feature_columns = [c for c in feature_columns if c not in cat_features] + new_cols
    return feature_columns, features_df


def _impute_for_transformer(features_df):
    """Fill missing values for transformer training on the expanded feature set."""
    # Stage columns: fill with 0
    for s in ["stage1", "stage2", "stage3"]:
        if s in features_df.columns:
            features_df[s] = features_df[s].fillna(0)

    # Impute stage creatinine values from discharge creatinine
    if "discharge_creatinine_raw" in features_df.columns:
        mask = features_df["stage1"] == 0
        features_df.loc[mask, "stage1_creatinine"] = (
            features_df.loc[mask, "discharge_creatinine_raw"] * 88.4
        )
    if "stage2_creatinine" in features_df.columns:
        mask = features_df["stage2"] == 0
        features_df.loc[mask, "stage2_creatinine"] = features_df.loc[mask, "stage1_creatinine"]
    if "stage3_creatinine" in features_df.columns:
        mask = features_df["stage3"] == 0
        features_df.loc[mask, "stage3_creatinine"] = features_df.loc[mask, "stage2_creatinine"]

    # Counts: fill with 0
    for pattern in ["records_count", "consult_count", "labs_count",
                    "pre-index_labs_count", "pre-index_medication",
                    "pre-index_vars", "pre-index_count"]:
        cols = [c for c in features_df.columns if pattern in c]
        features_df[cols] = features_df[cols].fillna(0)

    # Means/mins/maxes: fill with column median
    for pattern in ["records_mean", "records_min", "records_max",
                    "labs_mean", "labs_min", "labs_max",
                    "pre-index_labs_mean", "pre-index_labs_min", "pre-index_labs_max",
                    "pre-index_mean", "pre-index_min", "pre-index_max"]:
        cols = [c for c in features_df.columns if pattern in c]
        if cols:
            features_df[cols] = features_df[cols].fillna(features_df[cols].median())

    return features_df
