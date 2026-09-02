"""
ETL pipeline: raw Hing CSVs -> features.csv + cohort.csv

Reads 9 raw CSV files from the Hing data directory, processes each
data source (cohort, in-hospital variables, flowsheets, consultations,
labs, pre-hospital variables/labs/BMI/medication), then merges everything
into a single features.csv.

This only runs on the secure server (--etl flag). With nonsense data
the features.csv is already pre-built.
"""

import os
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from lab_normalization import (add_canonical_name, add_unit_aware_entity,
                               write_lab_normalization_report)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def categorize_goals_of_care(goal_text):
    if goal_text is None or goal_text == "?":
        return np.nan
    goal_mapping = {
        "Perioperative": "perioperative", "Acute": "acute",
        "Community": "community", "Transition": "transition",
        "ACL": "alc", "Intensive": "intensive",
        "Mobility": "mobility", "Assessment": "assessment",
        "Waiting": "waiting",
    }
    for key, value in goal_mapping.items():
        if key in goal_text:
            return value
    return "others"


SERVICE_LIST = [
    "General Internal Medicine", "Cardiac Surgery", "Emergency Medicine",
    "Nephrology", "Neurology", "Transplant", "Cardiology", "Gastroenterology",
    "Neurosurgery", "Surgery", "Family Medicine", "Critical Care", "Trauma",
    "Urology", "General Surgery", "Orthopedics", "Specialized Geriatrics",
    "Otolaryngology", "Stroke", "Ear Nose and Throat", "ICU", "Transplant",
    "Plastic Surgery", "Hematology",
]


def record_name(n):
    if "ARTERIAL" in n: return "arterial_line_bp"
    if "PRESSURE" in n: return "bp"
    if "ECG" in n: return "heart_rate_ecg"
    if "OXYGEN" in n: return "oxygen_therapy"
    if "BAR BMI" in n: return "bar_bmi"
    if "BMI" in n: return "bmi"
    if "MAC HEART RATE" in n: return "max_heart_rate"
    if "AORTIC" in n: return "aortic_heart_rate"
    if "AWARE" in n: return "aware"
    if n == "?": return np.nan
    return "others"


def record_value(v):
    if v == "Supplemental oxygen" or "AWARE" in str(v) or "Aware" in str(v) or "Yes" in str(v):
        return "1"
    return str(v)


def test_name(n):
    """SUPERSEDED by lab_normalization.normalize_test_name.

    Kept for reference only. It handled three special cases and passed every
    other name through unchanged, which is how random glucose reached the
    feature set under six in-hospital and ten pre-index name strings (editor
    point 5). Note also the typo: names containing "GFR" were mapped to
    "eGRF", so the eGFR features in features.csv were spelled "eGRF" while
    Appendix Table A3.2 reports them as "eGFR".

    Set config.NORMALIZE_LAB_NAMES = False to fall back to this behaviour and
    reproduce the originally submitted feature set.
    """
    if "CRP" in n: return "C-Reactive Protein"
    if n == "Glucose (mmol/L)": return "Glucose"
    if "GFR" in n or "Glomerular" in n: return "eGRF"
    return n


def bmi_name(n):
    return "bar_bmi" if "BAR" in n else "bmi"


def description(d):
    start = d.lower().find("consult to ") + 11
    return d.lower()[start:]


def trim(crosstab, ratio):
    threshold = crosstab.shape[0] * ratio
    to_drop = [c for c in crosstab.columns if crosstab[c].isna().sum() > threshold]
    crosstab.drop(to_drop, inplace=True, axis=1)


def _build_crosstab(df, id_col, prefix, name_col, value_col):
    """Build count/mean/min/max crosstab for a data source."""
    results = []
    for agg, func in [("count", "count"), ("mean", "mean"), ("min", "min"), ("max", "max")]:
        df[f"__{agg}"] = f"{prefix}_{agg}:" + df[name_col]
        ct = pd.crosstab(
            index=df[id_col], columns=df[f"__{agg}"],
            values=df[value_col], aggfunc=func,
        )
        results.append(ct)
    merged = results[0]
    for ct in results[1:]:
        merged = merged.merge(ct, on=id_col, how="left")
    return merged


def _prepare_labs(labs_df, top_n, source_label):
    """Attach the canonical entity name and keep the most frequent entities.

    Entity assignment happens BEFORE the top-N cut. Cutting on raw names first
    (as the original did) meant a single analyte spread thinly across ten name
    strings could be dropped entirely while a redundant duplicate of a common
    one was retained.
    """
    if config.NORMALIZE_LAB_NAMES:
        labs_df = add_canonical_name(labs_df)
        # Split any entity whose spellings disagree on units. Merging umol/L
        # with mmol/L, or a 24-hour excretion with a spot concentration, would
        # average incompatible scales into one crosstab cell.
        labs_df, mixed = add_unit_aware_entity(labs_df)
        name_col = "canonical_test_unit"
        n_raw = labs_df["TEST_NM"].nunique()
        n_entities = labs_df[name_col].nunique()
        print(f"[ETL] {source_label}: {n_raw} raw test names -> "
              f"{n_entities} clinical entities"
              + (f" ({len(mixed)} split by unit)" if mixed else ""))
    else:
        labs_df = labs_df.copy()
        labs_df["canonical_test"] = labs_df["TEST_NM"].apply(test_name)
        name_col = "canonical_test"
        print(f"[ETL] {source_label}: lab normalization DISABLED "
              f"(config.NORMALIZE_LAB_NAMES=False) — reproducing the submitted "
              f"feature set")

    top = list(labs_df[name_col].value_counts().index)[:top_n]
    labs_df = labs_df[labs_df[name_col].isin(top)]

    numeric = labs_df["TEST_RSLT"].astype(str).apply(
        lambda x: x.replace(".", "", 1).replace("-", "", 1).isnumeric())
    labs_df = labs_df[numeric]

    labs_df["patient_id"] = labs_df["id"]
    labs_df["test_date"] = pd.to_datetime(labs_df["test_date"], errors="coerce")
    labs_df["name"] = labs_df[name_col]
    labs_df["result"] = labs_df["TEST_RSLT"].astype(float)
    return labs_df


# ---------------------------------------------------------------------------
# Main ETL function
# ---------------------------------------------------------------------------

# Every file run_etl() writes. Kept as one list so the backup cannot drift out
# of step with the writes -- an earlier version protected only two of the ten.
ETL_OUTPUT_FILES = (
    "features.csv",
    "cohort.csv",
    "index_vars.csv",
    "index_records_crosstab.csv",
    "index_consultations_crosstab.csv",
    "index_labs_crosstab.csv",
    "pre_vars.csv",
    "pre_labs_crosstab.csv",
    "pre_bmi_crosstab.csv",
    "pre_medication.csv",
)


def preflight_report(raw_dir, out_dir):
    """Print exactly what will be read and what will be overwritten.

    Printed before anything is touched. The ETL reads one directory and writes
    another, and on this server the two have historically been confused -- the
    raw directory contains a full set of ETL outputs from some earlier run --
    so it is worth showing the paths and what is already sitting in them.
    """
    print(f"\n  READ FROM (never modified) : {raw_dir}")
    print(f"  WRITE TO                   : {out_dir}")

    existing = []
    for name in ETL_OUTPUT_FILES:
        path = os.path.join(out_dir, name)
        if os.path.exists(path):
            stat = os.stat(path)
            existing.append((name, stat.st_size / 1e6,
                             pd.Timestamp(stat.st_mtime, unit="s")))

    if not existing:
        print(f"  Nothing to overwrite: none of the {len(ETL_OUTPUT_FILES)} output "
              f"files exist there yet.")
        return

    print(f"\n  {len(existing)} existing file(s) WILL BE OVERWRITTEN "
          f"(all are backed up first):")
    for name, size_mb, mtime in existing:
        print(f"    {name:<34s} {size_mb:>8,.1f} MB   modified {mtime:%Y-%m-%d %H:%M}")


def _backup_existing(out_dir, filenames=ETL_OUTPUT_FILES):
    """Copy any existing ETL outputs aside before overwriting them.

    The ETL rewrites features.csv in place. On the server that file is the one
    the SUBMITTED results were computed from, and the resubmission needs it:
    several points in the response letter rest on a before/after comparison
    (untuned vs tuned, raw lab names vs normalized entities), which is not
    possible once the original is gone.

    Backups are timestamped and never overwritten, so re-running the ETL
    repeatedly cannot eat the original.
    """
    import shutil
    from datetime import datetime

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = os.path.join(out_dir, "pre_reanalysis_backup")
    saved = []

    for name in filenames:
        source = os.path.join(out_dir, name)
        if not os.path.exists(source):
            continue
        os.makedirs(backup_dir, exist_ok=True)
        stem, ext = os.path.splitext(name)
        target = os.path.join(backup_dir, f"{stem}_{stamp}{ext}")
        shutil.copy2(source, target)
        saved.append(target)

    if saved:
        print(f"[ETL] Backed up {len(saved)} existing file(s) before overwriting:")
        for path in saved:
            size = os.path.getsize(path) / 1e6
            print(f"    {path}  ({size:,.1f} MB)")
    return saved


def run_etl():
    """Run the full ETL pipeline: raw CSVs -> features.csv + cohort.csv."""
    raw_dir = config.get_raw_data_dir()
    out_dir = config.get_etl_output_dir()
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'-' * 70}\n[ETL] Pre-flight\n{'-' * 70}")
    preflight_report(raw_dir, out_dir)
    _backup_existing(out_dir)
    print(f"{'-' * 70}\n")

    # ---- 1. Cohort ----
    cohort = pd.read_csv(os.path.join(raw_dir, "cohort and outcome.csv"))
    print(f"[ETL] Loaded cohort -> {len(cohort)} patients")

    cohort.rename(columns={
        "id": "patient_id", "AGE_ADMIT": "age_admit",
        "stage1_dt": "stage1_date", "stage1_value": "stage1_creatinine",
        "stage2_dt": "stage2_date", "stage2_value": "stage2_creatinine",
        "stage3_dt": "stage3_date", "stage3_value": "stage3_creatinine",
        "higheststage": "highest_stage", "CKD_stage45": "ckd_stage45",
        "stroke_af": "stroke_after", "chf_af": "chf_after", "mi_af": "mi_after",
    }, inplace=True)
    cohort["admit_date"] = pd.to_datetime(cohort["AdmitDt"])
    cohort["discharge_date"] = pd.to_datetime(cohort["DischDt"])
    cohort["sex"] = (cohort["SEX"] == "F").astype(int)
    cohort["death_date"] = pd.to_datetime(cohort["death_date"])
    cohort = cohort[[
        "patient_id", "admit_date", "discharge_date", "sex", "age_admit",
        "total_los", "stage1", "stage1_date", "stage1_creatinine",
        "stage2", "stage2_date", "stage2_creatinine",
        "stage3", "stage3_date", "stage3_creatinine",
        "highest_stage", "death_date", "ckd_stage45",
        "stroke_after", "chf_after", "mi_after",
    ]]
    cohort.to_csv(os.path.join(out_dir, "cohort.csv"), index=False)

    # ---- 2. In-hospital variables ----
    index_vars = pd.read_csv(os.path.join(raw_dir, "in-hosp vars.csv"))
    index_vars.rename(columns={
        "id": "patient_id",
        "Cardiac_catheterization": "cardiac_catheterization",
        "Mechanical_ventilation": "mechanical_ventilation",
        "OT_assessment": "ot_assessment",
        "PT_assessment": "pt_assessment",
    }, inplace=True)

    index_vars["goals_of_care"] = index_vars["goals_of_care"].fillna("?").apply(categorize_goals_of_care)
    for cat in ["acute", "community", "transition", "others", "intensive",
                "mobility", "assessment", "perioperative"]:
        index_vars[f"goal_{cat}"] = (index_vars["goals_of_care"] == cat).astype(int)

    index_vars["covid_test_result"] = (index_vars["covid_test_result"] == "Positive").astype(int)

    for service in SERVICE_LIST:
        col = service.lower().replace(" ", "_")
        index_vars[col] = index_vars["admission_services"].str.lower().str.contains(
            service.lower(), na=False
        ).astype(int)

    cols_keep = (
        ["patient_id", "icu_inhosp", "icu"]
        + [f"goal_{c}" for c in ["acute", "community", "transition", "others",
                                  "intensive", "mobility", "assessment", "perioperative"]]
        + ["cardiac_surgery", "insulin", "beta_blocker", "covid_test_result",
           "smoke", "cardiac_catheterization", "ami", "chf", "dialysis",
           "mechanical_ventilation", "renal_ultralsound", "angiogram",
           "foley_catheter", "ot_assessment", "pt_assessment",
           "obstructive_uropathy", "sepsis", "admission_services"]
        + [s.lower().replace(" ", "_") for s in SERVICE_LIST]
    )
    index_vars = index_vars[cols_keep]
    index_vars = index_vars.add_prefix("index_vars:")
    index_vars.rename(columns={"index_vars:patient_id": "patient_id"}, inplace=True)
    index_vars.to_csv(os.path.join(out_dir, "index_vars.csv"), index=False)
    print(f"[ETL] Processed in-hospital variables -> {index_vars.shape}")

    # ---- 3. In-hospital flowsheet records ----
    records_path = os.path.join(raw_dir, "in-hosp flowsheet records.csv")
    if os.path.exists(records_path):
        index_records = pd.read_csv(records_path)
        index_records["patient_id"] = index_records["id"]
        index_records["record_date"] = pd.to_datetime(index_records["record_date"])
        index_records["name"] = index_records["FLO_MEAS_NAME"].apply(record_name)
        index_records["value"] = index_records["MEAS_VALUE"].apply(record_value)

        # Split BP into systolic/diastolic
        bp_mask = index_records["name"].isin(["bp", "arterial_line_bp"])
        index_bp = index_records.loc[bp_mask].copy()
        index_bp["systolic_bp"] = index_bp["value"].str.split("/").str[0].astype(float)
        index_bp["diastolic_bp"] = index_bp["value"].str.split("/").str[1].astype(float)

        bp_s = index_bp[["patient_id", "record_date", "name", "systolic_bp"]].rename(
            columns={"systolic_bp": "value"})
        bp_d = index_bp[["patient_id", "record_date", "name", "diastolic_bp"]].rename(
            columns={"diastolic_bp": "value"})
        non_bp = index_records.loc[~bp_mask, ["patient_id", "record_date", "name", "value"]]
        index_records = pd.concat([non_bp, bp_s, bp_d]).sort_values("patient_id")
        index_records["value"] = index_records["value"].astype(float)

        index_records_crosstab = _build_crosstab(
            index_records, "patient_id", "records", "name", "value"
        )
        index_records_crosstab.to_csv(os.path.join(out_dir, "index_records_crosstab.csv"))
        index_records_crosstab = pd.read_csv(os.path.join(out_dir, "index_records_crosstab.csv"))
        print(f"[ETL] Processed flowsheet records -> {index_records_crosstab.shape}")
    else:
        index_records_crosstab = pd.DataFrame(columns=["patient_id"])

    # ---- 4. In-hospital consultations ----
    consults_path = os.path.join(raw_dir, "in-hosp consultations.csv")
    if os.path.exists(consults_path):
        consults = pd.read_csv(consults_path)
        consults["patient_id"] = consults["id"]
        consults["consultation"] = consults["DESCRIPTION"].apply(description)
        consults["consult_count"] = "consult_count:" + consults["consultation"]
        consults_ct = pd.crosstab(
            index=consults["patient_id"], columns=consults["consult_count"],
            values=consults["consultation"], aggfunc="count",
        )
        trim(consults_ct, 199 / 200)
        consults_ct.to_csv(os.path.join(out_dir, "index_consultations_crosstab.csv"))
        consults_ct = pd.read_csv(os.path.join(out_dir, "index_consultations_crosstab.csv"))
        print(f"[ETL] Processed consultations -> {consults_ct.shape}")
    else:
        consults_ct = pd.DataFrame(columns=["patient_id"])

    # ---- 5. In-hospital labs ----
    index_labs_all = pd.read_csv(os.path.join(raw_dir, "in-hosp labs.csv"))
    index_labs = _prepare_labs(index_labs_all, config.TOP_LABS_IN_HOSPITAL, "in-hosp labs")

    index_labs_crosstab = _build_crosstab(
        index_labs, "patient_id", "labs", "name", "result"
    )
    index_labs_crosstab.to_csv(os.path.join(out_dir, "index_labs_crosstab.csv"))
    index_labs_crosstab = pd.read_csv(os.path.join(out_dir, "index_labs_crosstab.csv"))
    print(f"[ETL] Processed in-hospital labs -> {index_labs_crosstab.shape}")

    # ---- 6. Pre-hospital variables ----
    pre_vars_path = os.path.join(raw_dir, "pre-hosp vars.csv")
    if os.path.exists(pre_vars_path):
        pre_vars = pd.read_csv(pre_vars_path)
        pre_vars["covid_test_result"] = pre_vars["covid_test_result"].replace("Positive", 1)
        pre_vars.rename(columns={
            "id": "patient_id", "chf_pre": "chf", "pvd_pre": "pvd", "pud_pre": "pud",
            "mild_liver_disease_pre": "mild_liver_disease", "cancer_pre": "cancer",
            "mild_severe_liver_disease_pre": "mild_severe_liver_disease",
            "htn_pre": "hypertension", "diab_pre": "diabetes", "gout_pre": "gout",
        }, inplace=True)
        pre_vars = pre_vars[["patient_id", "chf", "pvd", "pud", "mild_liver_disease",
                              "cancer", "mild_severe_liver_disease", "hypertension",
                              "diabetes", "gout", "covid_test_result"]]
        pre_vars = pre_vars.add_prefix("pre-index_vars:")
        pre_vars.rename(columns={"pre-index_vars:patient_id": "patient_id"}, inplace=True)
        pre_vars.to_csv(os.path.join(out_dir, "pre_vars.csv"), index=False)
        print(f"[ETL] Processed pre-hospital variables -> {pre_vars.shape}")
    else:
        pre_vars = pd.DataFrame(columns=["patient_id"])

    # ---- 7. Pre-hospital labs ----
    pre_labs_path = os.path.join(raw_dir, "pre-hosp labs.csv")
    pre_labs_all = None
    if os.path.exists(pre_labs_path):
        pre_labs_all = pd.read_csv(pre_labs_path)
        pre_labs = _prepare_labs(pre_labs_all, config.TOP_LABS_PRE_INDEX, "pre-hosp labs")

        pre_labs_crosstab = _build_crosstab(
            pre_labs, "patient_id", "pre-index_labs", "name", "result"
        )
        pre_labs_crosstab.to_csv(os.path.join(out_dir, "pre_labs_crosstab.csv"))
        pre_labs_crosstab = pd.read_csv(os.path.join(out_dir, "pre_labs_crosstab.csv"))
        print(f"[ETL] Processed pre-hospital labs -> {pre_labs_crosstab.shape}")
    else:
        pre_labs_crosstab = pd.DataFrame(columns=["patient_id"])

    # ---- 8. Pre-hospital BMI ----
    bmi_path = os.path.join(raw_dir, "pre-hosp bmi.csv")
    if os.path.exists(bmi_path):
        pre_bmi = pd.read_csv(bmi_path)
        pre_bmi["patient_id"] = pre_bmi["id"]
        pre_bmi["record_date"] = pd.to_datetime(pre_bmi["record_date"])
        pre_bmi["name"] = pre_bmi["FLO_MEAS_NAME"].apply(bmi_name)
        pre_bmi["value"] = pre_bmi["MEAS_VALUE"]

        pre_bmi_crosstab = _build_crosstab(
            pre_bmi, "patient_id", "pre-index", "name", "value"
        )
        pre_bmi_crosstab.to_csv(os.path.join(out_dir, "pre_bmi_crosstab.csv"))
        pre_bmi_crosstab = pd.read_csv(os.path.join(out_dir, "pre_bmi_crosstab.csv"))
        print(f"[ETL] Processed pre-hospital BMI -> {pre_bmi_crosstab.shape}")
    else:
        pre_bmi_crosstab = pd.DataFrame(columns=["patient_id"])

    # ---- 9. Pre-hospital medication ----
    med_path = os.path.join(raw_dir, "pre-hosp medication.csv")
    if os.path.exists(med_path):
        pre_med = pd.read_csv(med_path)
        pre_med.rename(columns={
            "id": "patient_id", "AMINOGLYCOSIDE": "aminoglycoside",
            "AMPHOTERICIN_B": "amphotericin_b", "DIURETIC_K": "diuretic_k",
            "DIURETIC_NONK": "diuretic_non_k", "NSAIDS": "nsaids",
            "PPI": "ppi", "SGLT2": "sglt2", "VANCOMYCIN": "vancomycin",
        }, inplace=True)
        pre_med = pre_med[["patient_id", "aminoglycoside", "amphotericin_b",
                            "diuretic_k", "diuretic_non_k", "nsaids",
                            "ppi", "sglt2", "vancomycin"]]
        pre_med = pre_med.add_prefix("pre-index_medication:")
        pre_med.rename(columns={"pre-index_medication:patient_id": "patient_id"}, inplace=True)
        pre_med.to_csv(os.path.join(out_dir, "pre_medication.csv"), index=False)
        print(f"[ETL] Processed pre-hospital medication -> {pre_med.shape}")
    else:
        pre_med = pd.DataFrame(columns=["patient_id"])

    # ---- Final merge ----
    features = cohort
    for df in [index_vars, index_records_crosstab, consults_ct,
               index_labs_crosstab, pre_vars, pre_labs_crosstab,
               pre_bmi_crosstab, pre_med]:
        features = features.merge(df, on="patient_id", how="outer")

    features.to_csv(os.path.join(out_dir, "features.csv"), index=False)
    print(f"[ETL] Final merged features -> {features.shape[0]} patients x {features.shape[1]} columns")
    print(f"[ETL] Saved to {os.path.join(out_dir, 'features.csv')}")

    # Audit trail for editor point 5: which raw name strings collapsed into
    # which clinical entity, and how many redundant columns that removed.
    if config.NORMALIZE_LAB_NAMES:
        frames = {"in-hosp": index_labs_all}
        if pre_labs_all is not None:
            frames["pre-index"] = pre_labs_all
        try:
            write_lab_normalization_report(frames, config.get_reports_dir())
        except Exception as exc:                                   # noqa: BLE001
            print(f"[ETL] Could not write the lab normalization audit: {exc}")

    # Column inventory, so Multimedia Appendix 3 regenerates from the run
    inventory = pd.DataFrame({"index": range(features.shape[1]),
                              "column": features.columns})
    os.makedirs(config.get_reports_dir(), exist_ok=True)
    inventory.to_csv(os.path.join(config.get_reports_dir(),
                                  "feature_inventory.csv"), index=False)
    print(f"[ETL] Feature inventory -> "
          f"{os.path.join(config.get_reports_dir(), 'feature_inventory.csv')}")

    return features


if __name__ == "__main__":
    run_etl()
