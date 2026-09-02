"""
Outcome ascertainment and loss to follow-up (editor's point 3).

The problem
-----------
Patients with no qualifying outpatient eGFR during follow-up were classified as
non-progressors. The editor notes this is the one issue the manuscript does not
address anywhere -- Limitation 2 covers death only, and TRIPOD+AI item 8a points
readers at that paragraph for an implication it does not contain.

What is required
----------------
1. How many of the 4,687 had NO outpatient eGFR in the follow-up window.
2. How the untested compare with the tested on the James components and on the
   Table 3 characteristics.
3. A discussion of the direction of the resulting misclassification.

On (3): unascertained progressors counted as negatives inflate the apparent
specificity and depress PPV and the apparent event rate. A patient who was
never tested but flagged high-risk is scored a false positive; if they had in
fact progressed, they were a true positive. So the reported PPV is a LOWER
bound on the true PPV under the assumption that untested patients progress at
least as often as tested ones -- and untested patients are plausibly a
lower-risk group, since being tested is itself a marker of clinical concern.
`ascertainment_bounds()` computes the envelope both ways rather than asserting
a direction.

Data requirement
----------------
Needs a post-discharge lab extract. The nine files the ETL reads are all index
or pre-index. Run `python src/probe_server_data.py --server` first: it reports
whether any extract on disk carries eGFR in the 30-365 day window. If none
does, every function here returns a skip record naming exactly what to request,
and nothing silently guesses.
"""

from __future__ import annotations

import os
import sys
import json

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.analysis.predictions import threshold_metrics


FOLLOWUP_WINDOW_DAYS = (30, 365)


# ---------------------------------------------------------------------------
# Identify who was tested
# ---------------------------------------------------------------------------

def identify_tested_patients(features_df, followup_labs_path,
                             window_days=FOLLOWUP_WINDOW_DAYS,
                             egfr_entities=("egfr", "creatinine")):
    """Flag each patient as tested or untested during the follow-up window.

    Returns (DataFrame[patient_id, n_followup_tests, first_test_day, was_tested],
             status dict). When no follow-up extract is configured the frame is
    None and the status explains what is missing.
    """
    if not followup_labs_path or not os.path.exists(str(followup_labs_path)):
        message = (
            "No post-discharge laboratory extract is configured "
            "(config.FOLLOWUP_LABS_PATH is unset or missing). Editor point 3 "
            "cannot be answered without one.\n"
            "  1. Run `python src/probe_server_data.py --server` — section 2 "
            "reports whether any extract already on disk carries eGFR 30-365 "
            "days post-discharge.\n"
            "  2. If none does, request an AHS extract of outpatient "
            "eGFR/creatinine for these patients from discharge to +365 days, "
            "with test_date, result, unit and patient id.\n"
            "  3. Set config.FOLLOWUP_LABS_PATH to that file and re-run."
        )
        print(f"\n[Ascertainment] SKIPPED.\n{message}")
        return None, {"status": "skipped", "reason": message}

    from lab_normalization import add_canonical_name

    labs = pd.read_csv(followup_labs_path)
    labs = add_canonical_name(labs)

    id_col = "id" if "id" in labs.columns else "patient_id"
    labs[id_col] = labs[id_col].astype(str)
    labs["test_date"] = pd.to_datetime(labs["test_date"], errors="coerce")

    discharge = (features_df[["patient_id", "discharge_date"]]
                 .assign(patient_id=lambda d: d["patient_id"].astype(str)))
    discharge["discharge_date"] = pd.to_datetime(discharge["discharge_date"],
                                                 errors="coerce")

    merged = labs.merge(discharge, left_on=id_col, right_on="patient_id", how="inner")
    merged["days_post"] = (merged["test_date"] - merged["discharge_date"]).dt.days

    lo, hi = window_days
    in_window = merged[
        merged["canonical_test"].isin(egfr_entities)
        & merged["days_post"].between(lo, hi)
    ]

    per_patient = (in_window.groupby("patient_id")
                   .agg(n_followup_tests=("days_post", "size"),
                        first_test_day=("days_post", "min"),
                        last_test_day=("days_post", "max"))
                   .reset_index())

    out = (features_df[["patient_id"]]
           .assign(patient_id=lambda d: d["patient_id"].astype(str))
           .merge(per_patient, on="patient_id", how="left"))
    out["n_followup_tests"] = out["n_followup_tests"].fillna(0).astype(int)
    out["was_tested"] = out["n_followup_tests"] > 0

    # Two qualifying values three months apart are needed to MEET the outcome;
    # a patient with only one test could not have met it however sick they were.
    out["could_meet_outcome"] = out["n_followup_tests"] >= 2

    status = {
        "status": "ok",
        "source": str(followup_labs_path),
        "window_days": list(window_days),
        "n_total": int(len(out)),
        "n_tested": int(out["was_tested"].sum()),
        "n_untested": int((~out["was_tested"]).sum()),
        "pct_untested": float((~out["was_tested"]).mean() * 100),
        "n_could_meet_outcome": int(out["could_meet_outcome"].sum()),
        "n_single_test_only": int(((out["n_followup_tests"] == 1)).sum()),
    }

    print(f"\n[Ascertainment] Follow-up testing, days {lo}-{hi} post-discharge")
    print(f"  cohort                       : {status['n_total']:,}")
    print(f"  with >=1 eGFR/creatinine     : {status['n_tested']:,} "
          f"({100 - status['pct_untested']:.1f}%)")
    print(f"  with NO follow-up test       : {status['n_untested']:,} "
          f"({status['pct_untested']:.1f}%)   <-- the unascertained group")
    print(f"  with only ONE test           : {status['n_single_test_only']:,} "
          f"(cannot meet a two-value outcome definition)")
    print(f"  able to meet the outcome     : {status['n_could_meet_outcome']:,}")

    return out, status


# ---------------------------------------------------------------------------
# Compare tested vs untested
# ---------------------------------------------------------------------------

# Table 3 characteristics plus the James components, which is what the editor
# asked to see the comparison on.
COMPARISON_CONTINUOUS = [
    "age_admit", "total_los", "james_score",
    "baseline_creatinine_raw", "discharge_creatinine_raw",
]
COMPARISON_BINARY = [
    "sex", "ckd_stage45",
    "index_vars:icu", "index_vars:icu_inhosp", "index_vars:dialysis",
    "index_vars:sepsis", "index_vars:cardiac_surgery",
    "pre-index_vars:hypertension", "pre-index_vars:diabetes",
    "pre-index_vars:chf", "pre-index_vars:cancer",
]
COMPARISON_CATEGORICAL = ["highest_stage", "albuminuria_status_raw"]


def compare_tested_untested(features_df, tested_flags, output_dir=None):
    """Table 3-style comparison of the tested and untested groups.

    Continuous variables: mean (SD) with a two-sample t test.
    Binary and categorical: n (%) with a chi-squared test.

    Standardised mean differences are reported alongside p values because at
    this sample size a trivial difference can be significant, and it is the
    magnitude that determines how much the misclassification matters.
    """
    df = features_df.copy()
    df["patient_id"] = df["patient_id"].astype(str)
    df = df.merge(tested_flags[["patient_id", "was_tested"]],
                  on="patient_id", how="left")
    df["was_tested"] = df["was_tested"].fillna(False)

    tested = df[df["was_tested"]]
    untested = df[~df["was_tested"]]
    rows = []

    for col in COMPARISON_CONTINUOUS:
        if col not in df.columns:
            continue
        a = pd.to_numeric(tested[col], errors="coerce").dropna()
        b = pd.to_numeric(untested[col], errors="coerce").dropna()
        if len(a) < 2 or len(b) < 2:
            continue
        t_stat, p = stats.ttest_ind(a, b, equal_var=False)
        pooled_sd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
        rows.append({
            "variable": col, "type": "continuous",
            "tested": f"{a.mean():.2f} ({a.std(ddof=1):.2f})",
            "untested": f"{b.mean():.2f} ({b.std(ddof=1):.2f})",
            "smd": float((a.mean() - b.mean()) / pooled_sd) if pooled_sd > 0 else np.nan,
            "p_value": float(p),
        })

    for col in COMPARISON_BINARY:
        if col not in df.columns:
            continue
        a = pd.to_numeric(tested[col], errors="coerce").dropna().astype(bool)
        b = pd.to_numeric(untested[col], errors="coerce").dropna().astype(bool)
        if len(a) < 2 or len(b) < 2:
            continue
        table = np.array([[a.sum(), len(a) - a.sum()], [b.sum(), len(b) - b.sum()]])
        # chi2_contingency raises when any marginal is zero (a variable that is
        # constant across both groups, which is common among rare comorbidity
        # flags). Report NaN rather than failing the whole table.
        p = np.nan
        if table.sum() and table.sum(axis=0).min() > 0 and table.sum(axis=1).min() > 0:
            try:
                p = stats.chi2_contingency(table)[1]
            except ValueError:
                p = np.nan
        rows.append({
            "variable": col, "type": "binary",
            "tested": f"{int(a.sum())} ({a.mean() * 100:.1f}%)",
            "untested": f"{int(b.sum())} ({b.mean() * 100:.1f}%)",
            "smd": float(_smd_binary(a.mean(), b.mean())),
            "p_value": float(p),
        })

    for col in COMPARISON_CATEGORICAL:
        if col not in df.columns:
            continue
        for level in sorted(df[col].dropna().unique()):
            a = (tested[col] == level)
            b = (untested[col] == level)
            rows.append({
                "variable": f"{col} = {level}", "type": "categorical",
                "tested": f"{int(a.sum())} ({a.mean() * 100:.1f}%)",
                "untested": f"{int(b.sum())} ({b.mean() * 100:.1f}%)",
                "smd": float(_smd_binary(a.mean(), b.mean())),
                "p_value": np.nan,
            })

    table = pd.DataFrame(rows)
    table.insert(1, "n_tested", len(tested))
    table.insert(2, "n_untested", len(untested))

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "ascertainment_tested_vs_untested.csv")
        table.to_csv(path, index=False)
        print(f"[Ascertainment] Comparison table -> {path}")

    notable = table[table["smd"].abs() > 0.1].sort_values(
        "smd", key=lambda s: s.abs(), ascending=False)
    if len(notable):
        print("[Ascertainment] Variables differing by |SMD| > 0.1 "
              "(the conventional imbalance threshold):")
        for _, r in notable.head(12).iterrows():
            print(f"    {r['variable']:<36s} tested {r['tested']:>16s}  "
                  f"untested {r['untested']:>16s}  SMD {r['smd']:+.3f}")
    else:
        print("[Ascertainment] No variable differs by |SMD| > 0.1 — the tested and "
              "untested groups are well balanced, which bounds this bias tightly.")

    return table


def _smd_binary(p1, p2):
    """Standardised mean difference for two proportions."""
    denom = np.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / 2)
    return (p1 - p2) / denom if denom > 0 else np.nan


# ---------------------------------------------------------------------------
# Bounds under unascertained outcomes
# ---------------------------------------------------------------------------

def ascertainment_bounds(predictions, tested_flags, threshold,
                         assumed_rates=(0.0, "observed", 1.0)):
    """Metric envelope over assumptions about the untested patients' outcomes.

    Three scenarios, all reported rather than one asserted:
      0.0        every untested patient was truly a non-progressor -- this is
                 what the manuscript assumed;
      "observed" untested patients progressed at the same rate as the tested,
                 which is the missing-at-random assumption;
      1.0        every untested patient who was flagged high risk had in fact
                 progressed -- the extreme in the opposite direction.

    Only untested patients currently labelled non-progressors can change label;
    a tested progressor is ascertained and stays fixed.
    """
    merged = predictions.merge(
        tested_flags[["patient_id", "was_tested"]], on="patient_id", how="left")
    merged["was_tested"] = merged["was_tested"].fillna(False)

    ambiguous = (~merged["was_tested"]) & (merged["y_true"] == 0)
    n_ambiguous = int(ambiguous.sum())

    observed_rate = float(
        merged.loc[merged["was_tested"], "y_true"].mean()) if merged["was_tested"].any() else 0.0

    rng = np.random.default_rng(1202)
    scenarios = {}
    for rate in assumed_rates:
        y = merged["y_true"].to_numpy().copy()
        label = f"untested_rate_{rate}"
        if rate == "observed":
            label = f"untested_rate_observed_{observed_rate:.3f}"
            flip = rng.random(n_ambiguous) < observed_rate
        elif rate == 0.0:
            label = "untested_all_non_progressors_AS_REPORTED"
            flip = np.zeros(n_ambiguous, dtype=bool)
        else:
            label = "untested_all_progressors"
            flip = np.ones(n_ambiguous, dtype=bool)

        idx = np.where(ambiguous.to_numpy())[0]
        y[idx[flip]] = 1
        scenarios[label] = threshold_metrics(y, merged["y_proba"], threshold)

    print(f"\n[Ascertainment] Metric bounds at threshold {threshold:.2f} "
          f"({n_ambiguous} untested non-progressors)")
    for metric in ("prevalence", "sensitivity", "ppv", "alert_rate"):
        values = [s[metric] for s in scenarios.values() if np.isfinite(s[metric])]
        if values:
            print(f"  {metric:<12s} {min(values):.3f} to {max(values):.3f}")

    return {"analysis": "ascertainment_bounds", "threshold": threshold,
            "n_ambiguous": n_ambiguous, "observed_rate_in_tested": observed_rate,
            "scenarios": scenarios}


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_ascertainment_analysis(experiment_predictions, features_df,
                               followup_labs_path, output_dir, threshold=0.20):
    """Run the full ascertainment analysis and save results."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n{'=' * 70}\nOUTCOME ASCERTAINMENT & LOSS TO FOLLOW-UP  (editor point 3)"
          f"\n{'=' * 70}")

    tested_flags, status = identify_tested_patients(features_df, followup_labs_path)

    results = {"testing_status": status}

    if tested_flags is None:
        with open(os.path.join(output_dir, "ascertainment.json"), "w") as f:
            json.dump(results, f, indent=2, default=str)
        return results

    tested_flags.to_csv(
        os.path.join(output_dir, "ascertainment_per_patient.csv"), index=False)

    comparison = compare_tested_untested(features_df, tested_flags, output_dir)
    results["comparison_table"] = comparison.to_dict(orient="records")

    results["by_experiment"] = {}
    for label, preds in experiment_predictions.items():
        print(f"\n--- {label} ---")
        results["by_experiment"][label] = ascertainment_bounds(
            preds, tested_flags, threshold)

    with open(os.path.join(output_dir, "ascertainment.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[Ascertainment] Written to {output_dir}")
    return results
