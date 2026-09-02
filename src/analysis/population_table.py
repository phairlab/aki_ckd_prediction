"""
Population description, missing-data reporting and sample-size adequacy.

Covers three things the editor did not raise but reviewers reliably do, and
which TRIPOD+AI requires:

* Table 3, regenerated from the run rather than transcribed, overall and split
  by outcome. Also emits the tested-vs-untested split used by the ascertainment
  analysis, so both tables come from the same code.

* A missing-data table (TRIPOD+AI item 11). The submitted manuscript computed
  median missingness and printed it to the console; nothing reached the paper.
  With 478 candidate variables drawn from routine care, per-variable
  missingness is material to whether the expanded feature set could have worked
  at all.

* Sample-size adequacy. 286 events against 100 selected features is ~2.9 events
  per variable, far below the classic 10 EPV heuristic and below what Riley et
  al. (Stat Med 2019) require. Reporting it pre-empts the question and,
  usefully for this manuscript, supports its own argument: an expanded feature
  set that the sample cannot support is a reason to expect no gain.
"""

from __future__ import annotations

import os
import json

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Table 3
# ---------------------------------------------------------------------------

TABLE3_CONTINUOUS = [
    ("age_admit", "Age (years)"),
    ("total_los", "Length of stay (days)"),
    ("james_score", "James score"),
    ("baseline_creatinine_raw", "Baseline creatinine (mg/dL)"),
    ("discharge_creatinine_raw", "Discharge creatinine (mg/dL)"),
]

TABLE3_BINARY = [
    ("sex", "Sex (female)"),
    ("index_vars:icu", "Received ICU care"),
    ("index_vars:sepsis", "Sepsis"),
    ("index_vars:cardiac_surgery", "Cardiac surgery"),
    ("index_vars:mechanical_ventilation", "Mechanical ventilation"),
    ("pre-index_vars:hypertension", "Hypertension"),
    ("pre-index_vars:diabetes", "Diabetes"),
    ("pre-index_vars:chf", "Heart failure"),
    ("pre-index_vars:pvd", "Peripheral vascular disease"),
    ("pre-index_vars:cancer", "Cancer"),
    ("death_within_1yr", "Died within one year"),
]

TABLE3_CATEGORICAL = [
    ("highest_stage", "Highest AKI stage"),
    ("albuminuria_status_raw", "Albuminuria status"),
]


def build_population_table(features_df, group_col="ckd_stage45",
                           group_labels=("Did not progress", "Progressed"),
                           output_dir=None, filename="table3_population.csv"):
    """Descriptive table overall and stratified by `group_col`.

    Continuous variables as mean (SD), categorical as n (%), percentages within
    column -- matching the submitted Table 3 so the two can be compared.
    """
    df = features_df.copy()
    if group_col not in df.columns:
        raise KeyError(f"{group_col!r} not in features_df")

    grouping = df[group_col].astype(bool)
    groups = [("All", df),
              (group_labels[0], df[~grouping]),
              (group_labels[1], df[grouping])]

    rows = [{"Variable": "n", **{name: f"{len(sub)}" for name, sub in groups}}]

    for col, label in TABLE3_CONTINUOUS:
        if col not in df.columns:
            continue
        row = {"Variable": label}
        for name, sub in groups:
            values = pd.to_numeric(sub[col], errors="coerce").dropna()
            row[name] = (f"{values.mean():.2f} ({values.std(ddof=1):.2f})"
                         if len(values) else "—")
        rows.append(row)

    for col, label in TABLE3_BINARY:
        if col not in df.columns:
            continue
        row = {"Variable": label}
        for name, sub in groups:
            values = pd.to_numeric(sub[col], errors="coerce")
            n_pos = int(np.nansum(values.to_numpy() > 0))
            denom = int(values.notna().sum())
            row[name] = (f"{n_pos} ({n_pos / denom * 100:.1f}%)" if denom else "—")
        rows.append(row)

    for col, label in TABLE3_CATEGORICAL:
        if col not in df.columns:
            continue
        rows.append({"Variable": label, **{name: "" for name, _ in groups}})
        for level in sorted(df[col].dropna().unique(), key=str):
            row = {"Variable": f"    {level}"}
            for name, sub in groups:
                n_level = int((sub[col] == level).sum())
                row[name] = f"{n_level} ({n_level / len(sub) * 100:.1f}%)" if len(sub) else "—"
            rows.append(row)

    table = pd.DataFrame(rows)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, filename)
        table.to_csv(path, index=False)
        print(f"[Population] Table 3 -> {path}")

    return table


# ---------------------------------------------------------------------------
# Missing data (TRIPOD+AI item 11)
# ---------------------------------------------------------------------------

def missingness_report(features_df, feature_columns, output_dir=None,
                       top_n=40):
    """Per-variable missingness, with the distribution summarised.

    Also reports missingness stratified by outcome: a variable missing
    differentially by outcome is informative missingness, which a model can
    exploit in a way that will not transfer to deployment.
    """
    present = [c for c in feature_columns if c in features_df.columns]
    sub = features_df[present]

    overall = sub.isna().mean()
    by_outcome = None
    if "ckd_stage45" in features_df.columns:
        mask = features_df["ckd_stage45"].astype(bool)
        by_outcome = pd.DataFrame({
            "missing_progressed": sub[mask].isna().mean(),
            "missing_non_progressed": sub[~mask].isna().mean(),
        })
        by_outcome["difference"] = (by_outcome["missing_progressed"]
                                    - by_outcome["missing_non_progressed"])

    table = pd.DataFrame({"variable": overall.index, "missing_fraction": overall.values})
    if by_outcome is not None:
        table = table.merge(by_outcome.reset_index().rename(columns={"index": "variable"}),
                            on="variable", how="left")
    table = table.sort_values("missing_fraction", ascending=False).reset_index(drop=True)

    summary = {
        "n_variables": int(len(present)),
        "median_missing": float(overall.median()),
        "mean_missing": float(overall.mean()),
        "n_complete": int((overall == 0).sum()),
        "n_over_10pct": int((overall > 0.10).sum()),
        "n_over_50pct": int((overall > 0.50).sum()),
        "n_over_90pct": int((overall > 0.90).sum()),
    }
    if by_outcome is not None:
        differential = by_outcome["difference"].abs().sort_values(ascending=False)
        summary["n_differential_over_10pct"] = int((differential > 0.10).sum())
        summary["most_differential"] = [
            {"variable": v, "difference": float(by_outcome.loc[v, "difference"])}
            for v in differential.head(10).index
        ]

    print(f"\n[Missingness] {summary['n_variables']} candidate variables")
    print(f"  median missing        : {summary['median_missing'] * 100:.1f}%")
    print(f"  complete (0% missing) : {summary['n_complete']}")
    print(f"  >10% missing          : {summary['n_over_10pct']}")
    print(f"  >50% missing          : {summary['n_over_50pct']}")
    print(f"  >90% missing          : {summary['n_over_90pct']}")
    if by_outcome is not None:
        print(f"  differentially missing by outcome (>10pp): "
              f"{summary.get('n_differential_over_10pct', 0)}")

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        table.head(top_n).to_csv(
            os.path.join(output_dir, "missingness_top.csv"), index=False)
        table.to_csv(os.path.join(output_dir, "missingness_all.csv"), index=False)
        with open(os.path.join(output_dir, "missingness_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[Missingness] -> {output_dir}")

    return table, summary


# ---------------------------------------------------------------------------
# Sample size adequacy
# ---------------------------------------------------------------------------

def riley_minimum_sample_size(n_predictors, prevalence, expected_r2_cs=None,
                              target_shrinkage=0.9, max_mape=0.05):
    """Minimum sample size for a binary prediction model (Riley et al. 2019).

    Three criteria; the requirement is the largest.
      1. shrinkage of no more than 10% of the model's predictor effects;
      2. small optimism in the apparent Nagelkerke R-squared (<= 0.05);
      3. precise estimation of the overall risk (MAPE <= 0.05).

    `expected_r2_cs` is the anticipated Cox-Snell R-squared. Absent a value, a
    conservative default takes 15% of the maximum attainable Cox-Snell
    R-squared for the given prevalence, which is the usual pragmatic choice
    when no prior model is available.
    """
    p = float(prevalence)
    P = int(n_predictors)

    # Degenerate prevalence has no defined requirement; say so rather than
    # returning a NaN that propagates into round() downstream.
    if not (0 < p < 1) or P < 1:
        return {
            "n_predictors": P, "prevalence": p,
            "max_r2_cs": np.nan, "assumed_r2_cs": np.nan,
            "n_criterion1_shrinkage": np.nan,
            "n_criterion2_optimism": np.nan,
            "n_criterion3_risk_precision": np.nan,
            "n_required": np.nan, "events_required": np.nan,
            "epv_required": np.nan,
            "note": ("prevalence must be strictly between 0 and 1 and there must "
                     "be at least one predictor"),
        }

    # Maximum attainable Cox-Snell R-squared at this prevalence
    ln_lnull = p * np.log(p) + (1 - p) * np.log(1 - p)
    max_r2_cs = 1 - np.exp(2 * ln_lnull)

    r2_cs = float(expected_r2_cs) if expected_r2_cs is not None else 0.15 * max_r2_cs

    # Criterion 1: shrinkage
    n1 = P / ((target_shrinkage - 1) * np.log(1 - r2_cs / target_shrinkage))

    # Criterion 2: small optimism in R-squared
    shrinkage_2 = r2_cs / (r2_cs + 0.05 * max_r2_cs)
    n2 = P / ((shrinkage_2 - 1) * np.log(1 - r2_cs / shrinkage_2))

    # Criterion 3: precise risk estimate
    n3 = (1.96 / max_mape) ** 2 * p * (1 - p)

    required = float(max(n1, n2, n3))
    return {
        "n_predictors": P,
        "prevalence": p,
        "max_r2_cs": float(max_r2_cs),
        "assumed_r2_cs": float(r2_cs),
        "n_criterion1_shrinkage": float(n1),
        "n_criterion2_optimism": float(n2),
        "n_criterion3_risk_precision": float(n3),
        "n_required": required,
        "events_required": float(required * p),
        "epv_required": float(required * p / P) if P else np.nan,
    }


def sample_size_report(n_patients, n_events, feature_counts, output_dir=None):
    """Events per variable and the Riley requirement, per feature-set size."""
    prevalence = n_events / n_patients
    rows = []
    for n_features in feature_counts:
        riley = riley_minimum_sample_size(n_features, prevalence)
        required = riley["n_required"]
        finite = bool(np.isfinite(required)) and required > 0
        rows.append({
            "n_candidate_predictors": n_features,
            "n_patients_available": n_patients,
            "n_events_available": n_events,
            "events_per_variable": (n_events / n_features) if n_features else np.nan,
            "riley_n_required": (round(required) if finite else np.nan),
            "riley_events_required": (round(riley["events_required"]) if finite else np.nan),
            "ratio_available_to_required": (n_patients / required) if finite else np.nan,
            "adequate": (bool(n_patients >= required) if finite else False),
        })

    table = pd.DataFrame(rows)

    print(f"\n[SampleSize] n = {n_patients:,}, events = {n_events:,} "
          f"({prevalence * 100:.1f}%)")
    if not (0 < prevalence < 1):
        print("  Prevalence is degenerate (no events, or all events); the Riley "
              "requirement is undefined.")
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            table.to_csv(os.path.join(output_dir, "sample_size_adequacy.csv"), index=False)
        return table

    print(f"  {'predictors':>10s} {'EPV':>7s} {'Riley n':>10s} "
          f"{'available/required':>20s}  adequate")
    for _, r in table.iterrows():
        riley_n = ("{:>10d}".format(int(r["riley_n_required"]))
                   if np.isfinite(r["riley_n_required"]) else "{:>10s}".format("—"))
        ratio = ("{:>20.2f}".format(r["ratio_available_to_required"])
                 if np.isfinite(r["ratio_available_to_required"]) else "{:>20s}".format("—"))
        print(f"  {int(r['n_candidate_predictors']):>10d} "
              f"{r['events_per_variable']:>7.2f} {riley_n} {ratio}  "
              f"{'yes' if r['adequate'] else 'NO'}")

    if not table["adequate"].all():
        print("\n  At least one feature-set size exceeds what this sample supports.")
        print("  Worth stating plainly: it strengthens rather than weakens the")
        print("  manuscript's argument, since a feature set the data cannot support")
        print("  is a principled reason to expect no gain from expanding it.")

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "sample_size_adequacy.csv")
        table.to_csv(path, index=False)
        print(f"[SampleSize] -> {path}")

    return table


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_population_analyses(data, output_dir, feature_counts=(1, 6, 11, 100, 478)):
    """Table 3, missingness and sample-size adequacy in one call."""
    print(f"\n{'=' * 70}\nPOPULATION, MISSINGNESS AND SAMPLE SIZE\n{'=' * 70}")
    features_df = data["features_df"]

    table3 = build_population_table(features_df, output_dir=output_dir)
    missing_table, missing_summary = missingness_report(
        features_df, list(data["feature_names"]), output_dir=output_dir)
    size_table = sample_size_report(
        n_patients=int(len(features_df)),
        n_events=int(np.asarray(data["labels"]).sum()),
        feature_counts=feature_counts,
        output_dir=output_dir)

    return {"table3": table3, "missingness": missing_table,
            "missingness_summary": missing_summary, "sample_size": size_table}
