"""
Competing risk of death (editor's point 2).

The problem
-----------
The outcome requires two outpatient eGFR values <30 mL/min/1.73 m-squared
separated by three months. Patients who died within three months of discharge
were excluded. Patients who died LATER in the follow-up window without two
qualifying tests were classified as non-progressors -- 400 of them, 9.1% of the
analytic cohort. An unknown fraction would have met the outcome had they
survived long enough to be measured.

The editor explicitly does not want the study reframed as a time-to-event
analysis, since that would break the like-for-like comparison with the James
score. What is required is the MAGNITUDE of the bias.

Three analyses, in increasing order of what they assume
-------------------------------------------------------
1. `complete_case_analysis` -- restrict to patients who survived the full
   follow-up window and recompute everything. Every death-related
   misclassification is removed by construction. This is the cleanest
   sensitivity analysis and it needs no assumption at all, at the cost of
   changing the estimand to "risk among one-year survivors".

2. `bounds_analysis` -- the deterministic envelope. Reclassify every deceased
   non-progressor first as a non-event (as reported) and then as an event, and
   report both. The truth is inside those bounds whatever the unobserved
   outcomes were. This is the strongest statement available without any
   modelling assumption, and it is the one to put in the response letter.

3. `tipping_point_analysis` -- the fraction of deceased non-progressors that
   would have had to progress before a stated conclusion changes. Converts the
   bias from something acknowledged into something quantified.

`cumulative_incidence` (Aalen-Johansen) is also provided, but it needs a DATE
of CKD progression, which `features.csv` does not currently carry -- only the
binary flag. It is wired up and will run if an outcome-date column is supplied;
otherwise it reports precisely what is missing rather than guessing.
"""

from __future__ import annotations

import os
import json

import numpy as np
import pandas as pd

from src.analysis.predictions import threshold_metrics


# ---------------------------------------------------------------------------
# Cohort mortality description
# ---------------------------------------------------------------------------

def describe_mortality(features_df, horizon_days=365):
    """Counts needed for the response letter's opening sentence on this point."""
    df = features_df.copy()
    df["death_date"] = pd.to_datetime(df["death_date"], errors="coerce")
    df["discharge_date"] = pd.to_datetime(df["discharge_date"], errors="coerce")
    days = (df["death_date"] - df["discharge_date"]).dt.days

    died = df["death_date"].notna() & (days <= horizon_days)
    progressed = df["ckd_stage45"].astype(bool)

    summary = {
        "n_total": int(len(df)),
        "n_progressed": int(progressed.sum()),
        "event_rate": float(progressed.mean()),
        "n_died_within_horizon": int(died.sum()),
        "pct_died_within_horizon": float(died.mean() * 100),
        # The number the editor named: non-progressors who died in follow-up.
        "n_died_not_progressed": int((died & ~progressed).sum()),
        "pct_died_not_progressed": float((died & ~progressed).mean() * 100),
        "n_died_and_progressed": int((died & progressed).sum()),
        "median_days_to_death": (float(days[died].median()) if died.any() else None),
        "horizon_days": horizon_days,
    }

    print(f"\n[CompetingRisk] Cohort n = {summary['n_total']:,}")
    print(f"  progressed to CKD 4-5      : {summary['n_progressed']:,} "
          f"({summary['event_rate'] * 100:.1f}%)")
    print(f"  died within {horizon_days} d          : {summary['n_died_within_horizon']:,} "
          f"({summary['pct_died_within_horizon']:.1f}%)")
    print(f"  died, counted as non-events: {summary['n_died_not_progressed']:,} "
          f"({summary['pct_died_not_progressed']:.1f}%)   <-- the biased group")
    return summary


# ---------------------------------------------------------------------------
# 1. Complete case among survivors
# ---------------------------------------------------------------------------

def complete_case_analysis(predictions, features_df, threshold,
                           horizon_days=365):
    """Recompute threshold metrics after excluding deaths during follow-up.

    Uses the existing out-of-fold predictions rather than refitting: dropping
    rows from the evaluation does not change what the model learned, and
    refitting would confound the sensitivity analysis with a different training
    sample.
    """
    died = _died_within(features_df, horizon_days)
    merged = predictions.merge(died, on="patient_id", how="left")
    merged["died_in_followup"] = merged["died_in_followup"].fillna(False)

    n_dropped = int(merged["died_in_followup"].sum())
    survivors = merged[~merged["died_in_followup"]]

    full = threshold_metrics(merged["y_true"], merged["y_proba"], threshold)
    surv = threshold_metrics(survivors["y_true"], survivors["y_proba"], threshold)

    return {
        "analysis": "complete_case_survivors",
        "threshold": threshold,
        "n_excluded_deaths": n_dropped,
        "as_reported": full,
        "survivors_only": surv,
        "delta": {k: (surv[k] - full[k])
                  for k in ("sensitivity", "specificity", "ppv", "npv",
                            "alert_rate", "prevalence")
                  if np.isfinite(surv.get(k, np.nan)) and np.isfinite(full.get(k, np.nan))},
    }


# ---------------------------------------------------------------------------
# 2. Deterministic bounds
# ---------------------------------------------------------------------------

def bounds_analysis(predictions, features_df, threshold, horizon_days=365):
    """Best- and worst-case bounds on every threshold metric.

    Lower bound: deceased non-progressors are non-events, exactly as reported.
    Upper bound: every one of them would have progressed.

    Whatever the unobserved outcomes were, the truth lies between these two.
    That makes this the strongest claim available about the magnitude of the
    bias without assuming anything about the deceased.
    """
    died = _died_within(features_df, horizon_days)
    merged = predictions.merge(died, on="patient_id", how="left")
    merged["died_in_followup"] = merged["died_in_followup"].fillna(False)

    # Only non-progressors who died are ambiguous; a death after a recorded
    # progression does not change that patient's label.
    ambiguous = merged["died_in_followup"] & (merged["y_true"] == 0)
    n_ambiguous = int(ambiguous.sum())

    as_reported = threshold_metrics(merged["y_true"], merged["y_proba"], threshold)

    worst = merged["y_true"].copy()
    worst[ambiguous] = 1
    all_progressed = threshold_metrics(worst, merged["y_proba"], threshold)

    print(f"\n[CompetingRisk] Bounds at threshold {threshold:.2f} "
          f"({n_ambiguous} ambiguous patients)")
    for metric in ("prevalence", "sensitivity", "ppv", "alert_rate"):
        lo, hi = as_reported[metric], all_progressed[metric]
        print(f"  {metric:<12s} {min(lo, hi):.3f} to {max(lo, hi):.3f}"
              f"   (reported {lo:.3f})")

    return {
        "analysis": "deterministic_bounds",
        "threshold": threshold,
        "n_ambiguous": n_ambiguous,
        "assume_none_progressed": as_reported,      # as reported
        "assume_all_progressed": all_progressed,
        "bounds": {
            m: sorted([as_reported[m], all_progressed[m]])
            for m in ("prevalence", "sensitivity", "specificity", "ppv", "npv",
                      "alert_rate", "net_benefit")
            if np.isfinite(as_reported.get(m, np.nan))
            and np.isfinite(all_progressed.get(m, np.nan))
        },
    }


# ---------------------------------------------------------------------------
# 3. Tipping point
# ---------------------------------------------------------------------------

def tipping_point_analysis(predictions, features_df, threshold,
                           horizon_days=365, targets=None, seed=1202,
                           n_draws=200):
    """What fraction of deceased non-progressors must have progressed to move a metric?

    Sweeps the assumed progression fraction pi from 0 to 1. For each pi the
    ambiguous patients are randomly relabelled as events with probability pi,
    averaged over `n_draws` draws so the answer does not hinge on which
    particular patients were picked.

    `targets` maps a metric name to the value that would change the conclusion,
    e.g. {"ppv": 0.30} for "does PPV stay above 30%".
    """
    targets = targets or {"ppv": 0.30, "sensitivity": 0.50}
    rng = np.random.default_rng(seed)

    died = _died_within(features_df, horizon_days)
    merged = predictions.merge(died, on="patient_id", how="left")
    merged["died_in_followup"] = merged["died_in_followup"].fillna(False)

    ambiguous_idx = np.where(
        (merged["died_in_followup"] & (merged["y_true"] == 0)).to_numpy())[0]
    y_base = merged["y_true"].to_numpy().copy()
    proba = merged["y_proba"].to_numpy()

    rows = []
    for pi in np.linspace(0.0, 1.0, 21):
        draws = []
        for _ in range(1 if pi in (0.0, 1.0) else n_draws):
            y = y_base.copy()
            if len(ambiguous_idx):
                flip = rng.random(len(ambiguous_idx)) < pi
                y[ambiguous_idx[flip]] = 1
            draws.append(threshold_metrics(y, proba, threshold))
        row = {"assumed_progression_fraction": round(float(pi), 3),
               "n_reclassified": int(round(pi * len(ambiguous_idx)))}
        for metric in ("prevalence", "sensitivity", "specificity", "ppv",
                       "npv", "alert_rate"):
            values = [d[metric] for d in draws if np.isfinite(d[metric])]
            # PPV is undefined when a model flags nobody at this threshold.
            # Averaging an empty slice would emit a warning and return NaN;
            # returning NaN explicitly keeps the distinction visible downstream.
            row[metric] = float(np.mean(values)) if values else np.nan
        rows.append(row)

    sweep = pd.DataFrame(rows)

    tipping = {}
    for metric, target in targets.items():
        series = sweep[metric].to_numpy()
        defined = np.isfinite(series)
        if not defined.any():
            tipping[metric] = {
                "target": target, "value_at_pi_0": None, "value_at_pi_1": None,
                "crosses": None, "tipping_fraction": None,
                "status": "undefined across the whole range",
            }
            continue
        start_above = series[0] >= target
        crossed = np.where(np.where(defined,
                                    (series < target) if start_above else (series >= target),
                                    False))[0]
        tipping[metric] = {
            "target": target,
            "value_at_pi_0": float(series[0]) if defined[0] else None,
            "value_at_pi_1": float(series[-1]) if defined[-1] else None,
            "crosses": bool(len(crossed)),
            "tipping_fraction": (float(sweep["assumed_progression_fraction"].iloc[crossed[0]])
                                 if len(crossed) else None),
            "status": "ok",
        }

    print(f"\n[CompetingRisk] Tipping points at threshold {threshold:.2f}")
    for metric, info in tipping.items():
        if info.get("status") != "ok":
            # Undefined is not the same as robust: a model that flags nobody has
            # no PPV to defend, and saying "robust" here would be wrong.
            print(f"  {metric}: undefined across the range "
                  f"(the model flags no patients at this threshold)")
        elif info["crosses"]:
            print(f"  {metric}: crosses {info['target']} once "
                  f"{info['tipping_fraction'] * 100:.0f}% of deceased non-progressors "
                  f"are assumed to have progressed")
        else:
            print(f"  {metric}: never crosses {info['target']} "
                  f"({info['value_at_pi_0']:.3f} -> {info['value_at_pi_1']:.3f} "
                  f"across the full range) — the conclusion is robust")

    return {"analysis": "tipping_point", "threshold": threshold,
            "n_ambiguous": int(len(ambiguous_idx)),
            "sweep": sweep, "tipping": tipping}


# ---------------------------------------------------------------------------
# 4. Aalen-Johansen cumulative incidence (needs outcome dates)
# ---------------------------------------------------------------------------

def cumulative_incidence(features_df, outcome_date_col=None, horizon_days=365):
    """Cumulative incidence of CKD with death as a competing event.

    Requires a DATE of progression. `features.csv` currently carries only the
    binary `ckd_stage45` flag, so this returns a skip record naming exactly what
    is needed rather than silently substituting the naive estimate.
    """
    if outcome_date_col is None or outcome_date_col not in features_df.columns:
        message = (
            "Aalen-Johansen cumulative incidence needs a date of CKD progression. "
            "features.csv carries only the binary ckd_stage45 flag. To run this, "
            "add the date of the FIRST qualifying outpatient eGFR<30 to the cohort "
            "extract and pass its column name as outcome_date_col. The bounds and "
            "tipping-point analyses above do not need it and already quantify the "
            "bias the editor asked about."
        )
        print(f"\n[CompetingRisk] SKIPPED cumulative incidence.\n  {message}")
        return {"analysis": "cumulative_incidence", "status": "skipped",
                "reason": message}

    df = features_df.copy()
    df["discharge_date"] = pd.to_datetime(df["discharge_date"], errors="coerce")
    df["death_date"] = pd.to_datetime(df["death_date"], errors="coerce")
    df[outcome_date_col] = pd.to_datetime(df[outcome_date_col], errors="coerce")

    t_event = (df[outcome_date_col] - df["discharge_date"]).dt.days
    t_death = (df["death_date"] - df["discharge_date"]).dt.days

    # cause 1 = CKD progression, cause 2 = death, 0 = censored at the horizon
    time = np.full(len(df), float(horizon_days))
    cause = np.zeros(len(df), dtype=int)

    has_event = t_event.notna() & (t_event <= horizon_days)
    time[has_event.to_numpy()] = t_event[has_event].to_numpy()
    cause[has_event.to_numpy()] = 1

    has_death = t_death.notna() & (t_death <= horizon_days) & ~has_event
    time[has_death.to_numpy()] = t_death[has_death].to_numpy()
    cause[has_death.to_numpy()] = 2

    result = _aalen_johansen(time, cause, horizon_days)
    naive = float(df["ckd_stage45"].astype(bool).mean())
    result.update({
        "analysis": "cumulative_incidence",
        "status": "ok",
        "naive_proportion": naive,
        "absolute_difference_naive_minus_cif": naive - result["cif_ckd_at_horizon"],
    })

    print(f"\n[CompetingRisk] Cumulative incidence at {horizon_days} d")
    print(f"  CIF, CKD (death as competing) : {result['cif_ckd_at_horizon']:.4f}")
    print(f"  CIF, death                    : {result['cif_death_at_horizon']:.4f}")
    print(f"  naive proportion (as reported): {naive:.4f}")
    print(f"  difference                    : "
          f"{result['absolute_difference_naive_minus_cif']:+.4f}")
    return result


def _aalen_johansen(time, cause, horizon):
    """Non-parametric cumulative incidence for two competing causes.

    Hand-rolled rather than pulled from lifelines so the repository has one
    fewer hard dependency for a ~20-line estimator; the arithmetic is the
    standard product-limit form,
        CIF_k(t) = sum over event times u <= t of  S(u-) * d_k(u) / n(u).
    """
    time = np.asarray(time, dtype=float)
    cause = np.asarray(cause, dtype=int)

    order = np.argsort(time, kind="stable")
    time, cause = time[order], cause[order]

    n_at_risk = len(time)
    surv = 1.0
    cif = {1: 0.0, 2: 0.0}
    curve = []

    for t in np.unique(time):
        at_t = time == t
        n_events_1 = int(np.sum(at_t & (cause == 1)))
        n_events_2 = int(np.sum(at_t & (cause == 2)))
        n_any = int(np.sum(at_t))

        if n_at_risk > 0 and (n_events_1 or n_events_2):
            cif[1] += surv * n_events_1 / n_at_risk
            cif[2] += surv * n_events_2 / n_at_risk
            surv *= 1 - (n_events_1 + n_events_2) / n_at_risk

        n_at_risk -= n_any
        curve.append({"time": float(t), "cif_ckd": cif[1], "cif_death": cif[2],
                      "event_free_survival": surv})

    return {
        "cif_ckd_at_horizon": float(cif[1]),
        "cif_death_at_horizon": float(cif[2]),
        "event_free_survival_at_horizon": float(surv),
        "curve": curve,
        "horizon_days": horizon,
    }


# ---------------------------------------------------------------------------
# Helpers and orchestration
# ---------------------------------------------------------------------------

def _died_within(features_df, horizon_days):
    """patient_id -> died within the follow-up horizon."""
    df = features_df.copy()
    df["death_date"] = pd.to_datetime(df["death_date"], errors="coerce")
    df["discharge_date"] = pd.to_datetime(df["discharge_date"], errors="coerce")
    days = (df["death_date"] - df["discharge_date"]).dt.days
    return pd.DataFrame({
        "patient_id": df["patient_id"].astype(str),
        "died_in_followup": (df["death_date"].notna() & (days <= horizon_days)).to_numpy(),
    })


def run_competing_risk_analysis(experiment_predictions, features_df, output_dir,
                                threshold=0.20, horizon_days=365,
                                outcome_date_col=None):
    """Run every competing-risk analysis for every experiment and save results.

    `experiment_predictions` is {label: DataFrame} from predictions.load_many().
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n{'=' * 70}\nCOMPETING RISK OF DEATH  (editor point 2)\n{'=' * 70}")

    results = {
        "mortality": describe_mortality(features_df, horizon_days),
        "cumulative_incidence": cumulative_incidence(
            features_df, outcome_date_col, horizon_days),
        "by_experiment": {},
    }

    bounds_rows = []
    for label, preds in experiment_predictions.items():
        print(f"\n--- {label} ---")
        cc = complete_case_analysis(preds, features_df, threshold, horizon_days)
        bd = bounds_analysis(preds, features_df, threshold, horizon_days)
        tp = tipping_point_analysis(preds, features_df, threshold, horizon_days)

        tp["sweep"].to_csv(
            os.path.join(output_dir, f"tipping_point_{label}.csv"), index=False)

        results["by_experiment"][label] = {
            "complete_case": cc,
            "bounds": bd,
            "tipping": {k: v for k, v in tp.items() if k != "sweep"},
        }

        row = {"experiment": label}
        for metric in ("prevalence", "sensitivity", "specificity", "ppv",
                       "npv", "alert_rate"):
            row[f"{metric}_reported"] = bd["assume_none_progressed"][metric]
            row[f"{metric}_survivors_only"] = cc["survivors_only"][metric]
            lo, hi = bd["bounds"].get(metric, (np.nan, np.nan))
            row[f"{metric}_bound_lo"] = lo
            row[f"{metric}_bound_hi"] = hi
        bounds_rows.append(row)

    summary = pd.DataFrame(bounds_rows)
    summary.to_csv(os.path.join(output_dir, "competing_risk_summary.csv"), index=False)

    with open(os.path.join(output_dir, "competing_risk.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n[CompetingRisk] Written to {output_dir}")
    return results
