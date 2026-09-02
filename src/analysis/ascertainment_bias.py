#!/usr/bin/env python3
"""
Quantitative bias analysis for outcome ascertainment (editor's point 3),
without a follow-up laboratory extract.

The situation
-------------
The editor asked how many patients had no outpatient eGFR during follow-up and
how they compare with the tested group. Answering that literally needs a
post-discharge lab extract, and no such extract exists on the analysis server:
all 519,785 in-hospital lab rows fall between admission and discharge, and the
pre-index file is entirely pre-admission. A new AHS extract would take weeks to
months.

What can be done instead is arguably more informative than the descriptive
comparison that was requested, because it bounds the CONSEQUENCE of the
misclassification rather than describing the group it came from.

Three analyses, none of which need new data
-------------------------------------------
1. `analytic_correction` -- closed-form. If ascertainment of the outcome is
   independent of the model's predicted risk, and false positives are
   negligible (two outpatient eGFR values below 30, three months apart, is
   unlikely to be spurious), then for an ascertainment sensitivity Se:

       prevalence_true = prevalence_observed / Se
       PPV_true        = PPV_observed        / Se
       model sensitivity is UNBIASED
       specificity is attenuated only slightly

   That last point is the useful one. The manuscript's claim is a *comparison
   between models* at a fixed threshold, and non-differential outcome
   misclassification attenuates every model's PPV by the same factor Se. It
   therefore cannot reverse a ranking, and it cannot manufacture or hide a
   difference in NRI. The editor's own reading agrees on direction:
   "unascertained progressors counted as negatives most directly depress
   positive predictive value and the apparent event rate".

2. `probabilistic_bias_analysis` -- simulation, relaxing the independence
   assumption. Missed progressors are restored under three mechanisms:

     risk_independent  ascertainment unrelated to the model's prediction
     risk_increasing   sicker patients are more likely to be tested, so the
                       progressors who were missed sit at LOW predicted risk
                       (they become false negatives: sensitivity falls, PPV
                       barely moves)
     risk_decreasing   the opposite, included for completeness: missed
                       progressors sit at HIGH predicted risk (they were
                       flagged, so they were being counted as false positives:
                       PPV rises sharply)

   Reported across a grid of Se. The realistic mechanism is risk_increasing,
   since being tested is itself a marker of clinical concern, and it is also
   the one that is unfavourable to the models -- so it is the honest one to
   foreground.

3. `engagement_strata` -- the empirical substitute for the tested-versus-
   untested comparison. Whether a patient will be tested after discharge
   cannot be observed, but their prior engagement with outpatient testing can:
   the count of pre-index laboratory measurements is in the feature set
   already. Patients with no pre-index outpatient labs have no established
   testing pattern and are the least likely to be measured afterwards.

   If the observed event rate is flat across engagement strata, ascertainment
   bias is bounded and small. If it rises steeply, the gradient measures it.
   Either way it is an answer computed from data in hand, and it is the closest
   available analogue to the comparison the editor asked for.

References for the response letter: Lash, Fox & Fink, *Applying Quantitative
Bias Analysis to Epidemiologic Data* (Springer, 2009), chapters on outcome
misclassification and probabilistic bias analysis.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from src.analysis.predictions import threshold_metrics


DEFAULT_SE_GRID = (1.00, 0.95, 0.90, 0.85, 0.80, 0.70, 0.60, 0.50)


# ---------------------------------------------------------------------------
# 1. Closed-form correction
# ---------------------------------------------------------------------------

def analytic_correction(predictions, threshold, se_grid=DEFAULT_SE_GRID):
    """Corrected prevalence and PPV under risk-independent under-ascertainment.

    Derivation, with A the ascertainment indicator and Se = P(A=1 | Y=1):

        observed events      E_obs  = Se * E_true
        observed true pos    TP_obs = Se * TP_true
        false positives      FP_obs = FP_true + (1 - Se) * TP_true

        PPV_obs = Se*TP_true / (Se*TP_true + FP_true + (1-Se)*TP_true)
                = Se*TP_true / (TP_true + FP_true)
                = Se * PPV_true

    and model sensitivity TP_obs/E_obs = TP_true/E_true is unchanged. The alert
    rate is unchanged too, since it depends only on the predictions.
    """
    y = predictions["y_true"].to_numpy().astype(int)
    p = predictions["y_proba"].to_numpy(dtype=float)
    obs = threshold_metrics(y, p, threshold)

    rows = []
    for se in se_grid:
        rows.append({
            "assumed_ascertainment_sensitivity": se,
            "n_missed_progressors": int(round(obs["n_events"] * (1 / se - 1))),
            "prevalence_observed": obs["prevalence"],
            "prevalence_corrected": min(obs["prevalence"] / se, 1.0),
            "ppv_observed": obs["ppv"],
            "ppv_corrected": min(obs["ppv"] / se, 1.0) if np.isfinite(obs["ppv"]) else np.nan,
            "sensitivity_unbiased": obs["sensitivity"],
            "alert_rate_unchanged": obs["alert_rate"],
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. Probabilistic bias analysis
# ---------------------------------------------------------------------------

def _restore_weights(p_nonevent, mechanism, strength=2.0):
    """Relative probability that an observed non-event is a missed progressor.

    Under any mechanism the weight is proportional to the patient's estimated
    risk -- a missed progressor is still more likely to be someone the model
    scored highly -- multiplied by the probability that ascertainment FAILED
    for them, which is what the mechanism controls.
    """
    p = np.clip(p_nonevent, 1e-6, 1 - 1e-6)
    if mechanism == "risk_independent":
        fail = np.ones_like(p)
    elif mechanism == "risk_increasing":
        # Testing more likely as risk rises => failure more likely at low risk
        fail = (1 - p) ** strength
    elif mechanism == "risk_decreasing":
        fail = p ** strength
    else:
        raise ValueError(f"Unknown mechanism {mechanism!r}")
    w = p * fail
    total = w.sum()
    return w / total if total > 0 else np.full_like(p, 1 / len(p))


def probabilistic_bias_analysis(predictions, threshold, se_grid=DEFAULT_SE_GRID,
                                mechanisms=("risk_independent", "risk_increasing",
                                            "risk_decreasing"),
                                n_draws=200, seed=1202):
    """Metrics after restoring missed progressors under stated mechanisms."""
    y = predictions["y_true"].to_numpy().astype(int)
    p = predictions["y_proba"].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)

    nonevent_idx = np.where(y == 0)[0]
    n_events = int(y.sum())

    rows = []
    for mechanism in mechanisms:
        weights = _restore_weights(p[nonevent_idx], mechanism)
        for se in se_grid:
            n_missed = int(round(n_events * (1 / se - 1)))
            n_missed = min(n_missed, len(nonevent_idx))

            if n_missed == 0:
                draws = [threshold_metrics(y, p, threshold)]
            else:
                draws = []
                for _ in range(n_draws):
                    pick = rng.choice(nonevent_idx, size=n_missed,
                                      replace=False, p=weights)
                    y_star = y.copy()
                    y_star[pick] = 1
                    draws.append(threshold_metrics(y_star, p, threshold))

            row = {"mechanism": mechanism,
                   "assumed_ascertainment_sensitivity": se,
                   "n_restored": n_missed}
            for metric in ("prevalence", "sensitivity", "specificity", "ppv",
                           "npv", "alert_rate", "net_benefit"):
                vals = [d[metric] for d in draws if np.isfinite(d[metric])]
                row[metric] = float(np.mean(vals)) if vals else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def ranking_stability(experiment_predictions, threshold, se_grid=DEFAULT_SE_GRID,
                      mechanism="risk_increasing", n_draws=100, seed=1202):
    """Does under-ascertainment ever reorder the models?

    This is the question that actually matters. The manuscript does not claim
    an absolute risk; it claims that complex models do not beat the James score
    at the referral threshold. That claim survives outcome misclassification as
    long as the ORDER of the models is stable, which is what this measures.

    The same restored outcome vector is applied to every model, because they
    scored the same patients -- restoring different progressors per model would
    be comparing different datasets.
    """
    names = list(experiment_predictions)
    base = experiment_predictions[names[0]][["patient_id", "y_true"]].copy()
    y = base["y_true"].to_numpy().astype(int)

    aligned = {}
    for name, df in experiment_predictions.items():
        merged = base.merge(df[["patient_id", "y_proba"]], on="patient_id", how="left")
        aligned[name] = merged["y_proba"].to_numpy(dtype=float)

    mean_p = np.mean(np.vstack(list(aligned.values())), axis=0)
    nonevent_idx = np.where(y == 0)[0]
    weights = _restore_weights(mean_p[nonevent_idx], mechanism)
    n_events = int(y.sum())
    rng = np.random.default_rng(seed)

    rows = []
    for se in se_grid:
        n_missed = min(int(round(n_events * (1 / se - 1))), len(nonevent_idx))
        orders = []
        per_model = {name: [] for name in names}

        for _ in range(1 if n_missed == 0 else n_draws):
            y_star = y.copy()
            if n_missed:
                y_star[rng.choice(nonevent_idx, size=n_missed,
                                  replace=False, p=weights)] = 1
            scores = {}
            for name, proba in aligned.items():
                m = threshold_metrics(y_star, proba, threshold)
                scores[name] = m["ppv"] if np.isfinite(m["ppv"]) else -1.0
                per_model[name].append(scores[name])
            orders.append(tuple(sorted(scores, key=scores.get, reverse=True)))

        modal = max(set(orders), key=orders.count)
        rows.append({
            "assumed_ascertainment_sensitivity": se,
            "n_restored": n_missed,
            "distinct_orderings": len(set(orders)),
            "modal_ordering_share": orders.count(modal) / len(orders),
            "best_model": modal[0],
            "modal_ordering": " > ".join(modal),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Healthcare-engagement strata
# ---------------------------------------------------------------------------

def engagement_strata(predictions, features_df, threshold,
                      bins=(0, 1, 6, 21, np.inf),
                      labels=("none", "1-5", "6-20", "21+")):
    """Event rate and model performance by prior outpatient testing volume.

    The empirical stand-in for the tested-versus-untested comparison. Whether a
    patient WILL be tested after discharge is unobserved, but how much
    outpatient testing they had BEFORE admission is in the feature set, and it
    is a direct measure of engagement with outpatient care.

    Read it as follows. A flat observed event rate across strata means
    ascertainment is not strongly tied to engagement, which bounds this bias.
    A steep gradient measures it -- and the low-engagement stratum is the group
    the editor was asking about.
    """
    pre_cols = [c for c in features_df.columns
                if c.startswith("pre-index_labs_count:")]
    if not pre_cols:
        return None, {"status": "skipped",
                      "reason": "no pre-index_labs_count:* columns in features_df; "
                                "run the ETL so the pre-index lab crosstab is merged"}

    engagement = features_df[pre_cols].fillna(0).sum(axis=1)
    frame = pd.DataFrame({
        "patient_id": features_df["patient_id"].astype(str),
        "n_pre_index_labs": engagement.to_numpy(),
    })
    frame["engagement_stratum"] = pd.cut(frame["n_pre_index_labs"], bins=bins,
                                         labels=labels, right=False)

    merged = predictions.merge(frame, on="patient_id", how="left")

    rows = []
    for stratum in labels:
        sub = merged[merged["engagement_stratum"] == stratum]
        if sub.empty:
            continue
        m = threshold_metrics(sub["y_true"], sub["y_proba"], threshold)
        rows.append({
            "engagement_stratum": stratum,
            "n_patients": m["n"],
            "pct_of_cohort": 100 * m["n"] / len(merged),
            "n_events": m["n_events"],
            "observed_event_rate": m["prevalence"],
            "sensitivity": m["sensitivity"],
            "ppv": m["ppv"],
            "alert_rate": m["alert_rate"],
        })

    table = pd.DataFrame(rows)
    rates = table["observed_event_rate"].to_numpy()
    status = {
        "status": "ok",
        "n_pre_index_lab_columns": len(pre_cols),
        "event_rate_lowest_engagement": float(rates[0]) if len(rates) else None,
        "event_rate_highest_engagement": float(rates[-1]) if len(rates) else None,
        "relative_risk_high_vs_low": (float(rates[-1] / rates[0])
                                      if len(rates) and rates[0] > 0 else None),
    }
    return table, status


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_ascertainment_bias_analysis(experiment_predictions, features_df,
                                    output_dir, threshold=0.20,
                                    se_grid=DEFAULT_SE_GRID):
    """Run all three analyses and write the tables."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n{'=' * 70}")
    print("OUTCOME ASCERTAINMENT — QUANTITATIVE BIAS ANALYSIS  (editor point 3)")
    print(f"{'=' * 70}")
    print("No post-discharge lab extract exists, so who was tested cannot be")
    print("identified. These analyses bound the CONSEQUENCE of the resulting")
    print("misclassification instead, using only data in hand.")

    summary = {"threshold": threshold, "se_grid": list(se_grid)}

    # --- 3. engagement strata (report first: it is the empirical one) -------
    strata, strata_status = engagement_strata(
        experiment_predictions[next(iter(experiment_predictions))],
        features_df, threshold)
    summary["engagement"] = strata_status

    print(f"\n--- Observed event rate by prior outpatient testing volume ---")
    if strata is None:
        print(f"  SKIPPED: {strata_status['reason']}")
    else:
        strata.to_csv(os.path.join(output_dir, "engagement_strata.csv"), index=False)
        print(f"  {'stratum':<10s} {'n':>7s} {'events':>7s} {'rate':>8s} "
              f"{'PPV':>8s} {'sens':>8s}")
        for _, r in strata.iterrows():
            print(f"  {r['engagement_stratum']:<10s} {int(r['n_patients']):>7,} "
                  f"{int(r['n_events']):>7,} {r['observed_event_rate']:>8.4f} "
                  f"{r['ppv']:>8.3f} {r['sensitivity']:>8.3f}")
        rr = strata_status.get("relative_risk_high_vs_low")
        if rr:
            print(f"\n  event rate, highest vs lowest engagement: {rr:.2f}x")
            if rr < 1.5:
                print("  A shallow gradient bounds ascertainment bias: patients with no")
                print("  prior outpatient testing do not show a materially lower event")
                print("  rate, which is what under-ascertainment in that group would cause.")
            else:
                print("  A steep gradient is consistent with under-ascertainment among")
                print("  patients less engaged with outpatient care. Report it as such,")
                print("  and use the bias analysis below to bound the consequence.")

    # --- 1. analytic correction --------------------------------------------
    print(f"\n--- Closed-form correction (ascertainment independent of risk) ---")
    for name, preds in experiment_predictions.items():
        table = analytic_correction(preds, threshold, se_grid)
        table.insert(0, "experiment", name)
        table.to_csv(os.path.join(output_dir, f"analytic_correction_{name}.csv"),
                     index=False)
    baseline = next(iter(experiment_predictions))
    shown = analytic_correction(experiment_predictions[baseline], threshold, se_grid)
    print(f"  ({baseline})")
    print(f"  {'Se':>6s} {'missed':>8s} {'prev_obs':>9s} {'prev_true':>10s} "
          f"{'PPV_obs':>9s} {'PPV_true':>9s}")
    for _, r in shown.iterrows():
        print(f"  {r['assumed_ascertainment_sensitivity']:>6.2f} "
              f"{int(r['n_missed_progressors']):>8,} "
              f"{r['prevalence_observed']:>9.4f} {r['prevalence_corrected']:>10.4f} "
              f"{r['ppv_observed']:>9.3f} {r['ppv_corrected']:>9.3f}")
    print("\n  Model sensitivity and the alert rate are unbiased under this")
    print("  mechanism; only prevalence and PPV are attenuated, both by 1/Se.")

    # --- 2. probabilistic bias analysis ------------------------------------
    print(f"\n--- Simulation, relaxing the independence assumption ---")
    frames = []
    for name, preds in experiment_predictions.items():
        pba = probabilistic_bias_analysis(preds, threshold, se_grid)
        pba.insert(0, "experiment", name)
        frames.append(pba)
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(os.path.join(output_dir, "probabilistic_bias_analysis.csv"),
                    index=False)

    focus = combined[(combined["experiment"] == baseline)
                     & (combined["mechanism"] == "risk_increasing")]
    print(f"  ({baseline}, mechanism = risk_increasing, the realistic case)")
    print(f"  {'Se':>6s} {'restored':>9s} {'prev':>8s} {'sens':>8s} "
          f"{'spec':>8s} {'PPV':>8s}")
    for _, r in focus.iterrows():
        print(f"  {r['assumed_ascertainment_sensitivity']:>6.2f} "
              f"{int(r['n_restored']):>9,} {r['prevalence']:>8.4f} "
              f"{r['sensitivity']:>8.3f} {r['specificity']:>8.3f} {r['ppv']:>8.3f}")

    # --- ranking stability --------------------------------------------------
    if len(experiment_predictions) > 1:
        print(f"\n--- Does under-ascertainment ever reorder the models? ---")
        ranks = ranking_stability(experiment_predictions, threshold, se_grid)
        ranks.to_csv(os.path.join(output_dir, "ranking_stability.csv"), index=False)
        for _, r in ranks.iterrows():
            print(f"  Se={r['assumed_ascertainment_sensitivity']:.2f}  "
                  f"{r['distinct_orderings']:>2d} distinct ordering(s), "
                  f"modal share {r['modal_ordering_share']:.0%}, "
                  f"best = {r['best_model']}")
        best = set(ranks["best_model"])
        if len(best) == 1:
            print(f"\n  The same model ranks best at every assumed ascertainment")
            print(f"  sensitivity down to {min(se_grid):.2f}. The manuscript's")
            print(f"  comparative claim is therefore robust to this bias, which is")
            print(f"  the claim being made -- no absolute risk is asserted.")
        else:
            print(f"\n  The best model CHANGES across the range ({sorted(best)}).")
            print(f"  Report that honestly: the comparison is sensitive to outcome")
            print(f"  ascertainment and the conclusion should be qualified.")
        summary["ranking_best_models"] = sorted(best)

    with open(os.path.join(output_dir, "ascertainment_bias.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n[AscertainmentBias] Written to {output_dir}")
    return summary
