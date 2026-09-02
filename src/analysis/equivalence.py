"""
Formal equivalence and non-inferiority testing (editor's point 7c).

> "the equivalence claim should rest on a formal non-inferiority or equivalence
>  framework with a pre-specified margin rather than on non-significant P
>  values"

The statistical problem
-----------------------
The submitted manuscript argues that complex models offer no advantage from a
set of non-significant p values. Absence of evidence is not evidence of
absence: with 286 events, a non-significant difference is equally consistent
with no difference and with a difference the study was underpowered to detect.
An equivalence framework inverts the burden -- the null becomes "the models
DIFFER by at least the margin", and rejecting it is positive evidence for
equivalence, which is what the paper actually wants to claim.

Procedure
---------
Two one-sided tests (TOST). For a difference d with standard error se and a
pre-specified margin delta, equivalence at level alpha is declared when the
(1 - 2*alpha) confidence interval for d lies entirely inside
[-delta, +delta]. That is exactly equivalent to both one-sided tests rejecting,
and it is the form to report because the interval is interpretable on its own.

Non-inferiority of the simple model is the one-sided version: the complex model
is not better by more than delta, i.e. the upper bound of the one-sided
interval is below +delta.

The margin comes from `config.EQUIVALENCE_MARGIN_AUROC`, which is declared in
configuration BEFORE any analysis runs. That ordering is the point -- a margin
chosen after seeing the intervals is not pre-specified.

Variance
--------
Fold-level differences from cross-validation are positively correlated, because
any two folds share most of their training data, so a naive paired t test is
anti-conservative. The Nadeau-Bengio correction multiplies the sample variance
by (1/k + n_test/n_train). Using the corrected variance for the TOST interval
keeps this module consistent with the pairwise tests the evaluation suite
already produces.
"""

from __future__ import annotations

import os
import json
import glob

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Corrected variance
# ---------------------------------------------------------------------------

def nadeau_bengio_interval(diffs, n_test, n_train, ci_level=0.95):
    """(mean difference, standard error, ci_lo, ci_hi) with the CV correction.

    Mirrors `bengio_nadeau_test` in the evaluation suite; `verify_against_eval_suite()`
    asserts the two agree when the suite is importable.
    """
    diffs = np.asarray(diffs, dtype=float)
    k = len(diffs)
    if k < 2:
        return np.nan, np.nan, np.nan, np.nan

    mean_d = float(np.mean(diffs))
    var_d = float(np.var(diffs, ddof=1))
    corrected_var = (1.0 / k + n_test / n_train) * var_d
    if corrected_var <= 0:
        return mean_d, 0.0, mean_d, mean_d

    se = float(np.sqrt(corrected_var))
    t_crit = stats.t.ppf(1 - (1 - ci_level) / 2, df=k - 1)
    return mean_d, se, mean_d - t_crit * se, mean_d + t_crit * se


def verify_against_eval_suite(seed=0, k=10):
    """Check the corrected variance matches the evaluation suite's."""
    try:
        from core_eval_functions import bengio_nadeau_test
    except ImportError:
        return False

    rng = np.random.default_rng(seed)
    diffs = rng.normal(0.01, 0.02, k)
    n_test, n_train = 469.0, 4218.0

    _, se, lo, hi = nadeau_bengio_interval(diffs, n_test, n_train, 0.95)
    _, _, lo2, hi2 = bengio_nadeau_test(list(diffs), n_test, n_train, 0.95)

    if not (np.isclose(lo, lo2, atol=1e-10) and np.isclose(hi, hi2, atol=1e-10)):
        raise AssertionError(
            "nadeau_bengio_interval has drifted from the evaluation suite's "
            "bengio_nadeau_test. The equivalence intervals and the pairwise "
            "comparison table would disagree."
        )
    print("[Equivalence] Nadeau-Bengio interval verified against the evaluation suite.")
    return True


# ---------------------------------------------------------------------------
# TOST
# ---------------------------------------------------------------------------

def tost(diffs, margin, n_test, n_train, ci_level=0.95):
    """Two one-sided tests for equivalence within +/- margin.

    `diffs` are per-fold (comparison - baseline) differences on a metric where
    higher is better, so a positive difference favours the comparison model.

    Three separate one-sided readings are returned, because they answer
    different questions and conflating them is the easiest way to overclaim:

      not_superior  ci_hi < +margin   the comparison model is not BETTER than
                                      the baseline by more than the margin.
                                      *This is the manuscript's actual claim* --
                                      that added complexity buys nothing.
      not_inferior  ci_lo > -margin   the comparison model is not WORSE than the
                                      baseline by more than the margin.
      equivalent    both of the above  the difference is inside the margin in
                                      both directions.

    A model whose whole interval sits below -margin satisfies `not_superior`
    while being materially worse, so `not_superior` alone must never be
    reported as evidence of equivalence. `verdict` disambiguates.
    """
    diffs = np.asarray(diffs, dtype=float)
    k = len(diffs)
    mean_d, se, ci_lo, ci_hi = nadeau_bengio_interval(diffs, n_test, n_train, ci_level)

    if not np.isfinite(se) or se == 0:
        equivalent = bool(abs(mean_d) < margin)
        return {"k_folds": k, "mean_difference": mean_d, "se": se,
                "ci_lo": ci_lo, "ci_hi": ci_hi, "margin": margin,
                "p_lower": np.nan, "p_upper": np.nan, "p_tost": np.nan,
                "equivalent": equivalent,
                "not_superior": bool(mean_d < margin),
                "not_inferior": bool(mean_d > -margin),
                "verdict": _verdict(equivalent, bool(mean_d < margin),
                                    bool(mean_d > -margin), mean_d, margin),
                "two_sided_p": np.nan, "two_sided_significant": False,
                "note": "zero variance across folds"}

    df = k - 1
    # H0_lower: d <= -margin  ;  H0_upper: d >= +margin
    t_lower = (mean_d + margin) / se
    t_upper = (mean_d - margin) / se
    p_lower = float(stats.t.sf(t_lower, df))
    p_upper = float(stats.t.cdf(t_upper, df))
    p_tost = max(p_lower, p_upper)

    two_sided_p = float(2 * stats.t.sf(abs(mean_d / se), df))

    not_superior = bool(ci_hi < margin)
    not_inferior = bool(ci_lo > -margin)
    equivalent = not_superior and not_inferior

    return {
        "k_folds": k,
        "mean_difference": mean_d,
        "se": se,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "margin": margin,
        "p_lower": p_lower,
        "p_upper": p_upper,
        "p_tost": p_tost,
        "equivalent": equivalent,
        "not_superior": not_superior,
        "not_inferior": not_inferior,
        "verdict": _verdict(equivalent, not_superior, not_inferior, mean_d, margin,
                            ci_lo, ci_hi),
        "two_sided_p": two_sided_p,
        "two_sided_significant": bool(two_sided_p < 0.05),
    }


def _verdict(equivalent, not_superior, not_inferior, mean_d, margin,
             ci_lo=None, ci_hi=None):
    """One unambiguous label per comparison.

    Kept deliberately blunt: WORSE and BETTER are named as such rather than
    dressed up as one-sided successes, because the whole point of the exercise
    is to stop a non-result being read as a positive one.
    """
    if equivalent:
        return "EQUIVALENT"
    if ci_hi is not None and ci_hi < -margin:
        return "WORSE than baseline by more than the margin"
    if ci_lo is not None and ci_lo > margin:
        return "BETTER than baseline by more than the margin"
    if not_superior:
        return "NOT SUPERIOR (no benefit shown; inferiority not ruled out)"
    if not_inferior:
        return "NOT INFERIOR (superiority not ruled out)"
    return "INCONCLUSIVE"


# ---------------------------------------------------------------------------
# Fold metric loading
# ---------------------------------------------------------------------------

def load_fold_metrics(experiment_dir, metric="roc_auc"):
    """Per-fold values of one metric, ordered by fold number.

    Reads the `fold_N.json` files the CV engine writes.
    """
    values = {}
    for path in glob.glob(os.path.join(experiment_dir, "fold_*.json")):
        base = os.path.basename(path)
        if "predictions" in base or "model" in base:
            continue
        try:
            fold_num = int(base.replace("fold_", "").replace(".json", ""))
        except ValueError:
            continue
        with open(path) as f:
            data = json.load(f)
        if metric in data:
            values[fold_num] = float(data[metric])

    if not values:
        raise FileNotFoundError(
            f"No fold_N.json carrying {metric!r} in {experiment_dir}")
    return [values[k] for k in sorted(values)], sorted(values)


def _fold_sizes(experiment_dir):
    """(mean n_test, mean n_train) across folds, for the correction factor."""
    n_tests, n_trains = [], []
    for path in glob.glob(os.path.join(experiment_dir, "fold_*.json")):
        base = os.path.basename(path)
        if "predictions" in base or "model" in base:
            continue
        with open(path) as f:
            data = json.load(f)
        if "n" in data:
            n_tests.append(float(data["n"]))
        elif "accuracy" in data:
            pass
        if "n_train" in data:
            n_trains.append(float(data["n_train"]))

    n_test = float(np.mean(n_tests)) if n_tests else np.nan
    if n_trains:
        n_train = float(np.mean(n_trains))
        source = "measured"
    else:
        # Standard k-fold identity when training sizes were not recorded.
        k = max(len(n_tests), 2)
        n_train = n_test * (k - 1) if np.isfinite(n_test) else np.nan
        source = "assumed"
    return n_test, n_train, source


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_equivalence_analysis(experiment_dirs, baseline_name, output_dir,
                             margin, metric="roc_auc", ci_level=0.95,
                             labels=None):
    """TOST every experiment against the baseline on `metric`.

    A positive mean difference favours the comparison model.
    """
    os.makedirs(output_dir, exist_ok=True)
    labels = labels or {}

    print(f"\n{'=' * 70}\nEQUIVALENCE TESTING  (editor point 7c)\n{'=' * 70}")
    print(f"Metric              : {metric}")
    print(f"Pre-specified margin: +/- {margin}")
    print(f"Baseline            : {labels.get(baseline_name, baseline_name)}")
    print("Null hypothesis     : the models differ by at least the margin.")
    print("Rejecting it is positive evidence of equivalence, which a "
          "non-significant\n                      difference test is not.\n")

    if baseline_name not in experiment_dirs:
        raise KeyError(f"Baseline {baseline_name!r} not among "
                       f"{sorted(experiment_dirs)}")

    base_values, base_folds = load_fold_metrics(experiment_dirs[baseline_name], metric)
    n_test, n_train, size_source = _fold_sizes(experiment_dirs[baseline_name])

    rows = []
    for name, path in experiment_dirs.items():
        if name == baseline_name:
            continue
        try:
            comp_values, comp_folds = load_fold_metrics(path, metric)
        except FileNotFoundError as exc:
            print(f"  skipping {name}: {exc}")
            continue

        if comp_folds != base_folds:
            print(f"  skipping {name}: fold numbering differs from the baseline "
                  f"({comp_folds} vs {base_folds})")
            continue

        diffs = np.asarray(comp_values) - np.asarray(base_values)
        result = tost(diffs, margin, n_test, n_train, ci_level)

        verdict = result["verdict"]
        rows.append({
            "comparison": labels.get(name, name),
            "experiment": name,
            "baseline": baseline_name,
            "metric": metric,
            "mean_difference": result["mean_difference"],
            "ci_lo": result["ci_lo"],
            "ci_hi": result["ci_hi"],
            "margin": margin,
            "p_tost": result["p_tost"],
            "equivalent": result["equivalent"],
            "not_superior": result["not_superior"],
            "not_inferior": result["not_inferior"],
            "two_sided_p": result["two_sided_p"],
            "verdict": verdict,
            "n_test_per_fold": n_test,
            "n_train_per_fold": n_train,
            "n_train_source": size_source,
            "formatted": (f"{result['mean_difference']:+.4f} "
                          f"({int(ci_level * 100)}% CI {result['ci_lo']:+.4f} to "
                          f"{result['ci_hi']:+.4f})"),
        })

        print(f"  {labels.get(name, name):<42s} "
              f"d={result['mean_difference']:+.4f} "
              f"[{result['ci_lo']:+.4f}, {result['ci_hi']:+.4f}]  "
              f"p={result['p_tost']:.3f}  {verdict}")

    table = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, f"equivalence_{metric}.csv")
    table.to_csv(csv_path, index=False)

    n_equiv = int(table["equivalent"].sum()) if len(table) else 0
    n_not_sup = int(table["not_superior"].sum()) if len(table) else 0
    n_incon = int((table["verdict"] == "INCONCLUSIVE").sum()) if len(table) else 0
    n_worse = int(table["verdict"].str.startswith("WORSE").sum()) if len(table) else 0

    print(f"\n  {n_equiv} of {len(table)} comparison(s) met formal EQUIVALENCE "
          f"within +/- {margin}.")
    print(f"  {n_not_sup} showed NO SUPERIORITY over the baseline "
          f"(the manuscript's claim).")
    if n_worse:
        print(f"  {n_worse} were WORSE than the baseline by more than the margin — "
              f"note that these\n  also satisfy 'not superior', which is why the two "
              f"must be reported separately.")
    if n_incon:
        print(f"  {n_incon} INCONCLUSIVE: the interval extends beyond the margin in "
              f"both directions,\n  so the study cannot rule out a difference that "
              f"size. Report that honestly\n  rather than as equivalence.")

    summary = {
        "metric": metric, "margin": margin, "ci_level": ci_level,
        "baseline": baseline_name,
        "n_comparisons": len(table),
        "n_equivalent": n_equiv,
        "n_not_superior": n_not_sup,
        "n_not_inferior": int(table["not_inferior"].sum()) if len(table) else 0,
        "n_worse_than_margin": n_worse,
        "n_inconclusive": n_incon,
        "csv": csv_path,
    }
    with open(os.path.join(output_dir, f"equivalence_{metric}.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[Equivalence] -> {csv_path}")
    return table
