"""
SUPERSEDED -- do not use. Kept only so the pre-resubmission history is readable.

This module computed NRI on the models' RAW probabilities with a
population-SD (ddof=0) fold spread. The published Table 5 was produced on
RECALIBRATED probabilities with a sample SD (ddof=1), by `nri.py` in the
lancet-digital-health-eval-suite. The two therefore do not agree, and shipping
this file on the analysis path meant the repository cited in the paper did not
reproduce the paper's own table.

NRI now lives in one place:

    python nri.py \
        --baseline_dir <results>/<timestamp>_logreg_james_score_fold_results \
        --ordering example_ordering.json \
        --threshold 0.20 --recalibrate --bootstrap 2000 \
        --output_csv reports/nri_table.csv

Recalibration there and in src/analysis/predictions.py are the same procedure,
and predictions.verify_against_eval_suite() asserts they stay that way.
"""

"""
Net Reclassification Improvement (NRI) analysis.

Compares how well two models reclassify patients across risk categories
using cross-validation fold predictions.
"""

import os
import json
import numpy as np

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config


def calculate_nri(y_true, y_pred_model1, y_pred_model2, thresholds):
    """Calculate NRI between two sets of predicted probabilities.

    Parameters
    ----------
    y_true : np.ndarray — true outcomes (0/1)
    y_pred_model1 : np.ndarray — baseline model probabilities
    y_pred_model2 : np.ndarray — new model probabilities
    thresholds : list of float — sorted ascending, e.g. [0.2]

    Returns
    -------
    dict with NRI_total, NRI_cases, NRI_controls, and reclassification counts
    """
    def categorize_risk(probs, thresholds):
        categories = np.zeros(len(probs), dtype=int)
        for i, t in enumerate(thresholds):
            categories[probs >= t] = i + 1
        return categories

    cat1 = categorize_risk(y_pred_model1, thresholds)
    cat2 = categorize_risk(y_pred_model2, thresholds)

    cases = y_true == 1
    controls = y_true == 0

    cases_up = int(np.sum(cat2[cases] > cat1[cases]))
    cases_down = int(np.sum(cat2[cases] < cat1[cases]))
    cases_total = int(np.sum(cases))

    controls_up = int(np.sum(cat2[controls] > cat1[controls]))
    controls_down = int(np.sum(cat2[controls] < cat1[controls]))
    controls_total = int(np.sum(controls))

    nri_cases = (cases_up - cases_down) / cases_total if cases_total > 0 else 0
    nri_controls = (controls_down - controls_up) / controls_total if controls_total > 0 else 0

    return {
        "NRI_total": nri_cases + nri_controls,
        "NRI_cases": nri_cases,
        "NRI_controls": nri_controls,
        "cases_up": cases_up,
        "cases_down": cases_down,
        "controls_up": controls_up,
        "controls_down": controls_down,
        "cases_total": cases_total,
        "controls_total": controls_total,
        "movement": np.sign(cat2 - cat1).astype(float),
    }


def _load_fold_predictions(experiment_path, fold_num):
    """Load fold_N_predictions.json from an experiment directory."""
    path = os.path.join(experiment_path, f"fold_{fold_num}_predictions.json")
    with open(path) as f:
        return json.load(f)


def _nri_pooled_ci(y_true, movement, n_boot=2000, ci_level=0.95, seed=0):
    """Percentile bootstrap CI for pooled NRI. Assumes one row per patient."""
    y = np.asarray(y_true) == 1
    s = np.asarray(movement, dtype=float)
    rng = np.random.default_rng(seed)

    def nri(idx):
        yy, ss = y[idx], s[idx]
        if yy.sum() == 0 or yy.all():
            return np.nan
        return ss[yy].mean() - ss[~yy].mean()

    draws = np.array([nri(rng.integers(0, y.size, y.size)) for _ in range(n_boot)])
    draws = draws[np.isfinite(draws)]
    lo, hi = (1 - ci_level) / 2 * 100, (1 + ci_level) / 2 * 100
    return float(np.percentile(draws, lo)), float(np.percentile(draws, hi))


def calculate_nri_across_folds(exp1_path, exp2_path, thresholds, n_folds=10):
    """Calculate NRI across all CV folds.

    Parameters
    ----------
    exp1_path : str — path to baseline experiment folder
    exp2_path : str — path to new experiment folder
    thresholds : list of float — risk category thresholds (sorted ascending)
    n_folds : int

    Returns
    -------
    dict with per-fold and aggregated NRI results
    """
    print(f"\n[NRI] Baseline: {os.path.basename(exp1_path)}")
    print(f"[NRI] New:      {os.path.basename(exp2_path)}")
    print(f"[NRI] Thresholds: {[f'{t*100:.0f}%' for t in thresholds]}")

    fold_results = []
    fold_nri_values = []
    pooled_y, pooled_move = [], []

    for fold in range(1, n_folds + 1):
        d1 = _load_fold_predictions(exp1_path, fold)
        d2 = _load_fold_predictions(exp2_path, fold)

        # Verify test indices match (same CV split)
        if d1["test_indices"] != d2["test_indices"]:
            raise ValueError(f"Test indices don't match for fold {fold}")

        nri = calculate_nri(
            np.array(d1["y_true"]),
            np.array(d1["y_proba"]),
            np.array(d2["y_proba"]),
            thresholds,
        )
        fold_results.append(nri)
        fold_nri_values.append(nri["NRI_total"])

        pooled_y.append(np.array(d1["y_true"]))
        pooled_move.append(np.array(nri["movement"]))

        print(f"  Fold {fold}: NRI = {nri['NRI_total']:.4f}")

    # print([np.asarray(a).shape for a in pooled_y], [np.asarray(a).shape for a in pooled_move])

    # Aggregate
    mean_nri = float(np.mean(fold_nri_values))
    std_nri = float(np.std(fold_nri_values))

    total_cases_up = sum(r["cases_up"] for r in fold_results)
    total_cases_down = sum(r["cases_down"] for r in fold_results)
    total_cases = sum(r["cases_total"] for r in fold_results)
    total_controls_up = sum(r["controls_up"] for r in fold_results)
    total_controls_down = sum(r["controls_down"] for r in fold_results)
    total_controls = sum(r["controls_total"] for r in fold_results)

    overall_nri_cases = (total_cases_up - total_cases_down) / total_cases if total_cases > 0 else 0
    overall_nri_controls = (total_controls_down - total_controls_up) / total_controls if total_controls > 0 else 0
    overall_nri = overall_nri_cases + overall_nri_controls

    ci_lo, ci_hi = _nri_pooled_ci(np.concatenate(pooled_y), np.concatenate(pooled_move))
    print(f"  Pooled NRI: {overall_nri:.4f} (95% CI {ci_lo:.4f} to {ci_hi:.4f})")

    print(f"\n  Mean NRI: {mean_nri:.4f} +/- {std_nri:.4f}")
    print(f"  Pooled NRI: {overall_nri:.4f} (cases: {overall_nri_cases:.4f}, controls: {overall_nri_controls:.4f})")

    return {
        "experiment_1": exp1_path,
        "experiment_2": exp2_path,
        "thresholds": thresholds,
        "n_folds": n_folds,
        "fold_nri_values": fold_nri_values,
        "fold_results": [{k: v for k, v in r.items() if k != "movement"} for r in fold_results],
        "mean_nri": mean_nri,
        "std_nri": std_nri,
        "overall_nri": overall_nri,
        "overall_nri_cases": overall_nri_cases,
        "overall_nri_controls": overall_nri_controls,
        "overall_nri_ci": [ci_lo, ci_hi],
        "total_cases": total_cases,
        "total_controls": total_controls,
    }


def run_nri_comparisons(experiment_dirs, nri_pairs=None, thresholds=None):
    """Run NRI comparisons for all configured experiment pairs.

    Parameters
    ----------
    experiment_dirs : dict of {experiment_name: path_to_results_folder}
        Mapping from experiment config name to its output directory.
    nri_pairs : list of (baseline_name, new_name) tuples
        Defaults to config.NRI_PAIRS.
    thresholds : list of float
        Defaults to config.NRI_THRESHOLDS.

    Returns
    -------
    list of dicts — one result dict per comparison
    """
    if nri_pairs is None:
        nri_pairs = config.NRI_PAIRS
    if thresholds is None:
        thresholds = config.NRI_THRESHOLDS

    output_dir = os.path.join(config.get_experiments_dir(), "nri_comparisons")
    os.makedirs(output_dir, exist_ok=True)

    all_results = []

    for baseline_name, new_name in nri_pairs:
        if baseline_name not in experiment_dirs or new_name not in experiment_dirs:
            print(f"[NRI] Skipping {baseline_name} vs {new_name} — missing experiment results")
            continue

        results = calculate_nri_across_folds(
            experiment_dirs[baseline_name],
            experiment_dirs[new_name],
            thresholds=thresholds,
        )

        # Save to JSON
        out_file = os.path.join(output_dir, f"nri_{baseline_name}_vs_{new_name}.json")
        _save_json(results, out_file)
        print(f"[NRI] Saved to {out_file}")

        all_results.append(results)

    return all_results


def _save_json(obj, path):
    """Save dict to JSON, converting numpy types."""
    def convert(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, dict):
            return {k: convert(v) for k, v in o.items()}
        if isinstance(o, list):
            return [convert(v) for v in o]
        return o

    with open(path, "w") as f:
        json.dump(convert(obj), f, indent=4)
