"""
Threshold metrics across a range of decision thresholds (editor's point 7b).

> "it is the threshold metrics in Table 4 and the reclassification analysis in
>  Table 5 that are confined to 20% and should be reported across a range of
>  plausible thresholds."

This module covers the Table 4 half: sensitivity, specificity, PPV, NPV, alert
rate and net benefit at every threshold in `config.THRESHOLD_SWEEP`, with
patient-level bootstrap confidence intervals, for every experiment.

The Table 5 half (reclassification) is produced by `nri.py` in the
lancet-digital-health-eval-suite, which is the single home for NRI in this
project. `emit_nri_sweep_commands()` writes the exact invocations -- one per
threshold -- so the two sweeps cover the same grid on the same probability
scale.

Why this matters beyond compliance
----------------------------------
20% is one nephrologist survey's answer. Nephrology capacity differs by health
system, and the manuscript's own argument is that the models separate only away
from the decision boundary. A sweep either shows the models stay equivalent
across the plausible range -- which is a considerably stronger claim than
equivalence at one point -- or it shows a threshold where they do not, which is
a finding rather than an omission.
"""

from __future__ import annotations

import os
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.analysis.predictions import threshold_metrics


METRICS_TO_PLOT = ("sensitivity", "ppv", "alert_rate", "net_benefit")


def sweep_one_experiment(predictions, thresholds, n_boot=2000, ci_level=0.95,
                         seed=0):
    """Threshold metrics with bootstrap CIs for one experiment.

    Each patient contributes exactly one row (each is in exactly one test
    fold), so a patient-level bootstrap is a plain row resample. All thresholds
    are recomputed on the SAME resample, which preserves the correlation
    between adjacent thresholds -- resampling independently per threshold would
    make the sweep look jagged for no real reason.
    """
    y_true = predictions["y_true"].to_numpy().astype(int)
    y_proba = predictions["y_proba"].to_numpy(dtype=float)
    n = len(y_true)

    point = {t: threshold_metrics(y_true, y_proba, t) for t in thresholds}

    draws = {t: {m: [] for m in point[thresholds[0]] if m not in
                 ("threshold", "n", "n_events", "tp", "fp", "tn", "fn")}
             for t in thresholds}

    if n_boot:
        rng = np.random.default_rng(seed)
        for _ in range(n_boot):
            idx = rng.integers(0, n, n)
            yb, pb = y_true[idx], y_proba[idx]
            if yb.sum() == 0 or yb.sum() == len(yb):
                continue
            for t in thresholds:
                metrics = threshold_metrics(yb, pb, t)
                for m in draws[t]:
                    draws[t][m].append(metrics[m])

    lo_pct = (1 - ci_level) / 2 * 100
    hi_pct = 100 - lo_pct

    rows = []
    for t in thresholds:
        row = dict(point[t])
        for m, values in draws[t].items():
            arr = np.asarray(values, dtype=float)
            arr = arr[np.isfinite(arr)]
            if arr.size:
                row[f"{m}_ci_lo"] = float(np.percentile(arr, lo_pct))
                row[f"{m}_ci_hi"] = float(np.percentile(arr, hi_pct))
        rows.append(row)

    return pd.DataFrame(rows)


def run_threshold_sweep(experiment_predictions, thresholds, output_dir,
                        n_boot=2000, ci_level=0.95, seed=0,
                        labels=None):
    """Sweep every experiment and write the combined table plus figures."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n{'=' * 70}\nTHRESHOLD SWEEP  (editor point 7b)\n{'=' * 70}")
    print(f"Thresholds: {thresholds}")
    print(f"Bootstrap : {n_boot} resamples at {int(ci_level * 100)}%")

    labels = labels or {}
    frames = []
    for name, preds in experiment_predictions.items():
        print(f"  sweeping {name} ({len(preds):,} patients)...")
        df = sweep_one_experiment(preds, thresholds, n_boot, ci_level, seed)
        df.insert(0, "experiment", name)
        df.insert(1, "label", labels.get(name, name))
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    csv_path = os.path.join(output_dir, "threshold_sweep.csv")
    combined.to_csv(csv_path, index=False)

    formatted = _format_for_manuscript(combined, ci_level)
    formatted.to_csv(os.path.join(output_dir, "threshold_sweep_formatted.tsv"),
                     sep="\t", index=False)

    _plot_sweep(combined, output_dir, ci_level)

    print(f"[ThresholdSweep] -> {csv_path}")
    print(f"[ThresholdSweep] paste-ready -> "
          f"{os.path.join(output_dir, 'threshold_sweep_formatted.tsv')}")
    return combined


def _format_for_manuscript(combined, ci_level):
    """'point (95% CI lo to hi)' strings, one row per experiment x threshold."""
    pct = int(round(ci_level * 100))
    rows = []
    for _, r in combined.iterrows():
        row = {"Experiment": r["label"], "Threshold": f"{r['threshold']:.2f}"}
        for metric in ("sensitivity", "specificity", "ppv", "npv",
                       "alert_rate", "net_benefit"):
            value = r.get(metric)
            lo, hi = r.get(f"{metric}_ci_lo"), r.get(f"{metric}_ci_hi")
            if value is None or not np.isfinite(value):
                row[metric] = "—"
            elif lo is not None and np.isfinite(lo):
                row[metric] = f"{value:.3f} ({pct}% CI {lo:.3f} to {hi:.3f})"
            else:
                row[metric] = f"{value:.3f}"
        row["TP"] = int(r["tp"]); row["FP"] = int(r["fp"])
        row["FN"] = int(r["fn"]); row["TN"] = int(r["tn"])
        rows.append(row)
    return pd.DataFrame(rows)


def _plot_sweep(combined, output_dir, ci_level):
    """One panel per metric, one line per experiment, shaded CI."""
    experiments = list(dict.fromkeys(combined["label"]))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(experiments), 2)))

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, metric in zip(axes.ravel(), METRICS_TO_PLOT):
        for i, label in enumerate(experiments):
            sub = combined[combined["label"] == label].sort_values("threshold")
            ax.plot(sub["threshold"], sub[metric], marker="o", ms=4,
                    lw=2, color=colors[i], label=label)
            lo_col, hi_col = f"{metric}_ci_lo", f"{metric}_ci_hi"
            if lo_col in sub and sub[lo_col].notna().any():
                ax.fill_between(sub["threshold"], sub[lo_col], sub[hi_col],
                                color=colors[i], alpha=0.12, linewidth=0)

        ax.axvline(0.20, color="red", lw=1.2, alpha=0.55, ls="-")
        ax.set_xlabel("Decision threshold")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(metric.replace("_", " ").title(), fontweight="bold")
        ax.grid(alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles, labels_ = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles + [plt.Line2D([], [], color="red", lw=1.2)],
               labels_ + ["Referral threshold (0.20)"],
               loc="lower center", ncol=2, frameon=False, fontsize=9)
    fig.suptitle(f"Threshold-dependent performance across the plausible range "
                 f"(shaded: {int(ci_level * 100)}% CI)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0.10, 1, 0.97])
    fig.savefig(os.path.join(output_dir, "threshold_sweep.png"), dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Handoff to the evaluation suite for the reclassification half
# ---------------------------------------------------------------------------

def emit_nri_sweep_commands(experiment_dirs, baseline_name, thresholds,
                            output_dir, eval_suite_dir="../lancet-digital-health-eval-suite",
                            ordering_file="example_ordering.json"):
    """Write the nri.py invocations covering the same threshold grid.

    NRI lives in the evaluation suite, not here, so this repository does not
    ship a second implementation that could drift onto a different probability
    scale. This writes a runnable script rather than duplicating the maths.
    """
    os.makedirs(output_dir, exist_ok=True)
    baseline_dir = experiment_dirs.get(baseline_name)
    if baseline_dir is None:
        print(f"[ThresholdSweep] Baseline {baseline_name!r} not found; "
              f"skipping NRI command generation.")
        return None

    lines = [
        "#!/usr/bin/env bash",
        "# Reclassification across the same threshold grid as threshold_sweep.csv",
        "# (editor point 7b, Table 5 half). Generated by threshold_sweep.py.",
        "#",
        "# NRI is computed by the evaluation suite so that it uses the identical",
        "# recalibration map as the threshold metrics; this repository deliberately",
        "# does not carry its own NRI implementation.",
        "set -euo pipefail",
        "",
        f'EVAL_SUITE="{eval_suite_dir}"',
        f'BASELINE="{os.path.abspath(baseline_dir)}"',
        f'OUT="{os.path.abspath(output_dir)}"',
        'mkdir -p "$OUT/nri_by_threshold"',
        "",
        'cd "$EVAL_SUITE"',
        "",
    ]
    for t in thresholds:
        tag = f"{t:.2f}".replace(".", "p")
        lines += [
            f"echo '--- NRI at threshold {t:.2f} ---'",
            "python nri.py \\",
            '    --baseline_dir "$BASELINE" \\',
            f'    --ordering "{ordering_file}" \\',
            f"    --threshold {t} \\",
            "    --recalibrate \\",
            "    --bootstrap 2000 \\",
            f'    --output_csv "$OUT/nri_by_threshold/nri_{tag}.csv"',
            "",
        ]
    lines += [
        "echo 'Combining...'",
        'python - <<PY',
        "import glob, pandas as pd, os",
        f'out = os.path.join("{os.path.abspath(output_dir)}", "nri_by_threshold")',
        'frames = [pd.read_csv(p) for p in sorted(glob.glob(os.path.join(out, "nri_*.csv")))]',
        'if frames:',
        '    combined = pd.concat(frames, ignore_index=True)',
        '    combined.to_csv(os.path.join(out, "nri_all_thresholds.csv"), index=False)',
        '    print("wrote", os.path.join(out, "nri_all_thresholds.csv"))',
        "PY",
        "",
    ]

    path = os.path.join(output_dir, "run_nri_threshold_sweep.sh")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    os.chmod(path, 0o755)
    print(f"[ThresholdSweep] NRI sweep script -> {path}")
    return path
