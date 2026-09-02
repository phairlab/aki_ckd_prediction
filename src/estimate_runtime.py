#!/usr/bin/env python3
"""
Project the runtime of another tuning profile from a completed run.

Nothing in the pipeline predicts how long a run will take before it starts, and
it cannot sensibly be guessed: the cost depends on the GPU, the CPU core count,
how many workers share a card, and how aggressively Optuna's pruner fires. What
the pipeline DOES record is every ingredient needed to project one profile from
another, so this reads a finished run and does that arithmetic.

Usage
-----
    python src/estimate_runtime.py --server                    # newest run
    python src/estimate_runtime.py --server --workers 10
    python src/estimate_runtime.py --server --results-dir <path>

Method
------
For each experiment, per outer fold, cost splits in two:

    tuning      trials x inner folds x one model fit. Scales with the trial
                count, which is the only thing a profile changes.
    everything  the final refit, the recalibration inner CV, out-of-fold SHAP,
    else        feature selection, artifact writing. IDENTICAL across profiles.

Projection for a target profile is therefore

    fold_cost(target) = tuning_cost(observed) x trials(target)/trials(observed)
                        + other_cost(observed)

then wall clock is ceil(10 / workers) waves of that, summed over experiments,
since experiments run one after another.

The trial-count scaling is deliberately LINEAR, which is conservative. Optuna's
MedianPruner abandons a trial after its first inner fold once its running score
falls below the median of completed trials, and the median tends to improve as
the search proceeds -- so a longer search prunes a larger fraction and the real
cost grows sublinearly. Treat the projection as an upper bound; the observed
prune rate is reported so the size of that slack is visible.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import config


def _fmt(hours):
    if hours != hours:                       # NaN
        return "?"
    if hours < 1 / 60:
        return f"{hours * 3600:.0f}s"
    if hours < 1:
        return f"{hours * 60:.0f}min"
    return f"{hours:.1f}h"


def read_experiment(path):
    """Observed per-fold tuning and non-tuning cost for one experiment."""
    name = "_".join(os.path.basename(path).split("_")[2:-2])

    with open(os.path.join(path, "args.json")) as f:
        args = json.load(f)

    tuning_files = sorted(glob.glob(os.path.join(path, "tuning_fold_*_*.json")))
    tuning_sec, completed, pruned = [], 0, 0
    for tf in tuning_files:
        with open(tf) as f:
            rec = json.load(f)
        tuning_sec.append(rec.get("elapsed_sec", 0.0))
        completed += rec.get("n_trials_completed", 0)
        pruned += rec.get("n_trials_pruned", 0)

    fold_min = []
    params_csv = os.path.join(path, "selected_hyperparameters.csv")
    if os.path.exists(params_csv):
        df = pd.read_csv(params_csv)
        if "elapsed_min" in df.columns:
            fold_min = df["elapsed_min"].dropna().tolist()

    n_folds = max(len(fold_min), len(tuning_sec), 1)
    mean_tuning_sec = (sum(tuning_sec) / len(tuning_sec)) if tuning_sec else 0.0
    mean_fold_sec = (sum(fold_min) / len(fold_min) * 60) if fold_min else mean_tuning_sec
    other_sec = max(mean_fold_sec - mean_tuning_sec, 0.0)

    return {
        "experiment": name,
        "model_type": args.get("model_type"),
        "tuned": bool(args.get("tune")),
        "observed_trials": args.get("n_trials") or 0,
        "inner_folds": args.get("inner_folds"),
        "n_folds": n_folds,
        "mean_tuning_sec": mean_tuning_sec,
        "mean_other_sec": other_sec,
        "mean_fold_sec": mean_fold_sec,
        "trials_completed": completed,
        "trials_pruned": pruned,
        "prune_rate": (pruned / (completed + pruned)) if (completed + pruned) else 0.0,
    }


def project(rows, target_profile, workers, n_folds=None):
    """Project total wall clock for `target_profile`."""
    profile = config.TUNING_PROFILES[target_profile]
    out = []
    for r in rows:
        n_folds_r = n_folds or r["n_folds"]
        key = "tf_trials" if r["model_type"] == "transformer" else "xgb_trials"
        target_trials = profile[key]

        if r["tuned"] and r["observed_trials"]:
            scale = target_trials / r["observed_trials"]
        else:
            scale = 1.0
        tuning = r["mean_tuning_sec"] * scale
        fold_sec = tuning + r["mean_other_sec"]

        waves = math.ceil(n_folds_r / max(workers, 1))
        wall_sec = waves * fold_sec
        out.append({**r,
                    "target_trials": target_trials,
                    "scale": scale,
                    "projected_fold_sec": fold_sec,
                    "projected_wall_hours": wall_sec / 3600,
                    "projected_compute_hours": fold_sec * n_folds_r / 3600})
    return out


def main():
    p = argparse.ArgumentParser(description="Project runtime for another tuning profile")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--server", action="store_true")
    src.add_argument("--smoke", action="store_true")
    p.add_argument("--results-dir", default=None,
                   help="Results root (default: the configured experiments dir)")
    p.add_argument("--workers", type=int, default=10,
                   help="Workers the projection assumes (default 10)")
    args = p.parse_args()

    if args.smoke:
        config.USE_SMOKE_DATA, config.USE_NONSENSE_DATA = True, False
    elif args.server:
        config.USE_SMOKE_DATA, config.USE_NONSENSE_DATA = False, False

    root = args.results_dir or config.get_experiments_dir()
    dirs = sorted(d for d in glob.glob(os.path.join(root, "*_fold_results"))
                  if os.path.isdir(d))
    if not dirs:
        raise SystemExit(f"No *_fold_results directories under {root}")

    # Newest run of each experiment
    latest = {}
    for d in dirs:
        name = "_".join(os.path.basename(d).split("_")[2:-2])
        latest[name] = d

    rows = []
    for d in latest.values():
        try:
            rows.append(read_experiment(d))
        except Exception as exc:                               # noqa: BLE001
            print(f"  skipping {os.path.basename(d)}: {type(exc).__name__}: {exc}")

    print(f"\n{'=' * 78}\nOBSERVED  ({len(rows)} experiment(s) in {root})\n{'=' * 78}")
    print(f"  {'experiment':<26s} {'trials':>7s} {'prune':>6s} "
          f"{'tune/fold':>10s} {'other/fold':>11s} {'fold':>8s}")
    for r in sorted(rows, key=lambda x: -x["mean_fold_sec"]):
        print(f"  {r['experiment']:<26s} {r['observed_trials']:>7d} "
              f"{r['prune_rate'] * 100:>5.0f}% "
              f"{_fmt(r['mean_tuning_sec'] / 3600):>10s} "
              f"{_fmt(r['mean_other_sec'] / 3600):>11s} "
              f"{_fmt(r['mean_fold_sec'] / 3600):>8s}")

    observed_wall = sum(math.ceil(r["n_folds"] / max(args.workers, 1))
                        * r["mean_fold_sec"] for r in rows) / 3600
    print(f"\n  wall clock at {args.workers} workers, as observed: "
          f"{_fmt(observed_wall)}")

    for target in ("fast", "full", "deep"):
        proj = project(rows, target, args.workers)
        total_wall = sum(r["projected_wall_hours"] for r in proj)
        total_compute = sum(r["projected_compute_hours"] for r in proj)
        print(f"\n{'-' * 78}\nPROJECTED: --tuning {target}   "
              f"(upper bound; pruning makes the real cost lower)\n{'-' * 78}")
        print(f"  {'experiment':<26s} {'trials':>7s} {'scale':>7s} "
              f"{'fold':>9s} {'wall':>9s}")
        for r in sorted(proj, key=lambda x: -x["projected_wall_hours"]):
            print(f"  {r['experiment']:<26s} {r['target_trials']:>7d} "
                  f"{r['scale']:>6.1f}x {_fmt(r['projected_fold_sec'] / 3600):>9s} "
                  f"{_fmt(r['projected_wall_hours']):>9s}")
        print(f"  {'':<26s} {'':>7s} {'':>7s} {'TOTAL':>9s} "
              f"{_fmt(total_wall):>9s}")
        print(f"  ({_fmt(total_compute)} of compute across all folds, "
              f"spread over {args.workers} workers)")

    print(f"\n{'=' * 78}")
    print("Experiments run one after another, so wall clock is their sum. Folds")
    print(f"within an experiment run in parallel: ceil(10 / {args.workers}) wave(s).")
    print("Scaling with trial count is linear here, which is conservative --")
    print("MedianPruner abandons more trials as the search median improves, so a")
    print("longer search prunes a larger share. The prune column above shows how")
    print("much slack that already represents.")
    print(f"{'=' * 78}\n")


if __name__ == "__main__":
    main()
