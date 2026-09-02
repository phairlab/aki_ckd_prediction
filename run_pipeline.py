#!/usr/bin/env python3
"""
AKI-CKD prediction pipeline — main entry point.

Typical use
-----------
    # Inventory the raw data first (answers the follow-up-labs question)
    python src/probe_server_data.py --server

    # Everything, tuned, across 4 GPUs
    python run_pipeline.py --server --etl --gpus 0,1,2,3 --tuning full

    # Fast local check that the plumbing works
    python run_pipeline.py --nonsense --tuning smoke --gpus cpu

    # Re-run only the post-hoc analyses on existing fold results
    python run_pipeline.py --server --analyses-only

Stages
------
  etl        raw CSVs -> features.csv, with lab entity normalization
  train      nested cross-validation for each experiment
  analyses   population/missingness/sample size, competing risk,
             ascertainment, threshold sweep, equivalence testing

Evaluation (AUROC, calibration, decision curves, risk distributions, bootstrap
CIs, Nadeau-Bengio comparisons) and net reclassification are produced by the
lancet-digital-health-eval-suite, not here. `--emit-eval-commands` writes the
exact invocations against the directories this run produced.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# OpenMP guard -- must run before ANY third-party import
# ---------------------------------------------------------------------------
# torch bundles its own OpenMP runtime. A process that has imported torch and
# then runs multithreaded XGBoost segfaults immediately and silently (exit 139,
# no Python traceback). KMP_DUPLICATE_LIB_OK does not help; only single-threaded
# OpenMP does.
#
# The pipeline normally keeps them apart by process: XGBoost folds run in
# workers that never import torch, transformer folds import torch and never
# touch XGBoost. --sequential collapses that isolation into one process, so
# OpenMP is pinned to one thread there. It is slower, which is the correct
# trade for a debugging mode.
import os as _os
import sys as _sys

if "--sequential" in _sys.argv:
    for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        _os.environ.setdefault(_var, "1")

import argparse
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from src.plot_style import setup_global_style


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="AKI-CKD prediction pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter)

    source = p.add_mutually_exclusive_group()
    source.add_argument("--smoke", action="store_true",
                        help="Use the coherent synthetic dataset from "
                             "src/make_smoke_data.py. Exercises the whole "
                             "pipeline end to end; results go to a separate tree.")
    source.add_argument("--nonsense", action="store_true",
                        help="Use the legacy column-shuffled data (join key is "
                             "destroyed; prefer --smoke)")
    source.add_argument("--server", action="store_true",
                        help="Use the real data on the secure server")

    p.add_argument("--etl", action="store_true",
                   help="Rebuild features.csv from the raw extracts first")
    p.add_argument("--etl-only", action="store_true",
                   help="Run the ETL and stop. Use this to review "
                        "reports/lab_normalization_audit.csv before committing "
                        "to a long training run.")

    p.add_argument("--experiments", nargs="+", default=None,
                   help="Run only these experiments (names from config.ALL_EXPERIMENTS)")
    p.add_argument("--experiments-set", default="primary",
                   choices=sorted(config.EXPERIMENT_SETS),
                   help="Named group of experiments to run (default: primary)")

    p.add_argument("--tuning", default=config.DEFAULT_TUNING_PROFILE,
                   choices=sorted(config.TUNING_PROFILES) + ["off"],
                   help="Hyperparameter search budget per outer fold "
                        "(default: %(default)s). 'off' reproduces the "
                        "originally submitted untuned configuration.")

    p.add_argument("--gpus", default="auto",
                   help="Devices for fold-level parallelism: 'auto', 'cpu', "
                        "or a comma-separated list such as 0,1,2,3. Repeat an "
                        "index to place two workers on one GPU.")
    p.add_argument("--sequential", action="store_true",
                   help="Run folds in-process. Slower, but tracebacks and pdb work.")

    p.add_argument("--target", choices=["ckd", "ckdordeath"], default=None,
                   help="Override the outcome for all experiments")
    p.add_argument("--skip-shap", action="store_true",
                   help="Skip out-of-fold SHAP (the slowest per-fold step)")
    p.add_argument("--albuminuria-upcr", action="store_true",
                   help="SENSITIVITY ARM: allow urine protein:creatinine as a "
                        "last-resort albuminuria measurement when neither ACR "
                        "nor dipstick exists. Affects 229 patients (4.9%%). "
                        "Results go to separate '<name>_upcr' directories so "
                        "they cannot be confused with the primary analysis.")

    p.add_argument("--analyses-only", action="store_true",
                   help="Skip training; run the post-hoc analyses on existing results")
    p.add_argument("--skip-analyses", action="store_true",
                   help="Train only; do not run the post-hoc analyses")
    p.add_argument("--emit-eval-commands", action="store_true", default=True,
                   help="Write the evaluation-suite invocations for this run")
    p.add_argument("--eval-suite-dir", default="../lancet-digital-health-eval-suite",
                   help="Path to the evaluation suite, for the emitted commands")

    p.add_argument("--bootstrap", type=int, default=config.N_BOOTSTRAP,
                   help="Bootstrap resamples in the threshold sweep (default: %(default)s)")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------

def run_etl_stage():
    from src.etl import run_etl
    print(f"\n{'=' * 70}\nETL\n{'=' * 70}")
    if config.USE_NONSENSE_DATA:
        print("[ETL] The legacy nonsense data ships with a prebuilt features.csv "
              "and has no raw extracts; skipping.")
        return
    run_etl()


def run_training_stage(args, devices):
    """Nested CV for each selected experiment. Returns {name: results_dir}."""
    from src.data_preprocessing import preprocess_data
    from src.cross_validation import run_cross_validation
    from src.analysis.population_table import run_population_analyses

    names = args.experiments or config.EXPERIMENT_SETS[args.experiments_set]
    names = [config.resolve_experiment_name(n) for n in names]

    unknown = [n for n in names if n not in config.ALL_EXPERIMENTS]
    if unknown:
        raise SystemExit(f"Unknown experiment(s): {unknown}\n"
                         f"Available: {sorted(config.ALL_EXPERIMENTS)}")

    print(f"\n{'=' * 70}")
    print(f"TRAINING — {len(names)} experiment(s), tuning profile '{args.tuning}'")
    print(f"{'=' * 70}")
    for n in names:
        print(f"  - {n}")

    experiment_dirs = {}
    population_done = False

    risky = [n for n in names
             if config.ALL_EXPERIMENTS[n].model_type == "transformer"
             and config.ALL_EXPERIMENTS[n].feature_selection_method == "rfe"]
    if risky:
        print(f"\n  WARNING: {risky} use RFE, whose estimator is XGBoost, inside a "
              f"transformer\n  worker that also imports torch. On some installs that "
              f"combination segfaults\n  (see src/parallel.py). Prefer "
              f"feature_selection_method='selectkbest' for transformers.")

    for name in names:
        exp = config.ALL_EXPERIMENTS[name]
        exp = config.apply_tuning_profile(exp, args.tuning)

        from dataclasses import replace
        if args.albuminuria_upcr:
            # Distinct name so the sensitivity arm lands in its own result
            # directory and can never be pooled with the primary analysis.
            exp = replace(exp, name=f"{exp.name}_upcr")
        if args.target:
            exp = replace(exp, target=args.target)
        if args.skip_shap:
            exp = replace(exp, perform_shap=False)

        data = preprocess_data(exp)

        # Population, missingness and sample-size tables depend only on the
        # cohort, so they are produced once from the first experiment's data.
        if not population_done:
            run_population_analyses(data, config.get_reports_dir())
            data["cohort_log"].save(config.get_reports_dir())
            population_done = True

        experiment_dirs[name] = run_cross_validation(
            exp, data, devices=devices, sequential=args.sequential)

    return experiment_dirs


def write_ordering_file(experiment_dirs):
    """Write the experiment-name -> plot-label ordering for this run.

    Written ONCE, before anything consumes it, and every consumer is handed the
    same absolute path: the evaluation script, the NRI call and the NRI
    threshold sweep. Result directories carry a timestamp in their name, so an
    ordering file from a previous run names directories that no longer exist,
    and the evaluation suite refuses the run rather than silently evaluating the
    wrong thing.

    Never hand-edit this file, and never point the suite at its own
    `example_ordering.json` -- that one is a template pinned to whatever run its
    author last made.
    """
    reports = config.get_reports_dir()
    os.makedirs(reports, exist_ok=True)

    ordering = {os.path.basename(path): config.EXPERIMENT_LABELS.get(name, name)
                for name, path in experiment_dirs.items()}
    path = os.path.join(reports, "ordering.json")
    with open(path, "w") as f:
        json.dump(ordering, f, indent=4)

    print(f"\n[Ordering] {len(ordering)} experiment(s) -> {path}")
    for directory, label in ordering.items():
        print(f"    {label:<44s} {directory}")
    return os.path.abspath(path)


def run_analyses_stage(args, experiment_dirs, ordering_path):
    """Every post-hoc analysis that reads the fold predictions."""
    from src.analysis import predictions as pred_mod
    from src.analysis.competing_risk import run_competing_risk_analysis
    from src.analysis.ascertainment import run_ascertainment_analysis
    from src.analysis.ascertainment_bias import run_ascertainment_bias_analysis
    from src.analysis.threshold_sweep import run_threshold_sweep, emit_nri_sweep_commands
    from src.analysis.equivalence import run_equivalence_analysis
    from src.data_preprocessing import preprocess_data

    reports = config.get_reports_dir()
    os.makedirs(reports, exist_ok=True)

    # Keep this repository's recalibration identical to the evaluation suite's.
    pred_mod.verify_against_eval_suite()

    print(f"\n{'=' * 70}\nLOADING POOLED OUT-OF-FOLD PREDICTIONS\n{'=' * 70}")
    experiment_predictions = pred_mod.load_many(experiment_dirs, recalibrate=True)
    if not experiment_predictions:
        print("No experiment predictions found; nothing to analyse.")
        return

    for name, df in experiment_predictions.items():
        print(f"  {name:<28s} {len(df):>6,} patients, "
              f"{int(df['y_true'].sum()):>4} events")

    # Cohort frame for the analyses that need patient characteristics.
    #
    # Deliberately built from a james_score configuration rather than from
    # whichever experiment happens to be first: the expanded feature set
    # one-hot encodes `highest_stage` and `albuminuria_status_raw` away, and the
    # tested-vs-untested comparison needs them as categoricals.
    cohort_config = config.EXPERIMENTS["logreg_james_score"]
    features_df = preprocess_data(cohort_config, verbose=False)["features_df"]

    run_competing_risk_analysis(
        experiment_predictions, features_df,
        output_dir=os.path.join(reports, "competing_risk"),
        threshold=config.PRIMARY_THRESHOLD)

    run_ascertainment_analysis(
        experiment_predictions, features_df,
        followup_labs_path=config.get_followup_labs_path(),
        output_dir=os.path.join(reports, "ascertainment"),
        threshold=config.PRIMARY_THRESHOLD)

    # Runs whether or not a follow-up extract exists. When one does, the
    # descriptive comparison above is the direct answer and this bounds it;
    # when one does not, this is the answer.
    run_ascertainment_bias_analysis(
        experiment_predictions, features_df,
        output_dir=os.path.join(reports, "ascertainment"),
        threshold=config.PRIMARY_THRESHOLD)

    run_threshold_sweep(
        experiment_predictions, config.THRESHOLD_SWEEP,
        output_dir=os.path.join(reports, "threshold_sweep"),
        n_boot=args.bootstrap, ci_level=config.CI_LEVEL,
        labels=config.EXPERIMENT_LABELS)

    baseline = (config.NRI_BASELINE if config.NRI_BASELINE in experiment_dirs
                else next(iter(experiment_dirs)))
    try:
        run_equivalence_analysis(
            experiment_dirs, baseline,
            output_dir=os.path.join(reports, "equivalence"),
            margin=config.EQUIVALENCE_MARGIN_AUROC,
            metric="roc_auc", ci_level=config.CI_LEVEL,
            labels=config.EXPERIMENT_LABELS)
    except (KeyError, FileNotFoundError) as exc:
        print(f"[Equivalence] Skipped: {exc}")

    emit_nri_sweep_commands(
        experiment_dirs, baseline, config.THRESHOLD_SWEEP,
        output_dir=os.path.join(reports, "threshold_sweep"),
        ordering_file=ordering_path,
        eval_suite_dir=args.eval_suite_dir)


# ---------------------------------------------------------------------------
# Handoff to the evaluation suite
# ---------------------------------------------------------------------------

def emit_eval_commands(experiment_dirs, args, ordering_path):
    """Write an ordering file and a runnable script for the evaluation suite."""
    reports = config.get_reports_dir()
    os.makedirs(reports, exist_ok=True)

    results_root = config.get_experiments_dir()
    baseline = (config.NRI_BASELINE if config.NRI_BASELINE in experiment_dirs
                else next(iter(experiment_dirs), None))
    baseline_dir = experiment_dirs.get(baseline, "")

    script = f"""#!/usr/bin/env bash
# Evaluation and reclassification for this run.
# Generated by run_pipeline.py on {datetime.now():%Y-%m-%d %H:%M}.
#
# Discrimination, calibration, decision curves, risk distributions, bootstrap
# CIs and the Nadeau-Bengio comparisons all come from the evaluation suite, and
# NRI comes from nri.py there -- this repository deliberately ships no second
# implementation of either.
set -euo pipefail

EVAL_SUITE="{args.eval_suite_dir}"
RESULTS="{os.path.abspath(results_root)}"
ORDERING="{ordering_path}"
REPORTS="{os.path.abspath(reports)}"

cd "$EVAL_SUITE"

echo "=== Core evaluation (Tables 4/A2.1, Figures 2-4, A2.1-A2.2) ==="
python ldh_eval.py \\
    --input_dir "$RESULTS" \\
    --recurse \\
    --recalibrate \\
    --threshold {config.PRIMARY_THRESHOLD} \\
    --bengio-correction \\
    --ordering "$ORDERING" \\
    --bootstrap {config.N_BOOTSTRAP} \\
    --ci-level {config.CI_LEVEL} \\
    --seed {config.RANDOM_SEED}

echo "=== Net reclassification at the primary threshold (Table 5) ==="
python nri.py \\
    --baseline_dir "{os.path.abspath(baseline_dir)}" \\
    --ordering "$ORDERING" \\
    --threshold {config.PRIMARY_THRESHOLD} \\
    --recalibrate \\
    --bootstrap {config.N_BOOTSTRAP} \\
    --ci-level {config.CI_LEVEL} \\
    --seed {config.RANDOM_SEED} \\
    --output_csv "$REPORTS/nri_table5.csv"

echo
echo "For reclassification across the full threshold range (editor point 7b), run:"
echo "  $REPORTS/threshold_sweep/run_nri_threshold_sweep.sh"
"""
    path = os.path.join(reports, "run_evaluation.sh")
    with open(path, "w") as f:
        f.write(script)
    os.chmod(path, 0o755)

    print(f"[Eval] Run next       -> {path}")
    return path


# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if args.smoke:
        config.USE_SMOKE_DATA, config.USE_NONSENSE_DATA = True, False
    elif args.nonsense:
        config.USE_SMOKE_DATA, config.USE_NONSENSE_DATA = False, True
    elif args.server:
        config.USE_SMOKE_DATA, config.USE_NONSENSE_DATA = False, False

    if args.albuminuria_upcr:
        config.ALBUMINURIA_INCLUDE_UPCR = True

    from src import parallel
    devices = parallel.resolve_devices(args.gpus)

    started = datetime.now()
    print(f"\n{'#' * 70}")
    print(f"# AKI-CKD PREDICTION PIPELINE")
    print(f"# started   : {started:%Y-%m-%d %H:%M:%S}")
    data_label = ("smoke (coherent synthetic)" if config.USE_SMOKE_DATA
                  else "nonsense (column-shuffled)" if config.USE_NONSENSE_DATA
                  else "secure server")
    print(f"# data      : {data_label}")
    print(f"# tuning    : {args.tuning}")
    if config.ALBUMINURIA_INCLUDE_UPCR:
        print(f"# albuminuria: SENSITIVITY ARM — uPCR fallback enabled")
    print(f"# devices   : {parallel.describe_devices(devices)}")
    if args.sequential:
        print(f"# threads   : OpenMP pinned to 1 (--sequential; see src/parallel.py)")
    print(f"# results   : {config.get_experiments_dir()}")
    print(f"# reports   : {config.get_reports_dir()}")
    print(f"{'#' * 70}")

    setup_global_style()
    os.makedirs(config.get_reports_dir(), exist_ok=True)

    if args.etl or args.etl_only:
        run_etl_stage()

    if args.etl_only:
        reports = config.get_reports_dir()
        print(f"\n{'#' * 70}")
        print("# --etl-only: stopping before training.")
        print("#")
        print("# Review these before running the models -- a wrong lab merge")
        print("# propagates into every expanded-feature result:")
        print(f"#   {os.path.join(reports, 'lab_normalization_audit.csv')}")
        print(f"#   {os.path.join(reports, 'feature_inventory.csv')}")
        print("#")
        print("# Then re-run without --etl-only, and add --skip-etl to reuse")
        print("# the features.csv this just built.")
        print(f"{'#' * 70}\n")
        return

    if args.analyses_only:
        from src.analysis.predictions import find_experiment_dirs
        experiment_dirs = find_experiment_dirs(config.get_experiments_dir())
        if not experiment_dirs:
            raise SystemExit(
                f"--analyses-only found no *_fold_results directories in "
                f"{config.get_experiments_dir()}. Train first.")
        print(f"\nFound {len(experiment_dirs)} existing experiment(s): "
              f"{sorted(experiment_dirs)}")
    else:
        experiment_dirs = run_training_stage(args, devices)

    # Written once here, so the analyses, the evaluation script and the NRI
    # sweep all reference the same file for the directories this run produced.
    ordering_path = (write_ordering_file(experiment_dirs) if experiment_dirs
                     else None)

    if not args.skip_analyses:
        run_analyses_stage(args, experiment_dirs, ordering_path)

    if args.emit_eval_commands and experiment_dirs:
        emit_eval_commands(experiment_dirs, args, ordering_path)

    elapsed = datetime.now() - started
    print(f"\n{'#' * 70}")
    print(f"# Pipeline complete in {elapsed}")
    print(f"# Manuscript-ready tables: {config.get_reports_dir()}")
    print(f"# Next: bash {os.path.join(config.get_reports_dir(), 'run_evaluation.sh')}")
    print(f"{'#' * 70}\n")


if __name__ == "__main__":
    main()
