#!/usr/bin/env bash
#
# One command to produce every number the resubmission needs.
#
#   ./run_all.sh --server --gpus 0,1,2,3 --tuning full
#   ./run_all.sh --smoke                              # 2-minute end-to-end check
#
# Stages, in order:
#   0  probe        inventory the raw data (answers the follow-up-labs question)
#   1  pipeline     ETL -> nested cross-validation -> post-hoc analyses
#   2  evaluation   discrimination / calibration / decision curves / bootstrap CIs
#   3  reclassification  NRI at the primary threshold, and across the sweep
#
# Stages 2 and 3 run in the lancet-digital-health-eval-suite. Everything that
# lands in reports/ is manuscript-ready.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"
EVAL_SUITE="${EVAL_SUITE:-$HERE/../lancet-digital-health-eval-suite}"

DATA_FLAG="--server"
GPUS="auto"
TUNING="full"
EXPERIMENTS_SET="primary"
SKIP_ETL=""
ETL_ONLY=""
EXTRA=()

usage() {
    cat <<'USAGE'
Usage: ./run_all.sh [options]

  --server | --smoke | --nonsense   Data source (default: --server)
  --gpus LIST                       e.g. 0,1,2,3 | cpu | auto  (default: auto)
  --tuning PROFILE                  smoke | fast | full | deep | off  (default: full)
  --experiments-set NAME            primary | sensitivity_k | sensitivity_selector | all
  --skip-etl                        Reuse the existing features.csv
  --etl-only                        Rebuild features.csv and stop, so the lab
                                    normalization audit can be reviewed first
  --eval-suite PATH                 Path to lancet-digital-health-eval-suite
  --                                Pass everything after this to run_pipeline.py

Environment:
  PYTHON      interpreter to use (default: python)
  EVAL_SUITE  overrides --eval-suite
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --server|--smoke|--nonsense) DATA_FLAG="$1"; shift ;;
        --gpus)             GPUS="$2"; shift 2 ;;
        --tuning)           TUNING="$2"; shift 2 ;;
        --experiments-set)  EXPERIMENTS_SET="$2"; shift 2 ;;
        --skip-etl)         SKIP_ETL="1"; shift ;;
        --etl-only)         ETL_ONLY="1"; shift ;;
        --eval-suite)       EVAL_SUITE="$2"; shift 2 ;;
        -h|--help)          usage; exit 0 ;;
        --)                 shift; EXTRA=("$@"); break ;;
        *)                  echo "Unknown option: $1" >&2; usage; exit 1 ;;
    esac
done

ETL_FLAG="--etl"
[[ -n "$SKIP_ETL" ]] && ETL_FLAG=""

# Smoke mode needs its synthetic dataset generated first
if [[ "$DATA_FLAG" == "--smoke" ]]; then
    TUNING="${TUNING/full/smoke}"
    echo "=== Generating the synthetic smoke dataset ==="
    "$PYTHON" "$HERE/src/make_smoke_data.py"
fi

banner() { printf '\n\n%s\n%s\n%s\n' "$(printf '=%.0s' {1..74})" "$1" "$(printf '=%.0s' {1..74})"; }

# ---------------------------------------------------------------------------
banner "STAGE 0 — raw data inventory"
# Never fatal: on smoke data some sections have nothing to report, and the
# probe is diagnostic rather than a dependency of anything downstream.
"$PYTHON" "$HERE/src/probe_server_data.py" "$DATA_FLAG" || \
    echo "(probe reported problems; continuing — review reports/server_data_probe.json)"

# ---------------------------------------------------------------------------
banner "STAGE 1 — ETL, nested cross-validation, post-hoc analyses"
if [[ -n "$ETL_ONLY" ]]; then
    "$PYTHON" "$HERE/run_pipeline.py" "$DATA_FLAG" --etl-only
    exit 0
fi
"$PYTHON" "$HERE/run_pipeline.py" \
    "$DATA_FLAG" $ETL_FLAG \
    --gpus "$GPUS" \
    --tuning "$TUNING" \
    --experiments-set "$EXPERIMENTS_SET" \
    --eval-suite-dir "$EVAL_SUITE" \
    ${EXTRA[@]+"${EXTRA[@]}"}

# Locate the reports directory the pipeline actually used
REPORTS="$("$PYTHON" - <<PY
import sys; sys.path.insert(0, "$HERE")
import config
config.USE_SMOKE_DATA = "$DATA_FLAG" == "--smoke"
config.USE_NONSENSE_DATA = "$DATA_FLAG" == "--nonsense"
print(config.get_reports_dir())
PY
)"

# ---------------------------------------------------------------------------
banner "STAGES 2 & 3 — evaluation and reclassification"
if [[ ! -d "$EVAL_SUITE" ]]; then
    cat <<EOF
The evaluation suite was not found at:
    $EVAL_SUITE

Clone it beside this repository, or pass --eval-suite PATH. The commands for
this run have still been written to:
    $REPORTS/run_evaluation.sh
EOF
    exit 0
fi

bash "$REPORTS/run_evaluation.sh"

if [[ -x "$REPORTS/threshold_sweep/run_nri_threshold_sweep.sh" ]]; then
    banner "STAGE 3b — reclassification across the threshold sweep"
    bash "$REPORTS/threshold_sweep/run_nri_threshold_sweep.sh"
fi

# ---------------------------------------------------------------------------
banner "DONE"
cat <<EOF
Everything the resubmission needs is under:
    $REPORTS

  cohort_flow.csv                     Figure 1 attrition
  table3_population.csv               Table 3
  missingness_all.csv                 TRIPOD+AI item 11
  sample_size_adequacy.csv            events per variable and the Riley requirement
  lab_normalization_audit.csv         editor point 5 — REVIEW THIS BEFORE TRUSTING A REFIT
  feature_inventory.csv               Multimedia Appendix 3
  competing_risk/                     editor point 2
  ascertainment/                      editor point 3
  threshold_sweep/                    editor point 7b (Table 4 across thresholds)
  equivalence/                        editor point 7c
  nri_table5.csv                      Table 5
  <results>/overlay_results/           Tables 4 and A2.1-A2.2, Figures 2-4

Per-experiment tuning evidence (editor point 1) is in each fold_results folder:
  tuning_summary.csv, selected_hyperparameters.csv, tuning_fold_*_*.json
  shap_out_of_fold_summary.csv        editor point 6
EOF
