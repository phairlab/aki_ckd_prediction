#!/usr/bin/env bash
# Collect the text outputs needed to interpret a run and write the manuscript.
#
# Gathers summary-level CSV/JSON/TXT only. Deliberately EXCLUDES:
#   *.pt *.pkl *.npy      model checkpoints, scalers, imputers, raw SHAP arrays
#   fold_*_predictions.json  4,694 rows x 7 experiments x 2 runs; the evaluation
#                            suite needs them, interpretation does not
#   *.png                 figures; ask for those separately if wanted
#
# Usage:
#   ./collect_results.sh                # every run label present
#   ./collect_results.sh fast deep      # named labels only
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

RESULTS_ROOT="experiments/results/paper"
STAMP="$(date +%Y%m%d_%H%M)"
OUT="results_handoff_${STAMP}"
TARBALL="${OUT}.tar.gz"

# Portable label discovery: mapfile is bash 4+, find -printf is GNU-only.
LABELS=()
if [[ $# -gt 0 ]]; then
    LABELS=("$@")
else
    for d in "$RESULTS_ROOT"/*/; do
        [[ -d "$d" ]] || continue
        LABELS+=("$(basename "$d")")
    done
fi
if [[ ${#LABELS[@]} -eq 0 ]]; then
    echo "No run labels found under $RESULTS_ROOT" >&2
    exit 1
fi

rm -rf "$OUT"; mkdir -p "$OUT"
echo "Collecting: ${LABELS[*]}"

# --- run-independent: describes features.csv, not any one run ---------------
mkdir -p "$OUT/reports"
for f in lab_normalization_audit.csv feature_inventory.csv server_data_probe.json; do
    [[ -f "reports/$f" ]] && cp "reports/$f" "$OUT/reports/" && echo "  reports/$f"
done

for L in "${LABELS[@]}"; do
    echo "  --- $L ---"

    # Everything in the run's reports subtree is already summary-level.
    if [[ -d "reports/$L" ]]; then
        # find + cp rather than rsync, which is not guaranteed present
        while IFS= read -r f; do
            rel="${f#reports/$L/}"
            mkdir -p "$OUT/reports/$L/$(dirname "$rel")"
            cp "$f" "$OUT/reports/$L/$rel"
        done < <(find "reports/$L" -type f \
                      \( -name '*.csv' -o -name '*.json' -o -name '*.tsv' -o -name '*.sh' \))
        echo "      reports/$L ($(find "$OUT/reports/$L" -type f | wc -l | tr -d ' ') files)"
    fi

    # Per-experiment summaries, plus whatever the evaluation suite wrote.
    for d in "$RESULTS_ROOT/$L"/*_fold_results; do
        [[ -d "$d" ]] || continue
        name="$(basename "$d")"
        dest="$OUT/$RESULTS_ROOT/$L/$name"
        mkdir -p "$dest"
        for f in args.json aggregated_results.json \
                 tuning_summary.json tuning_summary.csv \
                 selected_hyperparameters.csv \
                 shap_out_of_fold_summary.csv \
                 full_feature_importances.txt \
                 aggregate_metrics.json bootstrap_ci.json; do
            [[ -f "$d/$f" ]] && cp "$d/$f" "$dest/"
        done
        # per-fold scalar metrics are small and show fold-to-fold spread
        for f in "$d"/fold_[0-9]*.json; do
            [[ -f "$f" ]] && [[ "$f" != *_predictions.json ]] \
                && [[ "$f" != *_best_params.json ]] && cp "$f" "$dest/"
        done
    done

    # Pre-recalibration pass (Table A2.1, Figure A2.1) lives under reports/,
    # already picked up by the reports subtree copy above.

    # Evaluation-suite overlay outputs (Tables 4 / A2.2, figure legends)
    if [[ -d "$RESULTS_ROOT/$L/overlay_results" ]]; then
        mkdir -p "$OUT/$RESULTS_ROOT/$L/overlay_results"
        find "$RESULTS_ROOT/$L/overlay_results" -maxdepth 1 -type f \
             \( -name '*.csv' -o -name '*.tsv' -o -name '*.json' \) \
             -exec cp {} "$OUT/$RESULTS_ROOT/$L/overlay_results/" \;
        echo "      overlay_results ($(find "$OUT/$RESULTS_ROOT/$L/overlay_results" -type f | wc -l | tr -d ' ') files)"
    else
        echo "      overlay_results MISSING — run reports/$L/run_evaluation.sh first"
    fi
done

# --- manifest so the contents are self-describing ---------------------------
{
    echo "AKI-CKD results handoff"
    echo "collected : $(date "+%Y-%m-%dT%H:%M:%S%z")"
    echo "host      : $(hostname)"
    echo "commit    : $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
    echo "branch    : $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
    echo "labels    : ${LABELS[*]}"
    echo
    echo "files:"
    find "$OUT" -type f | sed "s|^$OUT/||" | sort
} > "$OUT/MANIFEST.txt"

tar -czf "$TARBALL" "$OUT"
rm -rf "$OUT"

echo
echo "Wrote $TARBALL  ($(du -h "$TARBALL" | cut -f1))"
echo "$(tar -tzf "$TARBALL" | wc -l | tr -d ' ') entries"
