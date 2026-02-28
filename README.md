# AKI-CKD Prediction Pipeline

Predicts progression to CKD stage 4-5 after acute kidney injury (AKI) hospitalization.
Trains and evaluates XGBoost, Transformer, and Logistic Regression models across
multiple feature sets (Alberta Score, raw Alberta features, extended features, eGFR),
with SHAP explanations, UMAP projections, and NRI comparisons.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Quick start (nonsense data)

Run a fast test with the included 1000-row synthetic dataset:

```bash
python run_pipeline.py --nonsense --skip-shap --skip-umap
```

Run a single experiment:

```bash
python run_pipeline.py --nonsense --experiments xgb_alberta_raw --skip-shap --skip-umap
```

## Running on real data (secure server)

```bash
# With ETL (raw CSVs -> features.csv first):
python run_pipeline.py --server --etl

# Without ETL (features.csv already exists):
python run_pipeline.py --server
```

## CLI flags

| Flag | Description |
|------|-------------|
| `--nonsense` | Use the 1000-row synthetic test data |
| `--server` | Use real data on the secure server |
| `--etl` | Run ETL pipeline first (server only) |
| `--experiments NAME [NAME ...]` | Run only the listed experiments |
| `--target ckd\|ckdordeath` | Override outcome target |
| `--skip-shap` | Skip SHAP analysis |
| `--skip-umap` | Skip UMAP projections |
| `--nri-only` | Run NRI comparisons on existing results |

## Experiments

Defined in `config.py`:

| Name | Model | Feature Set |
|------|-------|-------------|
| `xgb_alberta_score` | XGBoost | Alberta Score (25-point) |
| `xgb_alberta_raw` | XGBoost | Raw Alberta features |
| `transformer_alberta_raw` | Transformer | Raw Alberta features |
| `xgb_expanded` | XGBoost | Extended (top 100 via RFE) |
| `transformer_expanded` | Transformer | Extended (top 100 via SelectKBest) |
| `transformer_egfr` | Transformer | eGFR features |

## Output structure

Results are saved under `experiments/results/`:

```
experiments/results/
├── YYYYMMDD_HHMMSS_xgb_alberta_raw_fold_results/
│   ├── args.json
│   ├── fold_1_predictions.json
│   ├── fold_1_metrics.json
│   ├── ...
│   ├── aggregated_results.json
│   ├── shap_beeswarm.png
│   ├── shap_bar.png
│   ├── umap_projection.png
│   └── learning_curves.png  (transformer only)
└── nri_comparisons/
    └── nri_xgb_alberta_score_vs_xgb_alberta_raw.json
```

## Configuration

Edit `config.py` to:
- Change `USE_NONSENSE_DATA` default
- Adjust server paths (`SERVER_DATA_DIR`, `SERVER_RAW_DATA_DIR`)
- Modify experiment definitions in `EXPERIMENTS`
- Update NRI comparison pairs in `NRI_PAIRS`
- Tweak plot settings in `PLOT_CONFIG`

## Project structure

```
config.py                   Central configuration
run_pipeline.py             Main entry point
src/
  data_preprocessing.py     Feature loading + cohort filtering
  cross_validation.py       Shared 10-fold CV engine
  etl.py                    Raw data -> features.csv (server)
  plot_style.py             Unified plot formatting
  alberta_score_helpers.py  Alberta score computation
  models/
    xgboost_model.py        XGBoost train/predict
    transformer_model.py    TabularTransformer architectures
    transformer_training.py Training loop + prediction
    logistic_regression.py  Logistic regression wrapper
  analysis/
    shap_analysis.py        SHAP (Tree + Kernel explainers)
    umap_projection.py      2D UMAP by outcome
    net_reclassification.py NRI comparisons
  evaluation/
    metrics.py              Fold-level + aggregated metrics
legacy/                     Deprecated scripts kept for reference
nonsense_data/              1000-row synthetic test dataset
```
