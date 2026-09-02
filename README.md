# AKI-CKD Prediction Pipeline

Predicts progression to CKD stage 4–5 within one year of discharge after an acute kidney
injury hospitalisation, and asks whether expanding from the six-variable **James risk score**
to 478 EHR-derived features — with and without complex models — improves clinical
decision-making for post-AKI nephrology referral.

> **Naming:** the manuscript calls the score the **James score** (James et al., *JAMA* 2017).
> Earlier versions of this codebase called it the "Alberta score". The code now uses *James*
> throughout; `alberta_score_helpers` and the old experiment names still resolve, with a
> deprecation warning.

This repository trains the models and runs the cohort-level analyses. **Evaluation**
(discrimination, calibration, decision curves, risk distributions, bootstrap CIs,
Nadeau-Bengio comparisons) and **net reclassification** are produced by
[`lancet-digital-health-eval-suite`](https://github.com/phairlab/lancet-digital-health-eval-suite),
which is the single home for both — this repository deliberately ships no second
implementation that could drift onto a different probability scale.

---

## Quick start

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

**Check the plumbing works (about two minutes, no GPU):**

```bash
./run_all.sh --smoke
```

**Everything, on the server, across four GPUs:**

```bash
./run_all.sh --server --gpus 0,1,2,3 --tuning full
```

Run the data inventory first — it answers whether the follow-up-labs analysis can run at all:

```bash
python src/probe_server_data.py --server
```

---

## Data modes

| Mode | Directory | Use |
|---|---|---|
| `--smoke` | `smoke_data/` | Coherent synthetic data from `src/make_smoke_data.py`. Patient ids link across files, so the **whole** pipeline runs end to end. Results go to a separate tree and can never be confused with the reported ones. |
| `--server` | `/data/kidney/...` | The real data. |
| `--nonsense` | `nonsense_data/` | Legacy. Every column was shuffled independently, which destroys the join key — preprocessing drops all but a couple of patients. Kept for backwards compatibility only; **prefer `--smoke`**. |

---

## Pipeline stages

```
probe      src/probe_server_data.py    inventory the raw data, quantify lab-name redundancy
  |
etl        src/etl.py                  raw CSVs -> features.csv, with lab entity normalization
  |
train      src/cross_validation.py     nested 10-fold stratified CV, tuned per outer fold
  |
analyses   src/analysis/*              population, competing risk, ascertainment,
  |                                    threshold sweep, equivalence
evaluate   ../lancet-digital-health-eval-suite   AUROC / calibration / DCA / CIs / NRI
```

### Nested cross-validation

The outer loop is the reported 10-fold **stratified** split. Inside each outer *training*
fold, an independent Optuna TPE search runs on its own inner `StratifiedKFold`, and the
configuration it selects is refitted on the whole outer training fold. The outer test fold is
never seen by the search.

`--tuning off` reproduces the originally submitted untuned configuration exactly, so the
effect of tuning can be quantified rather than assumed.

| Profile | XGBoost trials | Transformer trials | Rough cost |
|---|---|---|---|
| `smoke` | 4 | 2 | ~2 min, CPU |
| `fast` | 30 | 15 | a few hours, 4 GPUs |
| `full` | 100 | 40 | overnight, 4 GPUs |
| `deep` | 200 | 80 | longer |

### Multi-GPU

Outer folds are the unit of parallelism — they are fully independent, so distributing them
changes no statistics. Workers pull folds from a queue as they free up.

```bash
python run_pipeline.py --server --gpus 0,1,2,3     # four workers
python run_pipeline.py --server --gpus 0,0,1,1     # two workers per GPU
python run_pipeline.py --server --gpus cpu
python run_pipeline.py --server --sequential        # in-process, for debugging
```

With 10 folds on 4 GPUs the schedule is 4 + 4 + 2, so expect roughly 2.5× rather than 4×.
CPU threads are divided among workers automatically to avoid oversubscribing XGBoost.

> **Why torch and XGBoost are kept in separate processes.** torch bundles its own OpenMP
> runtime. A process that has imported torch and then runs multithreaded XGBoost **segfaults
> immediately and silently** (exit 139, no Python traceback); `KMP_DUPLICATE_LIB_OK` does not
> help. The pipeline keeps them apart by process: XGBoost folds run in workers that never
> import torch, and transformer folds import torch but (with the default SelectKBest selector)
> never touch XGBoost. GPU discovery happens in a throwaway subprocess so the parent stays
> torch-free. `--sequential` collapses that isolation, so it pins OpenMP to one thread.
> See the note above `probe_cuda_devices()` in `src/parallel.py`.

---

## Experiments

Defined in `config.py`.

| Name | Model | Features |
|---|---|---|
| `logreg_james_score` | Logistic regression | James score (1) — **primary baseline** |
| `xgb_james_score` | XGBoost | James score (1) — the submitted baseline, retained for comparison |
| `logreg_james_raw` | Logistic regression | James components (11 after encoding) |
| `xgb_james_raw` | XGBoost | James components |
| `transformer_james_raw` | Transformer | James components |
| `xgb_expanded` | XGBoost | 478 → 100 via SelectKBest |
| `transformer_expanded` | Transformer | 478 → 100 via SelectKBest |

Sensitivity sets: `--experiments-set sensitivity_k` (k ∈ 10/25/50/200) and
`sensitivity_selector` (RFE instead of SelectKBest).

Both architectures now use the **same model-agnostic selector**, so a difference between them
is attributable to the model rather than to RFE versus SelectKBest.

### Transformer architectures

| Name | Description |
|---|---|
| `row_token` | The originally submitted model. Embeds the whole row as **one token**, so self-attention runs over a length-1 sequence — softmax over a single key returns weight 1, and the block reduces to a position-wise feedforward network. **It is an MLP.** Retained for comparability, under an honest name. |
| `feature_token` | FT-Transformer style (Gorishniy et al., NeurIPS 2021). Each feature is its own token with a learned `[CLS]` token, so attention genuinely models interactions between clinical variables. |

Both are in the tuning search space, so a tuned result is an upper bound over architectures
rather than over one arbitrary choice.

---

## Outputs

Per experiment, under `experiments/results/paper/<timestamp>_<name>_fold_results/`:

| File | Contents |
|---|---|
| `fold_N_predictions.json` | held-out predictions, with `patient_ids` |
| `train_N_predictions.json` | out-of-fold **training** predictions, for the recalibration map |
| `fold_N.json` | per-fold metrics |
| `aggregated_results.json` | mean ± SD across folds |
| `tuning_fold_N_<model>.json` | every trial of that fold's search |
| `tuning_summary.csv` | selected configuration per fold, and its stability |
| `selected_hyperparameters.csv` | one row per fold |
| `shap_out_of_fold_summary.csv` | mean \|SHAP\| with SD across folds |

Manuscript-ready tables land in `reports/`:

| File | Manuscript element |
|---|---|
| `cohort_flow.csv` | Figure 1 |
| `table3_population.csv` | Table 3 |
| `missingness_all.csv` | TRIPOD+AI item 11 |
| `sample_size_adequacy.csv` | events per variable, Riley minimum sample size |
| `lab_normalization_audit.csv` | **review before trusting a refit** |
| `feature_inventory.csv` | Multimedia Appendix 3 |
| `competing_risk/` | competing risk of death |
| `ascertainment/` | outcome ascertainment and loss to follow-up |
| `threshold_sweep/` | threshold metrics across the plausible range |
| `equivalence/` | formal equivalence testing |
| `ordering.json`, `run_evaluation.sh` | handoff to the evaluation suite |

---

## Configuration

`config.py` holds everything that varies between runs:

- data paths, including `FOLLOWUP_LABS_PATH` for the ascertainment analysis
- `NORMALIZE_LAB_NAMES` — group labs by clinical entity rather than raw `TEST_NM`
- `PRIMARY_THRESHOLD` (0.20) and `THRESHOLD_SWEEP`
- `EQUIVALENCE_MARGIN_AUROC` — **pre-specified before any analysis runs**
- `TUNING_PROFILES`, experiment definitions, comparison pairs, plot settings

---

## Project structure

```
config.py                       central configuration
run_pipeline.py                 main entry point
run_all.sh                      one command for the whole resubmission
src/
  probe_server_data.py          raw data inventory (run this first)
  make_smoke_data.py            coherent synthetic data for end-to-end testing
  lab_normalization.py          clinical entity normalization + self-test
  etl.py                        raw CSVs -> features.csv
  data_preprocessing.py         cohort filters, James score, fold-local imputation plan
  cross_validation.py           nested CV engine
  tuning.py                     Optuna TPE search inside each outer fold
  parallel.py                   multi-GPU fold dispatch
  james_score_helpers.py        James score components and unit handling
  plot_style.py                 shared plot formatting
  models/
    transformer_model.py        RowToken / FeatureToken / Large architectures
    transformer_training.py     training loop, out-of-fold probabilities
    xgboost_model.py            thin XGBoost wrapper
    logistic_regression.py      thin logistic regression wrapper
  analysis/
    predictions.py              pooled out-of-fold loading + recalibration
    competing_risk.py           competing risk of death
    ascertainment.py            outcome ascertainment / loss to follow-up
    threshold_sweep.py          metrics across decision thresholds
    equivalence.py              TOST with a pre-specified margin
    population_table.py         Table 3, missingness, sample-size adequacy
    shap_analysis.py            SHAP plotting helpers
    umap_projection.py          2D UMAP (not recomputed for the resubmission)
  evaluation/metrics.py         fold-level metric computation
legacy/                         superseded scripts, kept for provenance
smoke_data/                     generated synthetic data (gitignored)
nonsense_data/                  legacy column-shuffled data
reports/                        manuscript-ready tables and figures
```

---

## Reproducibility notes

- `RANDOM_SEED = 1202` seeds the outer split, the inner splits, every model and every search.
- `mutual_info_classif` is seeded, so feature selection is reproducible.
- Imputation is fitted on the training fold only; the medians never see held-out data.
- XGBoost receives `NaN` and uses its native missing-value handling. Transformer and logistic
  regression inputs are imputed per fold (zeros for counts and stage indicators, training-fold
  medians for continuous measurements).
- `src/lab_normalization.py` carries a 27-case self-test asserting the ordering-sensitive
  mappings; run it with `python src/lab_normalization.py`.
