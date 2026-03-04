#!/usr/bin/env python3
"""
Main entry point for the AKI-CKD prediction pipeline.

Usage examples:
  python run_pipeline.py                                    # All experiments + NRI
  python run_pipeline.py --experiments xgb_alberta_raw      # Single experiment
  python run_pipeline.py --nonsense --skip-shap --skip-umap # Fast test run
  python run_pipeline.py --server --etl                     # Server: ETL + all experiments
  python run_pipeline.py --target ckdordeath                # Alternate target
  python run_pipeline.py --nri-only                         # NRI on existing results
"""

import argparse
import os
import sys

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from src.plot_style import setup_global_style
from src.data_preprocessing import preprocess_data
from src.cross_validation import run_cross_validation
from src.analysis.net_reclassification import run_nri_comparisons
from src.analysis.shap_analysis import perform_shap_analysis_tree, perform_shap_analysis_kernel
from src.analysis.umap_projection import generate_umap_projection


def parse_args():
    parser = argparse.ArgumentParser(description="AKI-CKD Prediction Pipeline")

    # Data source
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--nonsense", action="store_true",
                        help="Force use of nonsense test data")
    source.add_argument("--server", action="store_true",
                        help="Force use of real server data")

    # ETL
    parser.add_argument("--etl", action="store_true",
                        help="Run ETL (raw CSVs -> features.csv) before experiments")

    # Experiment selection
    parser.add_argument("--experiments", nargs="+",
                        choices=list(config.EXPERIMENTS.keys()),
                        help="Run only these experiments (default: all)")

    # Target
    parser.add_argument("--target", choices=["ckd", "ckdordeath"], default=None,
                        help="Override target for all experiments")

    # Skip flags
    parser.add_argument("--skip-shap", action="store_true",
                        help="Skip SHAP analysis")
    parser.add_argument("--skip-umap", action="store_true",
                        help="Skip UMAP projections")

    # NRI only
    parser.add_argument("--nri-only", action="store_true",
                        help="Only run NRI comparisons on existing results")

    return parser.parse_args()


def find_latest_experiment_dir(experiment_name):
    """Find the most recent results folder for a given experiment name."""
    results_dir = config.get_experiments_dir()
    if not os.path.isdir(results_dir):
        return None

    matches = []
    for d in os.listdir(results_dir):
        if experiment_name in d and "fold_results" in d:
            matches.append(os.path.join(results_dir, d))

    if not matches:
        return None
    return sorted(matches)[-1]  # latest by timestamp prefix


def run_shap(exp_config, model, X_data, feature_names, output_dir, device=None):
    """Run SHAP analysis for a completed experiment."""
    import pickle
    import numpy as np
    import torch

    if exp_config.model_type == "xgboost":
        # Use full model trained on entire dataset
        full_model_path = os.path.join(output_dir, "full_model.pkl")
        if not os.path.exists(full_model_path):
            print(f"[SHAP] No full model found at {full_model_path}, skipping.")
            return

        with open(full_model_path, "rb") as f:
            full_model = pickle.load(f)

        # Scale the full dataset
        full_scaler_path = os.path.join(output_dir, "full_scaler.pkl")
        with open(full_scaler_path, "rb") as f:
            full_scaler = pickle.load(f)

        X_scaled = full_scaler.transform(X_data.astype(float))

        # Apply feature selection if used
        full_rfe_path = os.path.join(output_dir, "full_rfe.pkl")
        if os.path.exists(full_rfe_path):
            with open(full_rfe_path, "rb") as f:
                rfe = pickle.load(f)
            X_scaled = rfe.transform(X_scaled)
            feature_names = feature_names[rfe.support_]

        perform_shap_analysis_tree(full_model, X_scaled, feature_names, output_dir)

    elif exp_config.model_type == "transformer":
        # Load last fold model for SHAP (KernelExplainer)
        from src.models.transformer_model import TabularTransformer, LargeTabularTransformer
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_data.astype(float))
        X_scaled = np.nan_to_num(X_scaled, nan=0.0)

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Find a fold model to load
        model_path = os.path.join(output_dir, "fold_1_model.pt")
        if not os.path.exists(model_path):
            print("[SHAP] No fold model found, skipping.")
            return

        # Infer n_features from saved model to avoid shape mismatch
        state_dict = torch.load(model_path, map_location="cpu")
        n_features_model = state_dict["input_embedding.weight"].shape[1]

        # Check if current data matches model's expected features
        if X_scaled.shape[1] != n_features_model:
            print(f"[SHAP] Feature mismatch: data has {X_scaled.shape[1]} features, "
                  f"model expects {n_features_model}. Using first {n_features_model} features.")
            X_scaled = X_scaled[:, :n_features_model]
            feature_names = feature_names[:n_features_model]

        if exp_config.model_size == "large":
            model = LargeTabularTransformer(n_features_model)
        else:
            model = TabularTransformer(n_features_model)
        model.load_state_dict(state_dict)
        model.to("cpu")  # SHAP on CPU to avoid CUDA kernel errors with variable batch sizes
        model.eval()

        def predict_fn(X):
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X)
                outputs = model(X_tensor)
                probs = torch.softmax(outputs, dim=1).numpy()
            return probs

        perform_shap_analysis_kernel(predict_fn, X_scaled, feature_names, output_dir)


def main():
    args = parse_args()

    # Set data source
    if args.nonsense:
        config.USE_NONSENSE_DATA = True
    elif args.server:
        config.USE_NONSENSE_DATA = False

    print(f"Data source: {'nonsense' if config.USE_NONSENSE_DATA else 'server'}")

    # Plot styling
    setup_global_style()

    # ETL
    if args.etl:
        if config.USE_NONSENSE_DATA:
            print("[ETL] Skipping ETL (nonsense data already has features.csv)")
        else:
            from src.etl import run_etl
            run_etl()

    # NRI-only mode
    if args.nri_only:
        experiment_dirs = {}
        # Scan results directory for all experiment folders, not just those in config
        results_dir = config.get_experiments_dir()
        if os.path.isdir(results_dir):
            for d in os.listdir(results_dir):
                if "fold_results" not in d:
                    continue
                # Extract experiment name from folder (format: YYYYMMDD_HHMM_<name>_fold_results)
                parts = d.split("_")
                if len(parts) >= 4:
                    # Join everything between timestamp and "fold_results"
                    name = "_".join(parts[2:-2])
                    full_path = os.path.join(results_dir, d)
                    # Keep only the latest if multiple runs exist
                    if name not in experiment_dirs or d > os.path.basename(experiment_dirs[name]):
                        experiment_dirs[name] = full_path
        print(f"[NRI] Found {len(experiment_dirs)} experiment(s): {list(experiment_dirs.keys())}")
        run_nri_comparisons(experiment_dirs)
        return

    # Select experiments
    experiment_names = args.experiments or list(config.EXPERIMENTS.keys())

    # Run experiments
    experiment_dirs = {}

    for name in experiment_names:
        exp_config = config.EXPERIMENTS[name]

        # Override target if specified
        if args.target:
            exp_config = config.ExperimentConfig(
                **{**exp_config.__dict__, "target": args.target}
            )

        # Override SHAP/UMAP flags
        if args.skip_shap:
            exp_config = config.ExperimentConfig(
                **{**exp_config.__dict__, "perform_shap": False}
            )
        if args.skip_umap:
            exp_config = config.ExperimentConfig(
                **{**exp_config.__dict__, "perform_umap": False}
            )

        # Preprocess
        data = preprocess_data(exp_config)

        # Cross-validation
        output_dir = run_cross_validation(exp_config, data)
        experiment_dirs[name] = output_dir

        # SHAP
        if exp_config.perform_shap:
            run_shap(exp_config, None, data["features"], data["feature_names"],
                     output_dir)

        # UMAP
        if exp_config.perform_umap:
            generate_umap_projection(
                data["features"], data["labels"],
                output_dir, exp_config.name,
            )

    # NRI comparisons
    if len(experiment_dirs) > 1:
        run_nri_comparisons(experiment_dirs)
    else:
        print("\n[NRI] Skipping NRI (need at least 2 experiments)")

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
