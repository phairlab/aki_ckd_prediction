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
Net Reclassification Improvement (NRI) Analysis Script

This script calculates the Net Reclassification Improvement between two trained models
using their cross-validation results. It compares how well models reclassify patients
across clinically meaningful risk categories.

Usage:
    python net_reclassification_results.py --exp1_path <path_to_experiment1> --exp2_path <path_to_experiment2>

The script expects experiment folders containing:
- fold_X_predictions.json files with test predictions for each fold
- Consistent test patient IDs across both experiments

Risk categories are defined as:
- Low risk: < 10% probability
- Intermediate risk: 10-20% probability  
- High risk: ≥ 20% probability

# Basic usage with default 10% and 20% thresholds
python src/net_reclassification_results.py \
  --exp1_path /data/kidney/Sacha/aki_ckd_prediction/experiments/20250711_smalltransformer_abpoints__fold_results \
  --exp2_path /data/kidney/Sacha/aki_ckd_prediction/experiments/20250711_smalltransformer_hing_fselect50_fold_results

"""

import argparse
import json
import os
import numpy as np
import pandas as pd
from pathlib import Path
import sys

def calculate_nri(y_true, y_pred_model1, y_pred_model2, thresholds):
    """
    Calculate Net Reclassification Improvement (NRI)
    
    Parameters:
    y_true: actual outcomes (0/1)
    y_pred_model1: predicted probabilities from baseline model
    y_pred_model2: predicted probabilities from new model
    thresholds: list of risk category thresholds [low_threshold, high_threshold]
    
    Returns:
    dict: NRI results including total NRI, NRI for cases, NRI for controls
    """
    
    def categorize_risk(probs, thresholds):
        """
        Categorize probabilities into risk groups based on thresholds
        
        Parameters:
        probs: array of probabilities
        thresholds: sorted list of threshold values
        
        Returns:
        categories: array of integers from 0 to len(thresholds), 
                  where 0 is lowest risk (below first threshold)
                  and len(thresholds) is highest risk (above last threshold)
        """
        categories = np.zeros(len(probs), dtype=int)
        for i, threshold in enumerate(thresholds):
            categories[probs >= threshold] = i + 1
        return categories
    
    # Categorize predictions
    risk_cat_1 = categorize_risk(y_pred_model1, thresholds)
    risk_cat_2 = categorize_risk(y_pred_model2, thresholds)
    
    # Separate cases (events) and controls (non-events)
    cases_mask = y_true == 1
    controls_mask = y_true == 0
    
    # Calculate reclassification for cases (events)
    cases_up = np.sum((risk_cat_2[cases_mask] > risk_cat_1[cases_mask]))
    cases_down = np.sum((risk_cat_2[cases_mask] < risk_cat_1[cases_mask]))
    cases_total = np.sum(cases_mask)
    
    # Calculate reclassification for controls (non-events)
    controls_up = np.sum((risk_cat_2[controls_mask] > risk_cat_1[controls_mask]))
    controls_down = np.sum((risk_cat_2[controls_mask] < risk_cat_1[controls_mask]))
    controls_total = np.sum(controls_mask)
    
    # Calculate NRI components
    nri_cases = (cases_up - cases_down) / cases_total if cases_total > 0 else 0
    nri_controls = (controls_down - controls_up) / controls_total if controls_total > 0 else 0
    
    # Total NRI
    nri = nri_cases + nri_controls
    
    return {
        'NRI_total': nri,
        'NRI_cases': nri_cases,
        'NRI_controls': nri_controls,
        'cases_up': cases_up,
        'cases_down': cases_down,
        'controls_up': controls_up,
        'controls_down': controls_down,
        'cases_total': cases_total,
        'controls_total': controls_total
    }

def load_fold_predictions(experiment_path, fold_num):
    """
    Load predictions for a specific fold from an experiment
    
    Parameters:
    experiment_path: path to experiment folder
    fold_num: fold number (1-10)
    
    Returns:
    dict: loaded predictions data
    """
    predictions_file = os.path.join(experiment_path, f"fold_{fold_num}_predictions.json")
    
    if not os.path.exists(predictions_file):
        raise FileNotFoundError(f"Predictions file not found: {predictions_file}")
    
    with open(predictions_file, 'r') as f:
        predictions_data = json.load(f)
    
    return predictions_data

def verify_test_indices_match(exp1_data, exp2_data, fold_num):
    """
    Verify that both experiments used the same test indices for a fold
    
    Parameters:
    exp1_data: predictions data from experiment 1
    exp2_data: predictions data from experiment 2  
    fold_num: fold number for error reporting
    
    Returns:
    bool: True if indices match, raises ValueError if not
    """
    indices1 = np.array(exp1_data['test_indices'])
    indices2 = np.array(exp2_data['test_indices'])
    
    if not np.array_equal(indices1, indices2):
        raise ValueError(f"Test indices do not match for fold {fold_num}!\n"
                        f"Experiment 1 has {len(indices1)} indices, "
                        f"Experiment 2 has {len(indices2)} indices.\n"
                        f"First 10 indices from exp1: {indices1[:10]}\n"
                        f"First 10 indices from exp2: {indices2[:10]}")
    
    return True

def calculate_fold_nri(exp1_path, exp2_path, fold_num, thresholds):
    """
    Calculate NRI for a single fold
    
    Parameters:
    exp1_path: path to experiment 1 folder
    exp2_path: path to experiment 2 folder
    fold_num: fold number
    thresholds: risk category thresholds
    
    Returns:
    dict: NRI results for this fold
    """
    # Load predictions for both experiments
    exp1_data = load_fold_predictions(exp1_path, fold_num)
    exp2_data = load_fold_predictions(exp2_path, fold_num)
    
    # Verify test indices match
    verify_test_indices_match(exp1_data, exp2_data, fold_num)
    
    # Extract data
    y_true = np.array(exp1_data['y_true'])
    y_proba_exp1 = np.array(exp1_data['y_proba'])
    y_proba_exp2 = np.array(exp2_data['y_proba'])
    
    # Verify same number of predictions
    if len(y_true) != len(y_proba_exp1) or len(y_true) != len(y_proba_exp2):
        raise ValueError(f"Prediction array lengths don't match for fold {fold_num}")
    
    # Calculate NRI
    nri_results = calculate_nri(y_true, y_proba_exp1, y_proba_exp2, thresholds)
    
    return nri_results

def get_experiment_info(exp_path):
    """
    Get experiment information from args.json file
    
    Parameters:
    exp_path: path to experiment folder
    
    Returns:
    dict: experiment configuration
    """
    args_file = os.path.join(exp_path, "args.json")
    
    if os.path.exists(args_file):
        with open(args_file, 'r') as f:
            return json.load(f)
    else:
        return {"info": "No args.json found"}

def calculate_nri_across_folds(exp1_path, exp2_path, thresholds=[0.1, 0.2], n_folds=10):
    """
    Calculate NRI across all cross-validation folds
    
    Parameters:
    exp1_path: path to experiment 1 folder (baseline model)
    exp2_path: path to experiment 2 folder (new model)
    thresholds: list of risk category thresholds, must be sorted in ascending order
                e.g., [0.1, 0.2] creates categories: <10%, 10-20%, ≥20%
                or [0.1, 0.2, 0.3] creates: <10%, 10-20%, 20-30%, ≥30%
    n_folds: number of CV folds
    
    Returns:
    dict: comprehensive NRI results
    """
    
    print(f"Calculating NRI between:")
    print(f"  Baseline model (exp1): {exp1_path}")
    print(f"  New model (exp2): {exp2_path}")
    print(f"  Risk thresholds: {[f'{t*100:.0f}%' for t in thresholds]}")
    print()
    
    # Get experiment information
    exp1_info = get_experiment_info(exp1_path)
    exp2_info = get_experiment_info(exp2_path)
    
    print("Experiment 1 configuration:")
    for key, value in exp1_info.items():
        print(f"  {key}: {value}")
    print()
    
    print("Experiment 2 configuration:")
    for key, value in exp2_info.items():
        print(f"  {key}: {value}")
    print()
    
    # Store results for each fold
    fold_results = []
    fold_nri_values = []
    
    # Calculate NRI for each fold
    for fold in range(1, n_folds + 1):
        try:
            print(f"Processing fold {fold}...")
            
            nri_result = calculate_fold_nri(exp1_path, exp2_path, fold, thresholds)
            fold_results.append(nri_result)
            fold_nri_values.append(nri_result['NRI_total'])
            
            print(f"  Fold {fold} NRI: {nri_result['NRI_total']:.4f}")
            print(f"    Cases: {nri_result['cases_up']} up, {nri_result['cases_down']} down "
                  f"(total: {nri_result['cases_total']})")
            print(f"    Controls: {nri_result['controls_up']} up, {nri_result['controls_down']} down "
                  f"(total: {nri_result['controls_total']})")
            
        except Exception as e:
            print(f"Error processing fold {fold}: {str(e)}")
            raise
    
    # Calculate aggregated statistics
    mean_nri = np.mean(fold_nri_values)
    std_nri = np.std(fold_nri_values)
    
    # Aggregate across all folds
    total_cases_up = sum(r['cases_up'] for r in fold_results)
    total_cases_down = sum(r['cases_down'] for r in fold_results)
    total_cases = sum(r['cases_total'] for r in fold_results)
    
    total_controls_up = sum(r['controls_up'] for r in fold_results)
    total_controls_down = sum(r['controls_down'] for r in fold_results)
    total_controls = sum(r['controls_total'] for r in fold_results)
    
    # Calculate overall NRI across all folds
    overall_nri_cases = (total_cases_up - total_cases_down) / total_cases if total_cases > 0 else 0
    overall_nri_controls = (total_controls_down - total_controls_up) / total_controls if total_controls > 0 else 0
    overall_nri = overall_nri_cases + overall_nri_controls
    
    print("\n" + "="*60)
    print("NRI RESULTS SUMMARY")
    print("="*60)
    print(f"Risk Categories:")
    # First category (lowest risk)
    print(f"  Category 0:        < {thresholds[0]*100:.0f}%")
    # Middle categories
    for i in range(len(thresholds)-1):
        print(f"  Category {i+1}:      {thresholds[i]*100:.0f}%-{thresholds[i+1]*100:.0f}%")
    # Last category (highest risk)
    print(f"  Category {len(thresholds)}: ≥ {thresholds[-1]*100:.0f}%")
    print()
    print(f"Fold-wise NRI:")
    print(f"  Mean ± SD:         {mean_nri:.4f} ± {std_nri:.4f}")
    print(f"  Range:             {min(fold_nri_values):.4f} to {max(fold_nri_values):.4f}")
    print()
    print(f"Overall NRI (pooled across folds):")
    print(f"  Total NRI:         {overall_nri:.4f}")
    print(f"  NRI for cases:     {overall_nri_cases:.4f}")
    print(f"  NRI for controls:  {overall_nri_controls:.4f}")
    print()
    print(f"Reclassification Summary:")
    print(f"  Cases (n={total_cases}):")
    print(f"    Moved up:        {total_cases_up} ({total_cases_up/total_cases*100:.1f}%)")
    print(f"    Moved down:      {total_cases_down} ({total_cases_down/total_cases*100:.1f}%)")
    print(f"  Controls (n={total_controls}):")
    print(f"    Moved up:        {total_controls_up} ({total_controls_up/total_controls*100:.1f}%)")
    print(f"    Moved down:      {total_controls_down} ({total_controls_down/total_controls*100:.1f}%)")
    
    # Interpretation
    print()
    print("Interpretation:")
    if overall_nri > 0:
        print(f"  The new model (exp2) provides better risk reclassification than the baseline model (exp1).")
        print(f"  An NRI of {overall_nri:.4f} indicates net improvement in patient risk classification.")
    elif overall_nri < 0:
        print(f"  The baseline model (exp1) provides better risk reclassification than the new model (exp2).")
        print(f"  An NRI of {overall_nri:.4f} indicates net worsening in patient risk classification.")
    else:
        print(f"  Both models provide equivalent risk reclassification performance.")
    
    # Compile results
    results = {
        'experiment_1_path': exp1_path,
        'experiment_2_path': exp2_path,
        'thresholds': thresholds,
        'n_folds': n_folds,
        'fold_nri_values': fold_nri_values,
        'fold_results': fold_results,
        'mean_nri': mean_nri,
        'std_nri': std_nri,
        'overall_nri': overall_nri,
        'overall_nri_cases': overall_nri_cases,
        'overall_nri_controls': overall_nri_controls,
        'total_cases': total_cases,
        'total_controls': total_controls,
        'total_cases_up': total_cases_up,
        'total_cases_down': total_cases_down,
        'total_controls_up': total_controls_up,
        'total_controls_down': total_controls_down,
        'experiment_1_info': exp1_info,
        'experiment_2_info': exp2_info
    }
    
    return results

def main():    
    # Development dictionary - comment/uncomment this section as needed
    args_dict = {
        'experiments_dir': '/data/kidney/Sacha/aki_ckd_prediction/experiments/results',
        'thresholds': [0.2],  # can be extended e.g., [0.1, 0.2, 0.3, 0.4]
        'n_folds': 10,
        'output_dir': '/data/kidney/Sacha/aki_ckd_prediction/experiments/nri_comparisons'
    }
    class Args:
        def __init__(self, args_dict):
            for k, v in args_dict.items():
                setattr(self, k, v)
    args = Args(args_dict)


    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Scan directory for experiment folders
    if not os.path.exists(args.experiments_dir):
        print(f"Error: Experiments directory not found: {args.experiments_dir}")
        sys.exit(1)

    # Get all subdirectories that contain 'fold_results' in their name
    experiment_paths = [
        os.path.join(args.experiments_dir, d) 
        for d in os.listdir(args.experiments_dir) 
        if os.path.isdir(os.path.join(args.experiments_dir, d)) and 'fold_results' in d
    ]

    # ###### TESTING
    # experiment_paths = [
    #     "/data/kidney/Sacha/aki_ckd_prediction/experiments/results/20250714_xgboost_abscore__fold_results"
    #     "/data/kidney/Sacha/aki_ckd_prediction/experiments/results/20251126_smalltransformer_abraw__fold_results",
    # ]
    # ###### END TESTING

    # Validate all paths exist
    invalid_paths = [path for path in experiment_paths if not os.path.exists(path)]
    if invalid_paths:
        print("Error: The following experiment paths do not exist:")
        for path in invalid_paths:
            print(f"  {path}")
        sys.exit(1)

    print(f"Found {len(experiment_paths)} experiments to compare")
    
    # Generate all unique combinations of experiments
    from itertools import combinations
    experiment_pairs = list(combinations(experiment_paths, 2))
    print(f"Will perform {len(experiment_pairs)} comparisons")

    # Validate thresholds
    if not args.thresholds:
        print("Error: Thresholds list cannot be empty")
        sys.exit(1)
    
    # Check thresholds are in ascending order and within valid range
    prev = 0
    for t in args.thresholds:
        if not (0 < t < 1):
            print(f"Error: All thresholds must be between 0 and 1. Got: {t}")
            sys.exit(1)
        if t <= prev:
            print(f"Error: Thresholds must be in strictly ascending order. Got: {args.thresholds}")
            sys.exit(1)
        prev = t
    
    # Helper function for JSON serialization
    def convert_for_json(obj):
        """Recursively convert numpy types to native Python types"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, dict):
            return {key: convert_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(item) for item in obj]
        else:
            return obj

    # Process each pair
    successful_comparisons = []
    failed_comparisons = []
    
    for exp1_path, exp2_path in experiment_pairs:
        try:
            print(f"\nComparing:")
            print(f"Experiment 1: {exp1_path}")
            print(f"Experiment 2: {exp2_path}")

            # Generate output filename
            exp1_name = os.path.basename(exp1_path)
            exp2_name = os.path.basename(exp2_path)
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(args.output_dir, f"nri_{exp1_name}_vs_{exp2_name}_{timestamp}.json")
            print(f"Output will be saved to: {output_file}")

            results = calculate_nri_across_folds(exp1_path, exp2_path,
                                               thresholds=args.thresholds, n_folds=args.n_folds)
            
            # Convert to JSON-serializable format
            results_serializable = convert_for_json(results)
            
            # Save results
            with open(output_file, 'w') as f:
                json.dump(results_serializable, f, indent=4)
            
            print(f"Results saved to: {output_file}")
            successful_comparisons.append((exp1_path, exp2_path))
            
        except Exception as e:
            print(f"Error comparing {exp1_path} vs {exp2_path}: {str(e)}")
            failed_comparisons.append((exp1_path, exp2_path, str(e)))
            continue  # Continue with next pair

    # Print summary
    print("\nComparison Summary")
    print("="*60)
    print(f"Total comparisons attempted: {len(experiment_pairs)}")
    print(f"Successful comparisons: {len(successful_comparisons)}")
    print(f"Failed comparisons: {len(failed_comparisons)}")
    
    if failed_comparisons:
        print("\nFailed Comparisons:")
        for exp1, exp2, error in failed_comparisons:
            print(f"\nExp1: {exp1}")
            print(f"Exp2: {exp2}")
            print(f"Error: {error}")

if __name__ == "__main__":
    main()
