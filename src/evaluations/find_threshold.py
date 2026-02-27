import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score, brier_score_loss
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import os



def load_fold_predictions(folder_path: Path) -> List[Dict]:
    """Load all fold prediction files from a results folder."""
    folds = []
    # Adjust the range to load folds numbered from 1 to 10
    for i in range(1, 11):
        file_path = folder_path / f"fold_{i}_predictions.json"
        if not file_path.exists():
            print(f"Warning: {file_path} not found, skipping")
            continue
        
        with open(file_path, 'r') as f:
            data = json.load(f)
            folds.append({
                'fold': i,
                'y_true': np.array(data['y_true']),
                'y_pred': np.array(data['y_pred']),
                'y_proba': np.array(data['y_proba'])
            })
    
    return folds

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict:
    """Calculate all evaluation metrics."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    metrics = {
        'accuracy': (tp + tn) / (tp + tn + fp + fn),
        'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'ppv': tp / (tp + fp) if (tp + fp) > 0 else 0,
        'npv': tn / (tn + fn) if (tn + fn) > 0 else 0,
        'f1': (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0,
        'alert_rate': (tp + fp) / len(y_true),
        'auroc': roc_auc_score(y_true, y_proba),
        'auprc': average_precision_score(y_true, y_proba),
        'brier_score': brier_score_loss(y_true, y_proba),  # Added Brier score calculation
        'tp': int(tp),
        'fp': int(fp),
        'tn': int(tn),
        'fn': int(fn),
        'n_samples': len(y_true)
    }

    return metrics

def find_optimal_threshold(y_true: np.ndarray, y_proba: np.ndarray, 
                          min_ppv: float = 0.25, 
                          max_alert_rate: float = 0.20) -> Tuple[float, pd.DataFrame]:
    """
    Find optimal threshold that maximizes sensitivity subject to constraints.
    
    Returns:
        optimal_threshold: The best threshold value
        threshold_df: DataFrame with metrics at all thresholds
    """
    thresholds = np.linspace(0.01, 0.99, 200)
    results = []
    
    for thresh in thresholds:
        y_pred = (y_proba >= thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        alert_rate = (tp + fp) / len(y_true)
        f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
        
        results.append({
            'threshold': thresh,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'ppv': ppv,
            'npv': npv,
            'f1': f1,
            'alert_rate': alert_rate,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'tn': tn
        })
    
    df = pd.DataFrame(results)
    
    # Find optimal: maximize sensitivity subject to PPV >= min_ppv and alert_rate <= max_alert_rate
    viable = df[(df['ppv'] >= min_ppv) & (df['alert_rate'] <= max_alert_rate)]
    
    if len(viable) == 0:
        print(f"Warning: No threshold meets constraints (PPV>={min_ppv}, alert_rate<={max_alert_rate})")
        print(f"Relaxing to PPV>={min_ppv*0.8}")
        viable = df[df['ppv'] >= min_ppv * 0.8]
        
        if len(viable) == 0:
            print("Warning: Even relaxed constraints not met. Using threshold with best F1.")
            optimal_idx = df['f1'].idxmax()
        else:
            optimal_idx = viable['sensitivity'].idxmax()
    else:
        optimal_idx = viable['sensitivity'].idxmax()
    
    optimal_threshold = df.loc[optimal_idx, 'threshold']
    
    return optimal_threshold, df

def plot_threshold_analysis(threshold_df: pd.DataFrame, optimal_threshold: float, 
                            save_path: Path):
    """Create visualization of threshold tradeoffs."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Sensitivity vs PPV
    axes[0, 0].plot(threshold_df['ppv'], threshold_df['sensitivity'], linewidth=2)
    axes[0, 0].axhline(y=0.75, color='r', linestyle='--', alpha=0.5, label='Target sens 75%')
    axes[0, 0].axvline(x=0.25, color='r', linestyle='--', alpha=0.5, label='Min PPV 25%')
    
    # Mark optimal point
    optimal_row = threshold_df[threshold_df['threshold'] == optimal_threshold].iloc[0]
    axes[0, 0].scatter([optimal_row['ppv']], [optimal_row['sensitivity']], 
                      color='red', s=100, zorder=5, label='Optimal')
    
    axes[0, 0].set_xlabel('PPV (Precision)', fontsize=12)
    axes[0, 0].set_ylabel('Sensitivity (Recall)', fontsize=12)
    axes[0, 0].set_title('Sensitivity vs PPV Trade-off', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # ROC-style curve
    axes[0, 1].plot(1 - threshold_df['specificity'], threshold_df['sensitivity'], linewidth=2)
    axes[0, 1].plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random')
    axes[0, 1].scatter([1 - optimal_row['specificity']], [optimal_row['sensitivity']], 
                      color='red', s=100, zorder=5, label='Optimal')
    axes[0, 1].set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    axes[0, 1].set_ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    axes[0, 1].set_title('ROC-style Curve', fontsize=14, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Alert rate vs Sensitivity
    axes[1, 0].plot(threshold_df['alert_rate'], threshold_df['sensitivity'], linewidth=2)
    axes[1, 0].axhline(y=0.75, color='r', linestyle='--', alpha=0.5)
    axes[1, 0].axvline(x=0.20, color='r', linestyle='--', alpha=0.5, label='Max alert 20%')
    axes[1, 0].scatter([optimal_row['alert_rate']], [optimal_row['sensitivity']], 
                      color='red', s=100, zorder=5, label='Optimal')
    axes[1, 0].set_xlabel('Alert Rate', fontsize=12)
    axes[1, 0].set_ylabel('Sensitivity', fontsize=12)
    axes[1, 0].set_title('Alert Rate vs Sensitivity', fontsize=14, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # All metrics vs threshold
    axes[1, 1].plot(threshold_df['threshold'], threshold_df['sensitivity'], 
                   label='Sensitivity', linewidth=2)
    axes[1, 1].plot(threshold_df['threshold'], threshold_df['specificity'], 
                   label='Specificity', linewidth=2)
    axes[1, 1].plot(threshold_df['threshold'], threshold_df['ppv'], 
                   label='PPV', linewidth=2)
    axes[1, 1].plot(threshold_df['threshold'], threshold_df['f1'], 
                   label='F1', linewidth=2, linestyle='--')
    axes[1, 1].axvline(x=optimal_threshold, color='r', linestyle='--', 
                      linewidth=2, label=f'Optimal ({optimal_threshold:.3f})')
    axes[1, 1].set_xlabel('Threshold', fontsize=12)
    axes[1, 1].set_ylabel('Metric Value', fontsize=12)
    axes[1, 1].set_title('Metrics vs Threshold', fontsize=14, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def analyze_results_folder(results_folder: Path, min_ppv: float = 0.25, 
                           max_alert_rate: float = 0.20):
    """Analyze a single results folder and save optimized metrics."""
    print(f"\n{'='*80}")
    print(f"Processing: {results_folder.name}")
    print(f"{'='*80}")
    
    # Load all folds
    folds = load_fold_predictions(results_folder)
    
    if len(folds) == 0:
        print("No valid folds found. Skipping.")
        return
    
    print(f"Loaded {len(folds)} folds")
    
    # Combine all folds
    y_true_all = np.concatenate([fold['y_true'] for fold in folds])
    y_proba_all = np.concatenate([fold['y_proba'] for fold in folds])
    
    print(f"Total samples: {len(y_true_all)}")
    print(f"Positive rate: {y_true_all.mean():.3f}")
    
    # Add an option to tune the threshold on 5 folds and evaluate on the other 5 folds
    use_half_folds = False  # Set this to True to enable the new behavior

    if use_half_folds:
        # Split folds into two groups: tuning_folds and evaluation_folds
        num_folds = len(folds)
        tuning_folds = folds[:num_folds // 2]  # First 5 folds
        evaluation_folds = folds[num_folds // 2:]  # Last 5 folds

        # Combine predictions and true labels for tuning folds
        y_true_tuning = np.concatenate([fold['y_true'] for fold in tuning_folds])
        y_proba_tuning = np.concatenate([fold['y_proba'] for fold in tuning_folds])

        # Find the optimal threshold using tuning folds
        optimal_threshold, threshold_metrics_df = find_optimal_threshold(y_true_tuning, y_proba_tuning, min_ppv, max_alert_rate)

        # Evaluate the optimal threshold on evaluation folds
        y_true_eval = np.concatenate([fold['y_true'] for fold in evaluation_folds])
        y_proba_eval = np.concatenate([fold['y_proba'] for fold in evaluation_folds])
        evaluation_metrics = calculate_metrics(y_true_eval, (y_proba_eval >= optimal_threshold).astype(int))

        # Print evaluation metrics for the evaluation folds
        print("Evaluation Metrics on Evaluation Folds:")
        for metric, value in evaluation_metrics.items():
            print(f"{metric}: {value:.4f}")

        # Evaluate the optimal threshold on all 10 folds
        y_true_all = np.concatenate([fold['y_true'] for fold in folds])
        y_proba_all = np.concatenate([fold['y_proba'] for fold in folds])
        all_folds_metrics = calculate_metrics(y_true_all, (y_proba_all >= optimal_threshold).astype(int))

        # Print evaluation metrics for all 10 folds
        print("Evaluation Metrics on All 10 Folds:")
        for metric, value in all_folds_metrics.items():
            print(f"{metric}: {value:.4f}")

        # Save metrics for all 10 folds as usual
        metrics_df = pd.DataFrame(threshold_metrics_df)
        metrics_df.to_csv(os.path.join(results_folder, "threshold_analysis.csv"), index=False)
        with open(os.path.join(results_folder, "optimized_metrics.json"), "w") as f:
            json.dump(all_folds_metrics, f, indent=4)
    else:
        # Existing behavior: tune and evaluate on all folds
        optimal_threshold, threshold_df = find_optimal_threshold(y_true_all, y_proba_all, min_ppv, max_alert_rate)

        # Save metrics for all 10 folds as usual
        metrics_df = pd.DataFrame(threshold_df)
        metrics_df.to_csv(os.path.join(results_folder, "threshold_analysis.csv"), index=False)
        with open(os.path.join(results_folder, "optimized_metrics.json"), "w") as f:
            json.dump(threshold_df.to_dict(orient="records"), f, indent=4)
    
    print(f"Optimal threshold: {optimal_threshold:.4f}")
    
    # Calculate metrics at optimal threshold
    y_pred_optimal = (y_proba_all >= optimal_threshold).astype(int)
    overall_metrics = calculate_metrics(y_true_all, y_pred_optimal, y_proba_all)
    overall_metrics['threshold'] = optimal_threshold
    
    print("\n" + "="*50)
    print("OPTIMIZED METRICS (all folds combined)")
    print("="*50)
    print(f"Threshold:   {optimal_threshold:.4f}")
    print(f"Sensitivity: {overall_metrics['sensitivity']:.3f}")
    print(f"Specificity: {overall_metrics['specificity']:.3f}")
    print(f"PPV:         {overall_metrics['ppv']:.3f}")
    print(f"NPV:         {overall_metrics['npv']:.3f}")
    print(f"F1:          {overall_metrics['f1']:.3f}")
    print(f"Alert Rate:  {overall_metrics['alert_rate']:.3f}")
    print(f"AUROC:       {overall_metrics['auroc']:.3f}")
    print(f"AUPRC:       {overall_metrics['auprc']:.3f}")
    
    # Calculate per-fold metrics at optimal threshold
    per_fold_metrics = []
    for fold in folds:
        y_pred_fold = (fold['y_proba'] >= optimal_threshold).astype(int)
        fold_metrics = calculate_metrics(fold['y_true'], y_pred_fold, fold['y_proba'])
        fold_metrics['fold'] = fold['fold']
        fold_metrics['threshold'] = optimal_threshold
        per_fold_metrics.append(fold_metrics)
    
    fold_df = pd.DataFrame(per_fold_metrics)
    
    print("\n" + "="*50)
    print("PER-FOLD METRICS (at optimal threshold)")
    print("="*50)
    print(fold_df[['fold', 'sensitivity', 'specificity', 'ppv', 'f1', 'alert_rate']].to_string(index=False))
    
    print("\nStability across folds:")
    for metric in ['sensitivity', 'specificity', 'ppv', 'f1']:
        mean_val = fold_df[metric].mean()
        std_val = fold_df[metric].std()
        print(f"{metric:12s}: {mean_val:.3f} (±{std_val:.3f})")
    
    # Create a subfolder for threshold analysis
    threshold_analysis_folder = os.path.join(results_folder, "threshold_analysis")
    os.makedirs(threshold_analysis_folder, exist_ok=True)
    
    # Save all files in the threshold_analysis folder
    threshold_df.to_csv(os.path.join(threshold_analysis_folder, "threshold_analysis.csv"), index=False)
    fold_df.to_csv(os.path.join(threshold_analysis_folder, "metrics_per_fold.csv"), index=False)
    with open(os.path.join(threshold_analysis_folder, "optimized_metrics.json"), "w") as f:
        json.dump(overall_metrics, f, indent=4)

    #############
    # Save results for a fixed threshold of 0.2
    fixed_threshold_folder = os.path.join(results_folder, "20percent_threshold")
    os.makedirs(fixed_threshold_folder, exist_ok=True)

    fixed_threshold = 0.2
    per_fold_metrics_fixed = []
    for fold in folds:
        y_pred_fold_fixed = (fold['y_proba'] >= fixed_threshold).astype(int)
        fold_metrics_fixed = calculate_metrics(fold['y_true'], y_pred_fold_fixed, fold['y_proba'])
        fold_metrics_fixed['fold'] = fold['fold']
        fold_metrics_fixed['threshold'] = fixed_threshold
        per_fold_metrics_fixed.append(fold_metrics_fixed)

    fold_df_fixed = pd.DataFrame(per_fold_metrics_fixed)
    fold_df_fixed.to_csv(os.path.join(fixed_threshold_folder, "metrics_per_fold.csv"), index=False)

    overall_metrics_fixed = calculate_metrics(y_true_all, (y_proba_all >= fixed_threshold).astype(int), y_proba_all)
    overall_metrics_fixed['threshold'] = fixed_threshold
    with open(os.path.join(fixed_threshold_folder, "overall_metrics.json"), "w") as f:
        json.dump(overall_metrics_fixed, f, indent=4)

    print(f"\nResults for fixed threshold (0.2) saved to: {fixed_threshold_folder}")
    #############

    # Save plots in the threshold_analysis folder
    plot_threshold_analysis(threshold_df, optimal_threshold, threshold_analysis_folder)
    
    # Save summary report
    with open(os.path.join(threshold_analysis_folder, "summary_report.txt"), 'w') as f:
        f.write(f"Threshold Optimization Results\n")
        f.write(f"{'='*80}\n\n")
        f.write(f"Model: {results_folder.name}\n")
        f.write(f"Total samples: {len(y_true_all)}\n")
        f.write(f"Positive rate: {y_true_all.mean():.3f}\n\n")
        f.write(f"Optimization constraints:\n")
        f.write(f"  - Minimum PPV: {min_ppv:.2f}\n")
        f.write(f"  - Maximum alert rate: {max_alert_rate:.2f}\n\n")
        f.write(f"Optimal threshold: {optimal_threshold:.4f}\n\n")
        f.write(f"Overall Metrics:\n")
        f.write(f"  Sensitivity: {overall_metrics['sensitivity']:.3f}\n")
        f.write(f"  Specificity: {overall_metrics['specificity']:.3f}\n")
        f.write(f"  PPV:         {overall_metrics['ppv']:.3f}\n")
        f.write(f"  NPV:         {overall_metrics['npv']:.3f}\n")
        f.write(f"  F1:          {overall_metrics['f1']:.3f}\n")
        f.write(f"  Alert Rate:  {overall_metrics['alert_rate']:.3f}\n")
        f.write(f"  AUROC:       {overall_metrics['auroc']:.3f}\n")
        f.write(f"  AUPRC:       {overall_metrics['auprc']:.3f}\n")
        f.write(f"  Brier Score: {overall_metrics['brier_score']:.3f}\n\n")  # Added Brier score to report
        f.write(f"Per-fold stability (mean ± std):\n")
        for metric in ['sensitivity', 'specificity', 'ppv', 'f1']:
            mean_val = fold_df[metric].mean()
            std_val = fold_df[metric].std()
            f.write(f"  {metric:12s}: {mean_val:.3f} (±{std_val:.3f})\n")
    
    print(f"\nResults saved to: {threshold_analysis_folder}")
    
    return overall_metrics, fold_df

def main(base_folder: str, min_ppv: float = 0.25, max_alert_rate: float = 0.20):
    """
    Main function to process all results folders.
    
    Args:
        base_folder: Path to folder containing *_results subfolders
        min_ppv: Minimum acceptable PPV (default: 0.25)
        max_alert_rate: Maximum acceptable alert rate (default: 0.20)
    """
    base_path = Path(base_folder)
    
    if not base_path.exists():
        print(f"Error: {base_folder} does not exist")
        return
    
    # Find all results folders
    results_folders = [f for f in base_path.iterdir() 
                      if f.is_dir() and f.name.endswith("_results")]
    
    if len(results_folders) == 0:
        print(f"No folders ending with '_results' found in {base_folder}")
        return
    
    print(f"Found {len(results_folders)} results folders")
    
    # Process each folder
    summary_results = []
    for folder in sorted(results_folders):
        try:
            overall_metrics, fold_df = analyze_results_folder(folder, min_ppv, max_alert_rate)
            summary_results.append({
                'model': folder.name,
                **overall_metrics
            })
        except Exception as e:
            print(f"Error processing {folder.name}: {e}")
            continue
    
    # Create comparison summary
    if len(summary_results) > 0:
        summary_df = pd.DataFrame(summary_results)
        comparison_path = base_path / "model_comparison.csv"
        summary_df.to_csv(comparison_path, index=False)
        
        print("\n" + "="*80)
        print("MODEL COMPARISON SUMMARY")
        print("="*80)
        print(summary_df[['model', 'threshold', 'sensitivity', 'specificity', 
                         'ppv', 'f1', 'alert_rate', 'auroc']].to_string(index=False))
        print(f"\nComparison saved to: {comparison_path}")

if __name__ == "__main__":
    import sys
    
    # Manual override - set these to use specific values instead of command line args
    MANUAL_BASE_FOLDER = "/data/kidney/Sacha/aki_ckd_prediction/experiments/results"  # e.g., "./experiments"
    MANUAL_MIN_PPV = 0.25  # e.g., 0.25
    MANUAL_MAX_ALERT_RATE = 0.20  # e.g., 0.20
    
    if MANUAL_BASE_FOLDER is not None:
        base_folder = MANUAL_BASE_FOLDER
        min_ppv = MANUAL_MIN_PPV if MANUAL_MIN_PPV is not None else 0.25
        max_alert_rate = MANUAL_MAX_ALERT_RATE if MANUAL_MAX_ALERT_RATE is not None else 0.20
    elif len(sys.argv) < 2:
        print("Usage: python threshold_optimization.py <path_to_folder> [min_ppv] [max_alert_rate]")
        print("Example: python threshold_optimization.py ./experiments 0.25 0.20")
        sys.exit(1)
    else:
        base_folder = sys.argv[1]
        min_ppv = float(sys.argv[2]) if len(sys.argv) > 2 else 0.25
        max_alert_rate = float(sys.argv[3]) if len(sys.argv) > 3 else 0.20

    
    main(base_folder, min_ppv, max_alert_rate)