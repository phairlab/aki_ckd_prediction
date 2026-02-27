import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
import numpy as np
import matplotlib.pyplot as plt


experiment_name_mapping = {
    "20250714_xgboost_abscore__fold_results": "XGBoost with Alberta Score",
    "20250731_xgboost_abpointsraw__fold_results": "XGBoost with Raw Alberta Features",
    # "20250714_xgboost_abpoints__fold_results": "XGBoost with Alberta Points Features",
    "20250714_xgboost_hing_fselect100_fold_results": "XGBoost with Extended Features (Top 100 Selected)",
    "20250711_smalltransformer_hing_fselect50_fold_results": "Small Transformer with Extended Features (Top 50 Selected)",
    "20251219_smalltransformer_hing_fselect100_fold_results": "Small Transformer with Extended Features (Top 100 Selected)",
    # "20250711_smalltransformer_abpoints__fold_results": "Small Transformer with Alberta Points Features",
    "20251126_smalltransformer_abraw__fold_results": "Small Transformer with Alberta Raw Features",
    "20251201_smalltransformer_egfr__fold_results": "Small Transformer with eGFR Features",
    # Add more mappings as needed
}


def generate_plots(experiments_path):
    """
    Generate decision curves for all experiments in the given folder.
    Also creates a combined plot with all decision curves.

    Args:
        experiments_path (str): Path to the experiments folder.
    """
    all_dca_results = {}  # Store all results for combined plot

    # Loop through all experiment folders
    for experiment_folder in os.listdir(experiments_path):
        experiment_path = os.path.join(experiments_path, experiment_folder)
        if not os.path.isdir(experiment_path):
            continue  # Skip if not a folder
        
        if experiment_folder not in experiment_name_mapping:
            continue  # Skip if not in mapping

        print(f"Processing experiment: {experiment_folder}")

        # Collect predictions and true labels from all folds
        all_predictions = []
        all_labels = []

        for file_name in os.listdir(experiment_path):
            if file_name.startswith("fold_") and file_name.endswith("_predictions.json"):
                file_path = os.path.join(experiment_path, file_name)
                with open(file_path, "r") as f:
                    data = json.load(f)
                    # Use the correct keys from the JSON file
                    all_predictions.extend(data["y_proba"])  # Use 'y_proba' for probabilities
                    all_labels.extend(data["y_true"])       # Use 'y_true' for true labels

        # Ensure we have data to process
        if not all_predictions or not all_labels:
            print(f"No predictions found for {experiment_folder}. Skipping...")
            continue

        # Generate decision curve
        dca_results = decision_curve_analysis(np.array(all_labels), np.array(all_predictions))
        model_name = experiment_name_mapping.get(experiment_folder, experiment_folder)

        # Save individual DCA plot
        output_path_dca = os.path.join(experiment_path, "decision_curve.png")
        plot_dca(dca_results, output_path_dca, model_name=model_name)

        # Save individual risk distribution plot
        output_path_risk = os.path.join(experiment_path, "risk_distribution.png")
        plot_risk_distribution(np.array(all_predictions), np.array(all_labels), output_path_risk, model_name=model_name)

        # Store results for combined plot
        all_dca_results[model_name] = dca_results

    # Generate combined plot
    if all_dca_results:
        combined_output_path = os.path.join(experiments_path, "combined_decision_curves.png")
        # Sort results to match the order in experiment_name_mapping
        ordered_dca_results = {experiment_name_mapping[key]: all_dca_results[experiment_name_mapping[key]] 
                               for key in experiment_name_mapping.keys() 
                               if experiment_name_mapping[key] in all_dca_results}
        plot_combined_dca(ordered_dca_results, combined_output_path)
        print(f"Combined decision curve saved to {combined_output_path}")



def plot_risk_distribution(y_pred_proba, y_true, output_path, model_name='Model'):
    """
    Plot the distribution of predicted probabilities for positive and negative cases.
    
    Args:
        y_pred_proba: array of predicted probabilities
        y_true: array of true labels (0/1)
        output_path: path to save the plot
        model_name: name of the model for the plot title
    """
    plt.figure(figsize=(10, 6))
    
    # Separate predictions by outcome
    pos_probs = y_pred_proba[y_true == 1]
    neg_probs = y_pred_proba[y_true == 0]
    
    # Plot histograms
    plt.hist(neg_probs, bins=50, alpha=0.5, label=f'Negative Cases (n={len(neg_probs)})', 
                color='blue', density=True)
    plt.hist(pos_probs, bins=50, alpha=0.5, label=f'Positive Cases (n={len(pos_probs)})', 
                color='red', density=True)
    
    plt.xlabel('Predicted Probability', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title(f'Risk Distribution - {model_name}', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.xlim([0, 1])
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()



def decision_curve_analysis(y_true, y_pred_proba, thresholds=None):
    """
    Perform decision curve analysis.
    
    Returns: dict with thresholds and net benefits for model and baselines
    """
    if thresholds is None:
        # thresholds = np.arange(0.01, 1.0, 0.01)
        thresholds = np.arange(0.001, 0.50, 0.001)  # Much denser, lower range
    
    # Calculate net benefit for your model
    nb_model = [net_benefit(y_true, y_pred_proba, t) for t in thresholds]
    
    # Treat all baseline
    prevalence = np.mean(y_true)
    nb_all = [prevalence - (1 - prevalence) * (t / (1 - t)) for t in thresholds]
    
    # Treat none baseline (always 0)
    nb_none = [0] * len(thresholds)
    
    return {
        'thresholds': thresholds,
        'net_benefit_model': nb_model,
        'net_benefit_all': nb_all,
        'net_benefit_none': nb_none
    }


def net_benefit(y_true, y_pred_proba, threshold):
    """
    Calculate net benefit at a given threshold.
    
    y_true: array of true outcomes (0/1)
    y_pred_proba: array of predicted probabilities
    threshold: risk threshold for intervention
    """
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    n = len(y_true)
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    
    net_benefit = (tp / n) - (fp / n) * (threshold / (1 - threshold))
    return net_benefit


def plot_dca(results, output_path, model_name='Model'):
    """Plot decision curve."""
    plt.figure(figsize=(10, 6))
    plt.plot(results['thresholds'], results['net_benefit_model'], 
             label=model_name, linewidth=2)
    plt.plot(results['thresholds'], results['net_benefit_all'], 
             label='Treat All', linestyle='--')
    plt.plot(results['thresholds'], results['net_benefit_none'], 
             label='Treat None', linestyle='--', color='black')
    
    plt.xlabel('Threshold Probability')
    plt.ylabel('Net Benefit')
    plt.legend()
    plt.grid(alpha=0.3)
    # plt.xlim([0, 1])

    # In your plot_combined_dca function
    plt.xlim([0, 0.5])  # Focus on clinically relevant range
    plt.ylim([-0.02, 0.08])  # Zoom into where net benefit lives

    plt.title('Decision Curve Analysis')

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_combined_dca(all_results, output_path):
    """Plot all decision curves on a single plot."""
    plt.figure(figsize=(12, 8))
    
    # Plot each model's decision curve
    for model_name, results in all_results.items():
        plt.plot(results['thresholds'], results['net_benefit_model'], 
                 label=model_name, linewidth=2)
    
    # Plot baselines (use the first result's baselines as they're all the same)
    first_result = next(iter(all_results.values()))
    plt.plot(first_result['thresholds'], first_result['net_benefit_all'], 
             label='Treat All', linestyle='--', color='gray', linewidth=2)
    plt.plot(first_result['thresholds'], first_result['net_benefit_none'], 
             label='Treat None', linestyle='--', color='black', linewidth=2)
    
    plt.xlabel('Threshold Probability', fontsize=12)
    plt.ylabel('Net Benefit', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(alpha=0.3)
    # plt.xlim([0, 1])

    # In your plot_combined_dca function
    plt.xlim([0, 0.5])  # Focus on clinically relevant range
    plt.ylim([-0.02, 0.08])  # Zoom into where net benefit lives

    plt.title('Decision Curve Analysis - All Models', fontsize=14)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


# Usage:
# results = decision_curve_analysis(y_true, y_pred_proba)
# plot_dca(results, model_name='XGBoost CKD Model')

if __name__ == "__main__":
    # Example usage
    experiments_folder = "/data/kidney/Sacha/aki_ckd_prediction/experiments/results"
    generate_plots(experiments_folder)
