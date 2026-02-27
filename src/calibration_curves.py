import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss


experiment_name_mapping = {
    "20250714_xgboost_abscore__fold_results": "XGBoost with Alberta Score Features",
    "20250731_xgboost_abpointsraw__fold_results": "XGBoost with Raw Alberta Features",
    "20250714_xgboost_abpoints__fold_results": "XGBoost with Alberta Points Features",
    "20250714_xgboost_hing_fselect100_fold_results": "XGBoost with Extended Features (Top 100 Selected)",
    "20250711_smalltransformer_hing_fselect50_fold_results": "Small Transformer with Extended Features (Top 50 Selected)",
    "20250711_smalltransformer_abpoints__fold_results": "Small Transformer with Alberta Points Features",
    "20251126_smalltransformer_abraw__fold_results": "Small Transformer with Alberta Raw Features",
    # Add more mappings as needed
}


def generate_calibration_curve(experiments_path):
    """
    Generate calibration curves for all experiments in the given folder.

    Args:
        experiments_path (str): Path to the experiments folder.
    """

    # Loop through all experiment folders
    for experiment_folder in os.listdir(experiments_path):
        experiment_path = os.path.join(experiments_path, experiment_folder)
        if not os.path.isdir(experiment_path):
            continue  # Skip if not a folder

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

        # Generate calibration curve
        prob_true, prob_pred = calibration_curve(all_labels, all_predictions, n_bins=10, strategy="uniform")

        # Calculate Brier score
        brier_score = brier_score_loss(all_labels, all_predictions)

        # Plot calibration curve
        plt.figure(figsize=(8, 6))
        plt.plot(prob_pred, prob_true, marker="o", label="Calibration curve")
        plt.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
        display_name = experiment_name_mapping.get(experiment_folder, experiment_folder)
        plt.title(f"Calibration Curve for {display_name}")
        plt.xlabel("Mean Predicted Probability")
        plt.ylabel("Fraction of Positives")
        plt.legend(loc="best")
        plt.grid()
        plt.text(0.6, 0.2, f"Brier Score: {brier_score:.4f}", fontsize=10, bbox=dict(facecolor="white", alpha=0.8))

        # Save the plot
        output_path = os.path.join(experiment_path, "calibration_curve.png")
        plt.savefig(output_path)
        plt.close()

        print(f"Calibration curve saved to: {output_path}")

if __name__ == "__main__":
    # Example usage
    experiments_folder = "/data/kidney/Sacha/aki_ckd_prediction/experiments/results"
    generate_calibration_curve(experiments_folder)