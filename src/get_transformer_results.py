import os
import re
import random
import math
import time
import json
import pickle
import argparse
import shap
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.metrics import precision_recall_curve, average_precision_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# from alberta_score_helpers import *
from data_preprocessing import *
from transformer_model_helpers import *


random.seed(1202)
np.random.seed(1202) 
torch.manual_seed(1202)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1202)


def fetch_args():
	# Development dictionary - comment/uncomment this section as needed
	args_dict = {
	    'hing_features': True,
	    'alberta_features': False,
	    'alberta_score': False,
		'target': 'ckd',  # 'ckd' or 'ckdordeath'
        'perform_cv': True,
        'perform_shapanalysis': False,
        'epochs': 50,            # Increased max epochs
        'batch_size': 32,        # Batch size for training
        'learning_rate': 5e-5,   # Reduced learning rate for slower, more stable convergence
        'early_stopping': 10,    # Increased patience - wait longer before stopping
        'validation_split': 0.15 # Portion of training data to use for validation
	}
	
	class Args:
	    def __init__(self, args_dict):
	        for k, v in args_dict.items():
	            setattr(self, k, v)
	
	return Args(args_dict)
	
	# # Command line argument parsing
	# parser = argparse.ArgumentParser(description='Run AKI-CKD prediction model')
	# parser.add_argument('--hing_features', action='store_true', default=False
	# 					help='Include features engineered by Hing')
	# parser.add_argument('--alberta_features', action='store_true', default=True, 
	# 					help='Include Alberta score component features')
	# parser.add_argument('--alberta_score', action='store_true', default=False,
	# 					help='Include Alberta score as a feature')
	# parser.add_argument('--perform_cv', action='store_true', default=True
	# 					help='Perform cross-validation step')
	# parser.add_argument('--perform_shapanalysis', action='store_true', default=True,
	# 					help='Perform SHAP analysis on the final model')
	
	# return parser.parse_args()

args = fetch_args()


# --------------------------------------------------------
# -*- set up the model type and device if we're using GPUs -*-
# --------------------------------------------------------

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name()}")

# Set up device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# --------------------------------------------------------
# -*- get preprocessed/cleaned dataset -*-
# --------------------------------------------------------

features_used, attributes_used, feature_names, features_df = preprocess_data(hing_features=args.hing_features, 
                                                                             alberta_features=args.alberta_features, 
                                                                             alberta_score=args.alberta_score,
                                                                             target=args.target,
                                                                             model_type="transformer")


# --------------------------------------------------------
# -*- run training loop and get results -*-
# --------------------------------------------------------

# Define the date and feature type for the subfolder name
current_date = datetime.now().strftime("%Y%m%d")
features_list = []
if args.hing_features: features_list.append("hing")
if args.alberta_features: features_list.append("abpoints")
if args.alberta_score: features_list.append("abscore")
extra = ""

features_string = "-".join(features_list)
subfolder_name = f"{current_date}_transformer_{features_string}_{extra}_fold_results"

# Change the directory to /data/kidney/Sacha/aki_ckd_prediction/
os.chdir("/data/kidney/Sacha/aki_ckd_prediction")

# Create the experiments folder and subfolder for this run
os.makedirs(f"experiments/{subfolder_name}", exist_ok=True)

# Initialize variables for storing results across folds
tprs = []  # True positive rates for ROC curve
aucs1 = []  # AUC values for ROC curve
aucs2 = []  # AUC values for Precision-Recall curve
precisions = []  # Precision values for PRC curve
mean_fpr1 = np.linspace(0, 1, 100)  # Mean false positive rates for ROC
mean_fpr2 = np.linspace(0, 1, 100)  # Mean recall values for PRC
i = 0  # Fold counter
accuracy_sum = 0  # Sum of accuracies across folds
sensitivity_sum = 0  # Sum of sensitivities across folds
specificity_sum = 0  # Sum of specificities across folds
ppv_sum = 0  # Sum of positive predictive values across folds
npv_sum = 0  # Sum of negative predictive values across folds
f1_sum = 0  # Sum of F1 scores across folds
y_real = []  # True labels for all folds
y_proba = []  # Predicted probabilities for all folds
best_threshold_roc = 0  # Sum of best thresholds for ROC curve
best_threshold_prc = 0  # Sum of best thresholds for PRC curve
all_val_losses = []  # Store validation losses for each fold
all_train_losses = []  # Store training losses for each fold
best_epochs = []  # Store the best epoch for each fold

# Initialize 10-fold cross-validation
cv = KFold(n_splits=10, shuffle=True, random_state=1202)
skip = 0

if args.perform_cv:
    # Perform cross-validation
    for i, (train, test) in enumerate(cv.split(features_used, attributes_used)):
        if i < skip:  # Skip the first skip folds (for if training gets halted)
            continue
        
        t0 = time.time()
        print(f"Training fold {i + 1}...")

        X_train, y_train = features_used[train].astype(float), attributes_used[train].astype('int8')
        X_test, y_test = features_used[test].astype(float), attributes_used[test].astype('int8')

        # # Handle NaN values in the baseline_creatinine_points column
        # if np.isnan(X_train).any() or np.isnan(X_val).any() or np.isnan(X_test).any():
        #     print("Found NaN values, filling with median values...")
        #     # Get column indices with NaN values
        #     nan_cols = np.where(np.isnan(X_train).any(axis=0))[0]
            
        #     for col in nan_cols:
        #         # Calculate median for each column (excluding NaN values)
        #         col_median = np.nanmedian(X_train[:, col])
        #         # Fill NaN values with median
        #         X_train[:, col] = np.nan_to_num(X_train[:, col], nan=col_median)
        #         X_test[:, col] = np.nan_to_num(X_test[:, col], nan=col_median)


        # Feature normalization is crucial for transformer models
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Train model with validation-based early stopping
        classifier, train_losses, val_losses, best_epoch = train_with_validation(
            X_train, y_train, 
            device=device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            validation_split=args.validation_split,
            early_stopping=args.early_stopping,
            learning_rate=args.learning_rate,
            model_size="small" #"large" if args.hing_features else "small" 
        )
        
        # Store training metrics
        all_train_losses.append(train_losses)
        all_val_losses.append(val_losses)
        best_epochs.append(best_epoch)
        
        # Generate predictions
        y_pred_raw, probas_1 = predict_transformer(classifier, X_test, device=device)
        y_pred = y_pred_raw.astype('int8')
        probas_ = np.vstack([1-probas_1, probas_1]).T  # Format probabilities as [prob_class0, prob_class1]

        # Compute confusion matrix and extract metrics
        cm = confusion_matrix(y_test, y_pred)
        TN = cm[0, 0]  # True negatives
        FP = cm[0, 1]  # False positives
        FN = cm[1, 0]  # False negatives
        TP = cm[1, 1]  # True positives
        P = TP + FN  # Total positives
        N = TN + FP  # Total negatives

        # Calculate performance metrics
        accuracy = (TP + TN) / (P + N)
        sensitivity = TP / P if P > 0 else 0  # Recall
        specificity = TN / N if N > 0 else 0
        ppv = TP / (TP + FP) if (TP + FP) > 0 else 0  # Precision
        npv = TN / (TN + FN) if (TN + FN) > 0 else 0
        f1 = f1_score(y_test, y_pred)

        # Accumulate metrics for averaging later
        accuracy_sum += accuracy
        sensitivity_sum += sensitivity
        specificity_sum += specificity
        ppv_sum += ppv
        npv_sum += npv
        f1_sum += f1

        # Compute ROC curve and find the best threshold
        fpr, tpr, roc_thresholds = roc_curve(y_test, probas_[:, 1])
        gmeans = tpr * (1 - fpr)  # Geometric mean of sensitivity and specificity
        ix1 = np.argmax(gmeans)  # Index of the best threshold
        best_threshold_roc += roc_thresholds[ix1]

        # Compute Precision-Recall curve and find the best threshold
        precision, recall, prc_thresholds = precision_recall_curve(y_test, probas_[:, 1])
        fscore = (2 * precision * recall) / (precision + recall + 1e-10)  # F1 score for PRC, avoid div by zero
        ix2 = np.argmax(fscore)  # Index of the best threshold
        if ix2 < len(prc_thresholds):
            best_threshold_prc += prc_thresholds[ix2]

        # Interpolate ROC and PRC curves for averaging
        tprs.append(np.interp(mean_fpr1, fpr, tpr))
        precisions.append(np.interp(mean_fpr2, recall, precision))
        y_real.append(y_test)
        y_proba.append(probas_[:, 1])
        tprs[-1][0] = 0.0  # Ensure the first TPR value is 0
        precisions[-1][0] = 1.0  # Ensure the first precision value is 1

        # Calculate AUC for ROC and PRC
        roc_auc = auc(fpr, tpr)
        prc_auc = auc(recall, precision)
        aucs1.append(roc_auc)
        aucs2.append(prc_auc)

        # Save fold-specific results to a JSON file
        fold_results = {
            "accuracy": accuracy,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "ppv": ppv,
            "npv": npv,
            "f1": f1,
            "roc_auc": roc_auc,
            "prc_auc": prc_auc,
            "best_threshold_roc": roc_thresholds[ix1],
            "best_threshold_prc": prc_thresholds[ix2] if ix2 < len(prc_thresholds) else 0.5,
            "best_epoch": best_epoch,
            "train_losses": train_losses,
            "val_losses": val_losses
        }

        # Convert fold_results values to Python native types for JSON serialization
        fold_results = {key: (value.item() if isinstance(value, np.generic) else value) for key, value in fold_results.items()}
        with open(f"experiments/{subfolder_name}/fold_{i + 1}.json", "w") as f:
            json.dump(fold_results, f, indent=4)

        # Save model
        torch.save(classifier.state_dict(), f"experiments/{subfolder_name}/fold_{i + 1}_model.pt")
        
        # Save scaler
        with open(f"experiments/{subfolder_name}/fold_{i + 1}_scaler.pkl", 'wb') as f:
            pickle.dump(scaler, f)

        print(f"Fold {i + 1} completed in {time.time() - t0:.2f} seconds.")
        print(f"Accuracy: {accuracy:.4f}, Sensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}")
        print(f"PPV: {ppv:.4f}, NPV: {npv:.4f}, F1: {f1:.4f}")
        print(f"ROC AUC: {roc_auc:.4f}, PRC AUC: {prc_auc:.4f}")
        print(f"Best ROC Threshold: {roc_thresholds[ix1]:.4f}, Best PRC Threshold: {fold_results['best_threshold_prc']:.4f}")
        print(f"Best epoch: {best_epoch} of {len(train_losses)}")

    # Plot learning curves across all folds
    plt.figure(figsize=(12, 8))
    for i, (train_losses, val_losses) in enumerate(zip(all_train_losses, all_val_losses)):
        max_epoch = len(train_losses)
        epochs_range = list(range(1, max_epoch + 1))
        plt.plot(epochs_range, train_losses, 'b-', alpha=0.3, label=f'Train Fold {i+1}' if i == 0 else None)
        plt.plot(epochs_range, val_losses, 'r-', alpha=0.3, label=f'Val Fold {i+1}' if i == 0 else None)
    
    # Plot mean learning curves
    max_len = min([len(losses) for losses in all_train_losses])
    mean_train = np.mean([losses[:max_len] for losses in all_train_losses], axis=0)
    mean_val = np.mean([losses[:max_len] for losses in all_val_losses], axis=0)
    plt.plot(range(1, max_len + 1), mean_train, 'b-', linewidth=2, label='Mean Training Loss')
    plt.plot(range(1, max_len + 1), mean_val, 'r-', linewidth=2, label='Mean Validation Loss')
    
    plt.title('Learning Curves Across All Folds')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"experiments/{subfolder_name}/learning_curves.png", dpi=300)

    # Print aggregated results over all folds
    num_folds = len(aucs1)
    print("\nAggregated Results Over All Folds:")
    print(f"Mean Accuracy: {accuracy_sum / num_folds:.4f}")
    print(f"Mean Sensitivity: {sensitivity_sum / num_folds:.4f}")
    print(f"Mean Specificity: {specificity_sum / num_folds:.4f}")
    print(f"Mean PPV: {ppv_sum / num_folds:.4f}")
    print(f"Mean NPV: {npv_sum / num_folds:.4f}")
    print(f"Mean F1 Score: {f1_sum / num_folds:.4f}")
    print(f"Mean ROC Best Threshold: {best_threshold_roc / num_folds:.4f}")
    print(f"Mean PRC Best Threshold: {best_threshold_prc / num_folds:.4f}")
    print(f"Mean ROC AUC: {np.mean(aucs1):.4f}")
    print(f"Mean PRC AUC: {np.mean(aucs2):.4f}")
    print(f"Mean Best Epoch: {np.mean(best_epochs):.1f}")
    
    # Save aggregated results
    aggregated_results = {
        "mean_accuracy": accuracy_sum / num_folds,
        "mean_sensitivity": sensitivity_sum / num_folds,
        "mean_specificity": specificity_sum / num_folds,
        "mean_ppv": ppv_sum / num_folds,
        "mean_npv": npv_sum / num_folds,
        "mean_f1": f1_sum / num_folds,
        "mean_roc_auc": np.mean(aucs1).item(),
        "mean_prc_auc": np.mean(aucs2).item(),
        "mean_best_threshold_roc": (best_threshold_roc / num_folds).item(),
        "mean_best_threshold_prc": (best_threshold_prc / num_folds).item(),
        "mean_best_epoch": np.mean(best_epochs).item(),
        "individual_fold_roc_aucs": [auc.item() for auc in aucs1],
        "individual_fold_prc_aucs": [auc.item() for auc in aucs2]
    }
    
    with open(f"experiments/{subfolder_name}/aggregated_results.json", "w") as f:
        json.dump(aggregated_results, f, indent=4)


# --------------------------------------------------------
# -*- train model on entire dataset and save feature importances -*-
# --------------------------------------------------------

print("Training model on entire dataset...")
t0 = time.time()

# Convert to float for training
X_full = features_used.astype(float)
y_full = attributes_used.astype('int8')

# Handle NaN values in the baseline_creatinine_points column
if np.isnan(X_full).any():
    print("Found NaN values, filling with median values...")
    # Get column indices with NaN values
    nan_cols = np.where(np.isnan(X_full).any(axis=0))[0]
    
    for col in nan_cols:
        # Calculate median for each column (excluding NaN values)
        col_median = np.nanmedian(X_full[:, col])
        # Fill NaN values with median
        X_full[:, col] = np.nan_to_num(X_full[:, col], nan=col_median)


# Standardize features
scaler_full = StandardScaler()
X_full_scaled = scaler_full.fit_transform(X_full)

# Split into train and validation for early stopping
X_train_full, X_val_full, y_train_full, y_val_full = train_test_split(
    X_full_scaled, y_full, 
    test_size=args.validation_split, 
    random_state=1202, 
    stratify=y_full
)

# Train the full model with early stopping
print(f"Training transformer model on {X_train_full.shape[0]} samples with validation set of {X_val_full.shape[0]} samples...")
full_model, train_losses, val_losses, best_epoch = train_with_validation(
    X_full_scaled, y_full,
    device=device,
    epochs=args.epochs, 
    batch_size=args.batch_size,
    validation_split=args.validation_split,
    early_stopping=args.early_stopping,
    learning_rate=args.learning_rate
)

# Save the model and scaler
torch.save(full_model.state_dict(), f"experiments/{subfolder_name}/full_model.pt")
with open(f"experiments/{subfolder_name}/full_model_scaler.pkl", 'wb') as f:
    pickle.dump(scaler_full, f)

# Plot learning curves
plt.figure(figsize=(12, 6))
plt.plot(range(1, len(train_losses) + 1), train_losses, 'b-', label='Training Loss')
plt.plot(range(1, len(val_losses) + 1), val_losses, 'r-', label='Validation Loss')
plt.axvline(x=best_epoch + 1, color='g', linestyle='--', label=f'Best Epoch ({best_epoch + 1})')
plt.title('Learning Curves for Full Model')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig(f"experiments/{subfolder_name}/full_model_learning_curves.png", dpi=300)

# Save training information
full_model_info = {
    "train_losses": train_losses,
    "val_losses": val_losses,
    "best_epoch": best_epoch,
    "feature_count": X_full.shape[1],
    "sample_count": X_full.shape[0],
    "training_time": time.time() - t0,
    "feature_names": feature_names.tolist()
}

with open(f"experiments/{subfolder_name}/full_model_info.json", 'w') as f:
    json.dump(full_model_info, f, indent=4)


# Save the args dictionary for easy replicability
args_dict = {}
for arg in dir(args):
	if not arg.startswith('_') and not callable(getattr(args, arg)):
		args_dict[arg] = getattr(args, arg)

with open(f"experiments/{subfolder_name}/args.json", 'w') as f:
	json.dump(args_dict, f, indent=4)


print(f"Full model training completed in {time.time() - t0:.2f} seconds.")
print(f"Best epoch: {best_epoch + 1}")
print(f"Model saved to experiments/{subfolder_name}/")


# --------------------------------------------------------
# -*- perform SHAP analysis on the full model -*-
# --------------------------------------------------------

if args.perform_shapanalysis:
    print("Performing SHAP analysis on the transformer model...")
    
    # Create a background dataset for SHAP
    background = X_full_scaled[:100]  # Use a subset for efficiency
    
    # Define a prediction function for the transformer model
    def model_predict(X):
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        full_model.eval()
        with torch.no_grad():
            outputs = full_model(X_tensor)
            probs = torch.softmax(outputs, dim=1)
            return probs.cpu().numpy()
    
    # Initialize the SHAP explainer
    try:
        # Try to use DeepExplainer for neural networks
        background_tensor = torch.tensor(background, dtype=torch.float32).to(device)
        explainer = shap.DeepExplainer(full_model, background_tensor)
        
        # Sample data to explain
        sample_to_explain = X_full_scaled[:200]  # Limit to 200 samples for efficiency
        sample_tensor = torch.tensor(sample_to_explain, dtype=torch.float32).to(device)
        
        # Compute SHAP values
        shap_values = explainer.shap_values(sample_tensor)
        
        # If shap_values is a list (one element per class), take the positive class (index 1)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        # Create a SHAP summary plot
        plt.figure(figsize=(12, 10))
        shap.summary_plot(shap_values, sample_to_explain, feature_names=list(feature_names), show=False)
        plt.tight_layout()
        plt.savefig(f"experiments/{subfolder_name}/shap_summary.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create a SHAP bar plot of feature importance
        plt.figure(figsize=(12, 10))
        shap.summary_plot(shap_values, sample_to_explain, feature_names=list(feature_names), plot_type='bar', show=False)
        plt.tight_layout()
        plt.savefig(f"experiments/{subfolder_name}/shap_importance.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"DeepExplainer failed: {str(e)}")
        print("Falling back to KernelExplainer...")
        
        # Fallback to KernelExplainer which works with any model
        explainer = shap.KernelExplainer(model_predict, background)
        sample_to_explain = X_full_scaled[:100]  # Use fewer samples for KernelExplainer
        shap_values = explainer.shap_values(sample_to_explain)
        
        # Create SHAP plots
        plt.figure(figsize=(12, 10))
        shap.summary_plot(shap_values[1], sample_to_explain, feature_names=list(feature_names), show=False)
        plt.tight_layout()
        plt.savefig(f"experiments/{subfolder_name}/shap_summary.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        plt.figure(figsize=(12, 10))
        shap.summary_plot(shap_values[1], sample_to_explain, feature_names=list(feature_names), plot_type='bar', show=False)
        plt.tight_layout()
        plt.savefig(f"experiments/{subfolder_name}/shap_importance.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"SHAP analysis completed and plots saved to experiments/{subfolder_name}/")

print("Script execution completed successfully!")