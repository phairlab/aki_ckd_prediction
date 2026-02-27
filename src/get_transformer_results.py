"""
Transformer Model Training and Evaluation Script for AKI-CKD Prediction

This script trains transformer models for predicting chronic kidney disease (CKD) outcomes
following acute kidney injury (AKI). It includes:

- Cross-validation with model saving for each fold
- Feature selection using SelectKBest with mutual information
- Feature importance quantification using permutation importance
- Comprehensive performance metrics and visualizations
- Model loading from previous experiments for inference

For each fold in cross-validation, the following files are saved:
- fold_X_model.pt: Trained transformer model weights
- fold_X_scaler.pkl: StandardScaler used for feature normalization
- fold_X_selector.pkl: Feature selector (if feature selection is enabled)
- fold_X.json: Performance metrics for the fold
- fold_X_predictions.json: Test set predictions (y_true, y_pred, y_proba, test_indices)
- fold_X_selected_features.json: List of selected features and selection scores
- fold_X_feature_importance.json: Feature importance scores from permutation importance
- fold_X_feature_importance.txt: Human-readable feature importance rankings

The full model trained on the entire dataset is also saved along with SHAP analysis results.
"""

import os
import re
import random
import math
import time
import json
import pickle
import argparse
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.inspection import permutation_importance
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
from shap_analysis import perform_shap_analysis_kernel, regenerate_shap_plots_kernel, load_shap_values
import traceback


random.seed(1202)
np.random.seed(1202) 
torch.manual_seed(1202)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1202)


def fetch_args():
    # Development dictionary - comment/uncomment this section as needed
    args_dict = {
        'hing_features': True,
        'alberta_features_points': False,
        'alberta_features_raw': False,
        'alberta_score': False,
        'eGFR_features': False,
        'target': 'ckd',  # 'ckd' or 'ckdordeath'
        'perform_feature_selection': True,  # Perform feature selection using SelectKBest
        'n_features_to_select': 100,  # Number of features to select (only used if perform_feature_selection=True)
        'perform_cv': True,
        'perform_shapanalysis': True,
        'modelsize': 'small',  # 'small' or 'large' - determines model architecture
        'epochs': 100,            # Increased max epochs
        'batch_size': 32,        # Batch size for training
        'learning_rate': 5e-5,   # Reduced learning rate for slower, more stable convergence
        'early_stopping': 10,    # Increased patience - wait longer before stopping
        'validation_split': 0.15, # Portion of training data to use for validation
        'load_from_experiment': "/data/kidney/Sacha/aki_ckd_prediction/experiments/20251219_smalltransformer_hing_fselect100_fold_results",  # Path to previous experiment folder to load models from (e.g., "experiments/20250711_transformer_hing__fold_results")
        'regenerate_shap_plots': None,  # Path to experiment folder to load SHAP values and regenerate plots (e.g., "experiments/20251219_smalltransformer_hing_fselect100_fold_results")
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

# If regenerating SHAP plots from a previous experiment, handle that and exit
if args.regenerate_shap_plots:
    print(f"Regenerating SHAP plots from experiment: {args.regenerate_shap_plots}")
    
    # Load original args
    original_args_path = os.path.join(args.regenerate_shap_plots, "args.json")
    if os.path.exists(original_args_path):
        print(f"Loading original experiment args from {original_args_path}")
        with open(original_args_path, 'r') as f:
            original_args_dict = json.load(f)
        print(f"Original experiment configuration:")
        for key, value in original_args_dict.items():
            print(f"  - {key}: {value}")
    else:
        print(f"Warning: Could not find args.json at {original_args_path}")
    
    # Load SHAP values
    shap_values_path = os.path.join(args.regenerate_shap_plots, "shap_values.pkl")
    if os.path.exists(shap_values_path):
        print(f"Loading SHAP values from {shap_values_path}")
        shap_values = load_shap_values(shap_values_path)
        
        # Try to load X_data that was explained
        X_data_path = os.path.join(args.regenerate_shap_plots, "shap_X_data.pkl")
        if os.path.exists(X_data_path):
            print(f"Loading X_data from {X_data_path}")
            with open(X_data_path, 'rb') as f:
                X_data = pickle.load(f)
        else:
            print("Warning: shap_X_data.pkl not found, using zeros (plots may not show feature values correctly)")
            n_samples = len(shap_values[0])
            n_features = len(shap_values[0][0]) if n_samples > 0 else 0
            X_data = np.zeros((n_samples, n_features))
        
        # Try to load feature names
        feature_names_path = os.path.join(args.regenerate_shap_plots, "shap_feature_names.pkl")
        if os.path.exists(feature_names_path):
            print(f"Loading feature names from {feature_names_path}")
            with open(feature_names_path, 'rb') as f:
                feature_names_loaded = pickle.load(f)
        else:
            # Fallback to full_model_info.json
            model_info_path = os.path.join(args.regenerate_shap_plots, "full_model_info.json")
            if os.path.exists(model_info_path):
                with open(model_info_path, 'r') as f:
                    model_info = json.load(f)
                feature_names_loaded = model_info.get('feature_names', None)
                if feature_names_loaded is None:
                    print("Warning: feature_names not found, using generic names")
                    feature_names_loaded = [f"feature_{i}" for i in range(len(shap_values[0][0]))]
            else:
                print("Warning: Could not find feature names, using generic names")
                feature_names_loaded = [f"feature_{i}" for i in range(len(shap_values[0][0]))]
        
        # Regenerate the plots
        regenerate_shap_plots_kernel(
            shap_values=shap_values,
            X_data=X_data,
            feature_names=feature_names_loaded,
            output_dir=args.regenerate_shap_plots,
            max_display=20
        )
        print("SHAP plot regeneration complete!")
    else:
        print(f"Error: Could not find shap_values.pkl at {shap_values_path}")
        print("Make sure the experiment was run with perform_shapanalysis=True")
    
    # Exit after regenerating plots
    exit(0)

# If loading from a previous experiment, override args with the original experiment's args
if args.load_from_experiment:
    original_args_path = os.path.join(args.load_from_experiment, "args.json")
    if os.path.exists(original_args_path):
        print(f"Loading original experiment args from {original_args_path}")
        with open(original_args_path, 'r') as f:
            original_args_dict = json.load(f)
        
        # Preserve the load_from_experiment setting and any inference-specific settings
        load_from_experiment = args.load_from_experiment
        perform_cv = getattr(args, 'perform_cv', True)
        perform_shapanalysis = getattr(args, 'perform_shapanalysis', False)
        
        # Update args with original experiment settings
        for key, value in original_args_dict.items():
            setattr(args, key, value)
        
        # Restore inference-specific settings
        args.load_from_experiment = load_from_experiment
        args.perform_cv = perform_cv
        args.perform_shapanalysis = perform_shapanalysis
        
        print(f"Using original experiment configuration:")
        print(f"  - hing_features: {args.hing_features}")
        print(f"  - alberta_features_points: {args.alberta_features_points}")
        print(f"  - alberta_features_raw: {args.alberta_features_raw}")
        print(f"  - alberta_score: {args.alberta_score}")
        print(f"  - target: {args.target}")
    else:
        print(f"Warning: Could not find args.json at {original_args_path}")
        print("Using current args configuration")


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

features_used, attributes_used, feature_names, features_df, eGFR_features = preprocess_data(hing_features=args.hing_features, 
                                                                                            alberta_features_points=args.alberta_features_points, 
                                                                                            alberta_features_raw=args.alberta_features_raw,
                                                                                            alberta_score=args.alberta_score,
                                                                                            eGFR_features=args.eGFR_features,
                                                                                            target=args.target,
                                                                                            model_type="transformer")


# print(eGFR_features)
# print(eGFR_features.shape)

print(attributes_used)


# --------------------------------------------------------
# -*- run training loop and get results -*-
# --------------------------------------------------------

# Define the date and feature type for the subfolder name
current_date = datetime.now().strftime("%Y%m%d")
features_list = []

if args.hing_features: features_list.append("hing")

if args.alberta_features_raw: features_list.append("abraw")
elif args.alberta_features_points: features_list.append("abpoints")

if args.alberta_score: features_list.append("abscore")

if args.eGFR_features: features_list.append("egfr")

extra = f"fselect{args.n_features_to_select}" if args.perform_feature_selection and args.hing_features else ""

features_string = "-".join(features_list)

# Modify subfolder name based on whether we're loading or training
if args.load_from_experiment:
    # Extract original experiment name and add inference suffix
    original_name = os.path.basename(args.load_from_experiment.rstrip('/'))
    subfolder_name = f"{current_date}_inference_{original_name}"
else:
    subfolder_name = f"{current_date}_{args.modelsize}transformer_{features_string}_{extra}_fold_results"

# Change the directory to /data/kidney/Sacha/aki_ckd_prediction/
os.chdir("/data/kidney/Sacha/aki_ckd_prediction")

# Create the experiments folder and subfolder for this run
os.makedirs(f"experiments/{subfolder_name}", exist_ok=True)

# Print mode information
if args.load_from_experiment:
    print(f"INFERENCE MODE: Loading models from {args.load_from_experiment}")
    print(f"Results will be saved to: experiments/{subfolder_name}")
else:
    print(f"TRAINING MODE: Training new models")
    print(f"Results will be saved to: experiments/{subfolder_name}")

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

if args.eGFR_features:  # overrides the other features
    features_used = eGFR_features
    feature_names = [str(i) for i in range(99)]

if args.perform_cv:
    # Perform cross-validation
    for i, (train, test) in enumerate(cv.split(features_used, attributes_used)):
        if i < skip:  # Skip the first skip folds (for if training gets halted)
            continue
        
        t0 = time.time()
        print(f"Processing fold {i + 1}...")

        X_train, y_train = features_used[train].astype(float), attributes_used[train].astype('int8')
        X_test, y_test = features_used[test].astype(float), attributes_used[test].astype('int8')

        # Feature normalization is crucial for transformer models
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Feature selection using SelectKBest with mutual information
        feature_names_fold = feature_names
        selector = None
        
        if args.perform_feature_selection and args.hing_features:
            print(f"Performing feature selection: selecting {args.n_features_to_select} features...")
            selector = SelectKBest(score_func=mutual_info_classif, k=args.n_features_to_select)
            X_train = selector.fit_transform(X_train, y_train)
            X_test = selector.transform(X_test)
            feature_names_fold = feature_names[selector.get_support()]
            print(f"Selected {len(feature_names_fold)} features for this fold")

        if args.load_from_experiment:
            # Load pre-trained model and scaler instead of training
            print(f"Loading pre-trained model from {args.load_from_experiment}/fold_{i + 1}_model.pt")
            
            # Load the saved scaler
            scaler_path = f"{args.load_from_experiment}/fold_{i + 1}_scaler.pkl"
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    scaler = pickle.load(f)
                print(f"Loaded scaler from {scaler_path}")
                
                # Re-transform the data with the loaded scaler
                X_test = scaler.transform(features_used[test].astype(float))
            else:
                print(f"Warning: Scaler not found at {scaler_path}, using current scaler")
            
            # Load the feature selector if it exists
            selector_path = f"{args.load_from_experiment}/fold_{i + 1}_selector.pkl"
            if os.path.exists(selector_path):
                with open(selector_path, 'rb') as f:
                    selector = pickle.load(f)
                X_test = selector.transform(X_test)
                feature_names_fold = feature_names[selector.get_support()]
                print(f"Loaded feature selector from {selector_path}")
            else:
                print(f"No feature selector found at {selector_path}, using all features")
                feature_names_fold = feature_names
            
            # Load the model
            model_path = f"{args.load_from_experiment}/fold_{i + 1}_model.pt"
            if not os.path.exists(model_path):
                print(f"Error: Model not found at {model_path}")
                continue
                
            # Create model instance with same architecture
            input_size = X_test.shape[1]
            
            # Check if args specify model size, default to 'small' if not specified
            model_size = getattr(args, 'modelsize', 'small')
            if model_size == 'large':
                classifier = LargeTabularTransformer(input_dim=input_size).to(device)
            else:
                classifier = TabularTransformer(input_dim=input_size).to(device)
            
            # Load the saved weights
            classifier.load_state_dict(torch.load(model_path, map_location=device))
            classifier.eval()
            
            print(f"Loaded model from {model_path}")
            
            # Set dummy values for training metrics since we didn't train
            train_losses = []
            val_losses = []
            best_epoch = 0
            
        else:
            # Train model with validation-based early stopping (original behavior)
            print(f"Training fold {i + 1}...")
            classifier, train_losses, val_losses, best_epoch = train_with_validation(
                X_train, y_train, 
                device=device,
                epochs=args.epochs,
                batch_size=args.batch_size,
                validation_split=args.validation_split,
                early_stopping=args.early_stopping,
                learning_rate=args.learning_rate,
                model_size=args.modelsize #"large" if args.hing_features else "small" 
            )
        
        # Store training metrics
        all_train_losses.append(train_losses)
        all_val_losses.append(val_losses)
        best_epochs.append(best_epoch)
        
        # Generate predictions
        if args.load_from_experiment:
            # Generate predictions manually since we loaded the model and scaler separately
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
            
            with torch.no_grad():
                outputs = classifier(X_test_tensor)
                probs = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(probs, dim=1)
            
            y_pred_raw = predictions.cpu().numpy()
            probas_1 = probs[:, 1].cpu().numpy()
        else:
            # Use the existing prediction function for newly trained models
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

        # Save model and scaler only if we trained a new model (not loading from previous experiment)
        if not args.load_from_experiment:
            # Save model
            torch.save(classifier.state_dict(), f"experiments/{subfolder_name}/fold_{i + 1}_model.pt")
            
            # Save scaler
            with open(f"experiments/{subfolder_name}/fold_{i + 1}_scaler.pkl", 'wb') as f:
                pickle.dump(scaler, f)
            
            # Save feature selector if feature selection was performed
            if selector is not None:
                with open(f"experiments/{subfolder_name}/fold_{i + 1}_selector.pkl", 'wb') as f:
                    pickle.dump(selector, f)
        
        # Calculate and save feature importance for this fold
        if not args.load_from_experiment or selector is not None:  # Only if we have features to analyze
            print(f"Calculating feature importance for fold {i + 1}...")
            
            # Manual permutation importance calculation (more reliable for transformers)
            try:
                # Get baseline accuracy
                X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
                classifier.eval()
                with torch.no_grad():
                    outputs = classifier(X_test_tensor)
                    baseline_predictions = torch.argmax(outputs, dim=1).cpu().numpy()
                baseline_accuracy = accuracy_score(y_test, baseline_predictions)
                
                # Calculate importance for each feature
                importances = []
                n_repeats = 5
                
                for feat_idx in range(X_test.shape[1]):
                    feature_importances = []
                    
                    for repeat in range(n_repeats):
                        # Make a copy and shuffle this feature
                        X_test_permuted = X_test.copy()
                        np.random.shuffle(X_test_permuted[:, feat_idx])
                        
                        # Get predictions with shuffled feature
                        X_permuted_tensor = torch.tensor(X_test_permuted, dtype=torch.float32).to(device)
                        with torch.no_grad():
                            outputs = classifier(X_permuted_tensor)
                            permuted_predictions = torch.argmax(outputs, dim=1).cpu().numpy()
                        
                        # Calculate accuracy drop
                        permuted_accuracy = accuracy_score(y_test, permuted_predictions)
                        importance = baseline_accuracy - permuted_accuracy
                        feature_importances.append(importance)
                    
                    # Store mean and std for this feature
                    importances.append({
                        'mean': np.mean(feature_importances),
                        'std': np.std(feature_importances)
                    })
                
                # Extract means and stds
                importance_means = np.array([imp['mean'] for imp in importances])
                importance_stds = np.array([imp['std'] for imp in importances])
                
                # Save feature importance results
                feature_importance_data = {
                    'feature_names': feature_names_fold.tolist(),
                    'importance_mean': importance_means.tolist(),
                    'importance_std': importance_stds.tolist(),
                    'baseline_accuracy': baseline_accuracy,
                    'n_features_selected': len(feature_names_fold),
                    'selection_method': 'SelectKBest_mutual_info' if selector is not None else 'all_features'
                }
                
                # Sort by importance for easier interpretation
                sorted_idx = np.argsort(importance_means)[::-1]
                
                # Save detailed feature importance file
                with open(f"experiments/{subfolder_name}/fold_{i + 1}_feature_importance.txt", 'w') as f:
                    f.write("Rank\tFeature\tImportance_Mean\tImportance_Std\n")
                    for rank, idx in enumerate(sorted_idx):
                        f.write(f"{rank+1}\t{feature_names_fold[idx]}\t{importance_means[idx]:.6f}\t{importance_stds[idx]:.6f}\n")
                
                # Save JSON version for easy loading
                with open(f"experiments/{subfolder_name}/fold_{i + 1}_feature_importance.json", 'w') as f:
                    json.dump(feature_importance_data, f, indent=4)
                    
                print(f"Feature importance calculated and saved for fold {i + 1}")
                
            except Exception as e:
                print(f"Warning: Could not calculate feature importance for fold {i + 1}: {e}")

        # Convert feature_names_fold to list if it's a numpy array
        if isinstance(feature_names_fold, np.ndarray):
            feature_names_fold = feature_names_fold.tolist()

        # Save selected features list for this fold
        selected_features = {
            'selected_feature_names': feature_names_fold,  # Removed .tolist()
            'n_features_selected': len(feature_names_fold),
            'selection_method': 'SelectKBest_mutual_info' if selector is not None else 'all_features',
            'selection_scores': selector.scores_.tolist() if selector is not None else None
        }
        with open(f"experiments/{subfolder_name}/fold_{i + 1}_selected_features.json", 'w') as f:
            json.dump(selected_features, f, indent=4)

        # Save test set predictions for this fold
        test_predictions = {
            'y_true': y_test.tolist(),
            'y_pred': y_pred.tolist(),
            'y_proba': probas_[:, 1].tolist(),  # Probability of positive class
            'test_indices': test.tolist()  # Original indices of test samples
        }
        with open(f"experiments/{subfolder_name}/fold_{i + 1}_predictions.json", 'w') as f:
            json.dump(test_predictions, f, indent=4)

        print(f"Fold {i + 1} completed in {time.time() - t0:.2f} seconds.")
        print(f"Accuracy: {accuracy:.4f}, Sensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}")
        print(f"PPV: {ppv:.4f}, NPV: {npv:.4f}, F1: {f1:.4f}")
        print(f"ROC AUC: {roc_auc:.4f}, PRC AUC: {prc_auc:.4f}")
        print(f"Best ROC Threshold: {roc_thresholds[ix1]:.4f}, Best PRC Threshold: {fold_results['best_threshold_prc']:.4f}")
        print(f"Best epoch: {best_epoch} of {len(train_losses)}")

    # Plot learning curves across all folds (only if we actually trained models)
    if not args.load_from_experiment and all_train_losses and all_val_losses:
        plt.figure(figsize=(12, 8))
        for i, (train_losses, val_losses) in enumerate(zip(all_train_losses, all_val_losses)):
            if train_losses and val_losses:  # Check if losses exist
                max_epoch = len(train_losses)
                epochs_range = list(range(1, max_epoch + 1))
                plt.plot(epochs_range, train_losses, 'b-', alpha=0.3, label=f'Train Fold {i+1}' if i == 0 else None)
                plt.plot(epochs_range, val_losses, 'r-', alpha=0.3, label=f'Val Fold {i+1}' if i == 0 else None)
        
        # Plot mean learning curves
        valid_train_losses = [losses for losses in all_train_losses if losses]
        valid_val_losses = [losses for losses in all_val_losses if losses]
        
        if valid_train_losses and valid_val_losses:
            max_len = min([len(losses) for losses in valid_train_losses])
            mean_train = np.mean([losses[:max_len] for losses in valid_train_losses], axis=0)
            mean_val = np.mean([losses[:max_len] for losses in valid_val_losses], axis=0)
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
    
    # Only print mean best epoch if we actually trained models
    if not args.load_from_experiment and best_epochs and any(epoch > 0 for epoch in best_epochs):
        print(f"Mean Best Epoch: {np.mean([epoch for epoch in best_epochs if epoch > 0]):.1f}")
    
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
        "individual_fold_roc_aucs": [auc.item() for auc in aucs1],
        "individual_fold_prc_aucs": [auc.item() for auc in aucs2],
        "loaded_from_experiment": args.load_from_experiment
    }
    
    # Only add best epoch info if we actually trained models
    if not args.load_from_experiment and best_epochs and any(epoch > 0 for epoch in best_epochs):
        valid_epochs = [epoch for epoch in best_epochs if epoch > 0]
        aggregated_results["mean_best_epoch"] = np.mean(valid_epochs).item()
    
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
    
    # Define a prediction function for the transformer model
    def model_predict(X):
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        full_model.eval()
        with torch.no_grad():
            outputs = full_model(X_tensor)
            probs = torch.softmax(outputs, dim=1)
            return probs.cpu().numpy()
    
    # Use the shared SHAP analysis function
    shap_values = perform_shap_analysis_kernel(
        model_predict_fn=model_predict,
        X_data=X_full_scaled,
        feature_names=feature_names,
        output_dir=f"experiments/{subfolder_name}",
        background_samples=100,
        samples_to_explain=100,
        max_display=20,
        verbose=True
    )

print("Script execution completed successfully!")