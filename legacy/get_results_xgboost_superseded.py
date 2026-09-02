"""
SUPERSEDED -- pre-refactor duplicate, kept only for provenance.

This module predates the refactor into config.py / run_pipeline.py / src/{models,
analysis,evaluation}. It is not on the analysis path and is not maintained. The
live equivalents are:

    get_results_xgboost.py        -> src/cross_validation.py + src/models/xgboost_model.py
    get_transformer_results.py    -> src/cross_validation.py + src/models/transformer_*.py
    transformer_model_helpers.py  -> src/models/transformer_model.py
    shap_analysis.py              -> src/analysis/shap_analysis.py (plotting) and
                                     src/cross_validation.py::_fold_shap (out-of-fold values)

Do not use for the resubmission: none of these carry the fold-local imputation,
the nested hyperparameter search, the stratified split, or the out-of-fold SHAP.
"""

"""
XGBoost Model Training and Evaluation Script for AKI-CKD Prediction

This script trains XGBoost models for predicting chronic kidney disease (CKD) outcomes
following acute kidney injury (AKI). It includes:

- Cross-validation with model saving for each fold
- Feature selection using Recursive Feature Elimination (RFE)
- SHAP analysis for model interpretability
- Comprehensive performance metrics and visualizations

For each fold in cross-validation, the following files are saved:
- fold_X_model.pkl: Trained XGBoost model
- fold_X_scaler.pkl: StandardScaler used for feature normalization
- fold_X_rfe.pkl: RFE object (if feature selection is enabled)
- fold_X.json: Performance metrics for the fold
- fold_X_feature_importances.txt: Feature importance rankings
- fold_X_predictions.json: Test set predictions (y_true, y_pred, y_proba, test_indices)

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
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.metrics import precision_recall_curve, average_precision_score

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn

from xgboost import XGBClassifier

# from alberta_score_helpers import *
from data_preprocessing import *
from transformer_model_helpers import *
from shap_analysis import perform_shap_analysis_tree, regenerate_shap_plots_tree, load_shap_values

random.seed(1202)
np.random.seed(1202) 


def fetch_args():
	# Development dictionary - comment/uncomment this section as needed
	args_dict = {
		'hing_features': False,
		'alberta_features_points': False,
		'alberta_features_raw': True,
		'alberta_score': False,
		'model_type': 'xgboost',  # 'xgboost' or 'logreg'
		'perform_cv': True,
		'perform_shapanalysis': True,
		'target': 'ckd',  # 'ckd' or 'ckdordeath
		'feature_selection': True, 
		'n_features_to_select': 100,  # Number of features to select with RFE
		'rfe_step': 0.1,  # Step size for RFE
		'random_state': 1202,  # Random seed for reproducibility
		'load_from_experiment': "/data/kidney/Sacha/aki_ckd_prediction/experiments/20250714_xgboost_abpoints__fold_results",

	}
	
	class Args:
		def __init__(self, args_dict):
			for k, v in args_dict.items():
				setattr(self, k, v)
	
	return Args(args_dict)
	
	# # Command line argument parsing
	# parser = argparse.ArgumentParser(description='Run AKI-CKD prediction model')
	# parser.add_argument('--hing_features', type=str, default='True', help='Include features engineered by Hing')
	# parser.add_argument('--alberta_features', type=str, default='False', help='Include points Alberta score component features')
	# parser.add_argument('--alberta_features_raw', type=str, default='False', help='Include raw Alberta score component features')
	# parser.add_argument('--alberta_score', type=str, default='False', help='Include Alberta score as a feature')
	# parser.add_argument('--model_type', type=str, default='xgboost', help='Model type: xgboost or logreg')
	# parser.add_argument('--perform_cv', type=str, default='True', help='Perform cross-validation step')
	# parser.add_argument('--perform_shapanalysis', type=str, default='True', help='Perform SHAP analysis on the final model')
	# parser.add_argument('--target', type=str, default='ckd', help='Target variable: ckd or ckdordeath')
	# parser.add_argument('--feature_selection', type=str, default='True', help='Perform feature selection using RFE')
	# parser.add_argument('--n_features_to_select', type=int, default=240, help='Number of features to select with RFE')
	# parser.add_argument('--rfe_step', type=float, default=0.1, help='Step size for RFE')
	# parser.add_argument('--random_state', type=int, default=1202, help='Random seed for reproducibility')
	
	# return parser.parse_args()

args = fetch_args()

# If loading from a previous experiment to regenerate plots, handle that separately
if args.load_from_experiment:
	print(f"Loading from experiment: {args.load_from_experiment}")
	
	# Load original args
	original_args_path = os.path.join(args.load_from_experiment, "args.json")
	if os.path.exists(original_args_path):
		print(f"Loading original experiment args from {original_args_path}")
		with open(original_args_path, 'r') as f:
			original_args_dict = json.load(f)
		print(f"Original experiment configuration:")
		for key, value in original_args_dict.items():
			print(f"  - {key}: {value}")
	else:
		print(f"Warning: Could not find args.json at {original_args_path}")
	
	# Load and regenerate SHAP plots
	shap_values_path = os.path.join(args.load_from_experiment, "shap_values.pkl")
	if os.path.exists(shap_values_path):
		print(f"Loading SHAP values from {shap_values_path}")
		shap_values = load_shap_values(shap_values_path)
		
		# Get feature names from the SHAP values object
		if hasattr(shap_values, 'feature_names') and shap_values.feature_names is not None:
			feature_names_loaded = shap_values.feature_names
		else:
			# Try to load from full_model_feature_importances.txt
			importances_path = os.path.join(args.load_from_experiment, "full_model_feature_importances.txt")
			if os.path.exists(importances_path):
				print(f"Loading feature names from {importances_path}")
				importances_df = pd.read_csv(importances_path, sep='\t')
				feature_names_loaded = importances_df['Feature'].tolist()
			else:
				print("Warning: Could not determine feature names, using generic names")
				feature_names_loaded = [f"feature_{i}" for i in range(shap_values.shape[1])]
		
		# Regenerate the plots
		regenerate_shap_plots_tree(
			shap_values=shap_values,
			feature_names=feature_names_loaded,
			output_dir=args.load_from_experiment,
			max_display=20
		)
		print("SHAP plot regeneration complete!")
	else:
		print(f"Error: Could not find shap_values.pkl at {shap_values_path}")
		print("Make sure the experiment was run with perform_shapanalysis=True")
	
	# Exit after regenerating plots
	exit(0)

# # Convert string boolean arguments to actual booleans - this is so silly
# args.hing_features = args.hing_features.lower() == 'true'
# args.alberta_features = args.alberta_features.lower() == 'true'
# args.alberta_features_raw = args.alberta_features_raw.lower() == 'true'
# args.alberta_score = args.alberta_score.lower() == 'true'
# args.perform_cv = args.perform_cv.lower() == 'true'
# args.perform_shapanalysis = args.perform_shapanalysis.lower() == 'true'
# args.feature_selection = args.feature_selection.lower() == 'true'

# --------------------------------------------------------
# -*- get preprocessed/cleaned dataset -*-
# --------------------------------------------------------

features_used, attributes_used, feature_names, features_df, _ = preprocess_data(hing_features=args.hing_features, 
                                                                             alberta_features_points=args.alberta_features_points, 
                                                                             alberta_features_raw=args.alberta_features_raw,
                                                                             alberta_score=args.alberta_score,
																			 eGFR_features=False,  # not using eGFR features for XGBoost model
                                                                             target=args.target,
																			 model_type=args.model_type)


print(feature_names)

# --------------------------------------------------------
# -*- run training loop and get results -*-

# Define the date and feature type for the subfolder name
current_date = datetime.now().strftime("%Y%m%d")
features_list = []
if args.hing_features: features_list.append("hing")
if args.alberta_features_points: features_list.append("abpoints")
if args.alberta_features_raw: features_list.append("abpointsraw")
if args.alberta_score: features_list.append("abscore")
extra = f"fselect{args.n_features_to_select}" if args.feature_selection and args.hing_features else ""


features_string = "-".join(features_list)
subfolder_name = f"{current_date}_{args.model_type}_{features_string}_{extra}_fold_results"

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
# plt.clf()  # Clear any existing plots
best_threshold_roc = 0  # Sum of best thresholds for ROC curve
best_threshold_prc = 0  # Sum of best thresholds for PRC curve

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

		# Feature normalization is crucial for transformer models
		scaler = StandardScaler()
		X_train = scaler.fit_transform(X_train)
		X_test = scaler.transform(X_test)

		feature_names_fold = feature_names

		if args.feature_selection and args.hing_features:  # perform RFE on the training set
			classifier = XGBClassifier(random_state=1202)
			rfe = RFE(estimator=classifier, n_features_to_select=args.n_features_to_select, step=0.1, verbose=1)  # 381 seconds
			rfe = rfe.fit(X_train, y_train) 

			X_train = rfe.transform(X_train)
			X_test = rfe.transform(X_test)

			feature_names_fold = feature_names[rfe.support_]

		# Your existing XGBoost code
		classifier = XGBClassifier(random_state=1202)
		classifier.fit(X_train, y_train)
		y_pred = classifier.predict(X_test).astype('int8')
		probas_ = classifier.predict_proba(X_test)

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
		sensitivity = TP / P  # Recall
		specificity = TN / N
		ppv = TP / (TP + FP)  # Precision
		npv = TN / (TN + FN)
		f1 = f1_score(y_test, y_pred)

		# Accumulate metrics for averaging later
		accuracy_sum += accuracy
		sensitivity_sum += sensitivity
		specificity_sum += specificity
		ppv_sum += ppv
		npv_sum += npv
		f1_sum += f1

		# Predict probabilities for ROC and PRC analysis
		probas_ = classifier.predict_proba(X_test)

		# Compute ROC curve and find the best threshold
		fpr, tpr, roc_thresholds = roc_curve(y_test, probas_[:, 1])
		gmeans = tpr * (1 - fpr)  # Geometric mean of sensitivity and specificity
		ix1 = np.argmax(gmeans)  # Index of the best threshold
		best_threshold_roc += roc_thresholds[ix1]

		# Compute Precision-Recall curve and find the best threshold
		precision, recall, prc_thresholds = precision_recall_curve(y_test, probas_[:, 1])
		fscore = (2 * precision * recall) / (precision + recall)  # F1 score for PRC
		ix2 = np.argmax(fscore)  # Index of the best threshold
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
			"best_threshold_prc": prc_thresholds[ix2]
		}

		# Convert fold_results values to Python native types for JSON serialization
		fold_results = {key: (value.item() if isinstance(value, np.generic) else value) for key, value in fold_results.items()}
		with open(f"experiments/{subfolder_name}/fold_{i + 1}.json", "w") as f:
			json.dump(fold_results, f, indent=4)

		# Save the trained model for this fold
		with open(f"experiments/{subfolder_name}/fold_{i + 1}_model.pkl", 'wb') as f:
			pickle.dump(classifier, f)

		# Save the scaler for this fold (needed for inference)
		with open(f"experiments/{subfolder_name}/fold_{i + 1}_scaler.pkl", 'wb') as f:
			pickle.dump(scaler, f)

		# If RFE was used, save the RFE object for this fold
		if args.feature_selection and args.hing_features:
			with open(f"experiments/{subfolder_name}/fold_{i + 1}_rfe.pkl", 'wb') as f:
				pickle.dump(rfe, f)

		# Save test set predictions for this fold
		test_predictions = {
			'y_true': y_test.tolist(),
			'y_pred': y_pred.tolist(),
			'y_proba': probas_[:, 1].tolist(),  # Probability of positive class
			'test_indices': test.tolist()  # Original indices of test samples
		}
		with open(f"experiments/{subfolder_name}/fold_{i + 1}_predictions.json", 'w') as f:
			json.dump(test_predictions, f, indent=4)

		# save feature importances to a file
		importances = classifier.feature_importances_
		indices = np.argsort(importances)[::-1]
		with open(f"experiments/{subfolder_name}/fold_{i + 1}_feature_importances.txt", 'w') as f:
			for j in range(X_train.shape[1]):
				f.write('%d\t%d\t%s\t%.6f\n' % (j, indices[j], feature_names_fold[indices[j]], importances[indices[j]]))


		print("Fold {} completed in {:.2f} seconds.".format(i + 1, time.time() - t0))
		# i += 1  # Increment fold counter

		# assert False


	# Print aggregated results over all folds
	print("Aggregated Results Over All Folds:")
	print(f"Mean Accuracy: {accuracy_sum / 10:.4f}")
	print(f"Mean Sensitivity: {sensitivity_sum / 10:.4f}")
	print(f"Mean Specificity: {specificity_sum / 10:.4f}")
	print(f"Mean PPV: {ppv_sum / 10:.4f}")
	print(f"Mean NPV: {npv_sum / 10:.4f}")
	print(f"Mean F1 Score: {f1_sum / 10:.4f}")
	print(f"Mean ROC Best Threshold: {best_threshold_roc / 10:.4f}")
	print(f"Mean PRC Best Threshold: {best_threshold_prc / 10:.4f}")
	print(f"Mean ROC AUC: {np.mean(aucs1):.4f}")
	print(f"Mean PRC AUC: {np.mean(aucs2):.4f}")


# --------------------------------------------------------
# -*- train model on entire dataset and save feature importances -*-

print("Training model on entire dataset...")
t0 = time.time()

# Convert to float for XGBoost
X_full = features_used.astype(float)
y_full = attributes_used.astype('int8')

# Initialize the classifier
full_classifier = XGBClassifier(random_state=1202)

# Apply RFE if using Hing features
if args.hing_features:
	print("Performing RFE on the entire dataset...")
	rfe_full = RFE(estimator=full_classifier, n_features_to_select=240, step=0.1, verbose=1)
	rfe_full.fit(X_full, y_full)
	X_full = rfe_full.transform(X_full)
	feature_names_full = feature_names[rfe_full.support_]
else:
	feature_names_full = feature_names

# Train the classifier on the entire dataset
full_classifier.fit(X_full, y_full)

# Save feature importances
importances = full_classifier.feature_importances_
indices = np.argsort(importances)[::-1]

with open(f"experiments/{subfolder_name}/full_model_feature_importances.txt", 'w') as f:
	f.write("Rank\tIndex\tFeature\tImportance\n")
	for j in range(X_full.shape[1]):
		f.write(f"{j+1}\t{indices[j]}\t{feature_names_full[indices[j]]}\t{importances[indices[j]]:.6f}\n")

# Save the model
with open(f"experiments/{subfolder_name}/full_model.pkl", 'wb') as f:
	pickle.dump(full_classifier, f)

# If RFE was used, also save the RFE object
if args.hing_features and args.feature_selection:
	with open(f"experiments/{subfolder_name}/full_model_rfe.pkl", 'wb') as f:
		pickle.dump(rfe_full, f)

# Save the args dictionary for easy replicability
args_dict = {}
for arg in dir(args):
	if not arg.startswith('_') and not callable(getattr(args, arg)):
		args_dict[arg] = getattr(args, arg)

with open(f"experiments/{subfolder_name}/args.json", 'w') as f:
	json.dump(args_dict, f, indent=4)

print(f"Full model training completed in {time.time() - t0:.2f} seconds.")
print(f"Model and feature importances saved to experiments/{subfolder_name}/")

# --------------------------------------------------------
# -*- perform SHAP analysis on the full model -*-

print("I'm here")

if args.perform_shapanalysis:

	print("I'm here")
	# Prepare feature names based on whether RFE was applied
	if args.hing_features:
		shap_feature_names = feature_names_full
	else:
		shap_feature_names = feature_names
	
	# Generate descriptive title based on experiment configuration
	model_name = args.model_type.upper() if args.model_type else "Model"
	feature_desc_parts = []
	if args.hing_features: feature_desc_parts.append("Expanded Features")
	if args.alberta_features_points: feature_desc_parts.append("Alberta Points")
	if args.alberta_features_raw: feature_desc_parts.append("Alberta Features")
	if args.alberta_score: feature_desc_parts.append("Alberta Score")
	feature_desc = ", ".join(feature_desc_parts) if feature_desc_parts else "Features"
	
	beeswarm_title = f"{model_name} SHAP Feature Impact ({feature_desc})"
	bar_title = f"{model_name} SHAP Feature Importance ({feature_desc})"

	print(beeswarm_title, bar_title)
	
	shap_values = perform_shap_analysis_tree(
		classifier=full_classifier,
		X_data=X_full,
		feature_names=shap_feature_names,
		output_dir=f"experiments/{subfolder_name}",
		max_display=20,
		beeswarm_title=beeswarm_title,
		bar_title=bar_title
	)