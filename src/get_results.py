import os
import re
import random
import math
import time
import json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.metrics import precision_recall_curve, average_precision_score

from xgboost import XGBClassifier
from datetime import datetime

from alberta_score_helpers import *
import os
import pickle
import shap

random.seed(1202)
np.random.seed(1202) 

# which features to include for the final model
hing_features = False
alberta_features = False
alberta_score = True

perform_cv = False  # whether to perform cross-validation or train on the entire dataset
perform_shapanalysis = True  # whether to perform SHAP analysis on the final model

# --------------------------------------------------------
# -*- load in the features.csv file -*-

# Construct the file path
file_path = "/data/kidney/Hing/features.csv"

# Load the CSV file into a DataFrame
features_df = pd.read_csv(file_path)  # contains all the features that hing engineered

# cast AdmitDt and DischDt to datetime
features_df['admit_date'] = pd.to_datetime(features_df['admit_date'])
features_df['discharge_date'] = pd.to_datetime(features_df['discharge_date'])

# drop patients who died before they were discharged
features_df.drop(features_df[features_df['death_date'] <= features_df['discharge_date']].index, inplace=True)

if hing_features:
    features_df['admit_to_stage1'] = (pd.to_datetime(features_df['stage1_date']) - pd.to_datetime(features_df['admit_date'])).dt.days
    features_df['stage1_to_discharge'] = (pd.to_datetime(features_df['discharge_date']) - pd.to_datetime(features_df['stage1_date'])).dt.days
    features_df['admit_to_stage2'] = (pd.to_datetime(features_df['stage2_date']) - pd.to_datetime(features_df['admit_date'])).dt.days
    features_df['stage2_to_discharge'] = (pd.to_datetime(features_df['discharge_date']) - pd.to_datetime(features_df['stage2_date'])).dt.days
    features_df['admit_to_stage3'] = (pd.to_datetime(features_df['stage3_date']) - pd.to_datetime(features_df['admit_date'])).dt.days
    features_df['stage3_to_discharge'] = (pd.to_datetime(features_df['discharge_date']) - pd.to_datetime(features_df['stage3_date'])).dt.days

# contains only what we need for the Alberta score + ckd_stage45
alberta_df = features_df[["patient_id", "admit_date", "discharge_date", "sex", "age_admit", "highest_stage", "ckd_stage45"]]
alberta_df = alberta_df.rename(columns={"sex": "sex_raw", "age_admit": "age_admit_raw", "highest_stage": "highest_stage_raw"})

print("done loading features.csv")

# --------------------------------------------------------
# -*- load in the lab tests -*-

# Load in the index labs

index_labs_file_path = "/data/kidney/Hing/in-hosp labs.csv"
index_labs_df = pd.read_csv(index_labs_file_path)

# Load the pre-index labs
prehosp_labs_file_path =  "/data/kidney/Hing/pre-hosp labs.csv"
prehosp_labs_df = pd.read_csv(prehosp_labs_file_path)

# combine the two
all_labs_df = pd.concat([index_labs_df, prehosp_labs_df], ignore_index=True)

# Cast test_date, AdmitDt, and DischDt to datetime
all_labs_df['test_date'] = pd.to_datetime(all_labs_df['test_date'])
all_labs_df['AdmitDt'] = pd.to_datetime(all_labs_df['AdmitDt'])
all_labs_df['DischDt'] = pd.to_datetime(all_labs_df['DischDt'])

print("done loading lab tests")


# --------------------------------------------------------
# -*- alberta score points -*-

# sex
alberta_df["sex_points"] = alberta_df.sex_raw.map({1: 3, 0: 0})

# age
alberta_df["age_admit_points"] = alberta_df.age_admit_raw.map(age_mapping)

# highest stage
alberta_df["highest_stage_points"] = alberta_df.highest_stage_raw.map(stage_mapping)

# baseline creatinine
alberta_df['baseline_creatinine_raw'] = alberta_df.apply(
    lambda row: get_baseline_creatinine(row['patient_id'], all_labs_df, row['admit_date']), axis=1
)
alberta_df["baseline_creatinine_points"] = alberta_df.baseline_creatinine_raw.map(baseline_creatinine_mapping)

# discharge creatinine
alberta_df['discharge_creatinine_raw'] = alberta_df.apply(
    lambda row: get_discharge_creatinine(row['patient_id'], all_labs_df, row['admit_date'], row['discharge_date']), axis=1
)
alberta_df["discharge_creatinine_points"] = alberta_df.discharge_creatinine_raw.map(discharge_creatinine_mapping)

# albuminuria status
alberta_df['albuminuria_status_raw'] = alberta_df.apply(
    lambda row: get_albuminuria_status(row['patient_id'], all_labs_df, row['admit_date'], row['discharge_date'])[0], axis=1
)
alberta_df["albuminuria_status_points"] = alberta_df.albuminuria_status_raw.map(albuminuria_status_mapping)

print("done calculating alberta score points")


# --------------------------------------------------------
# -*- drop patients who don't have a baseline creatinine and discharge creatinine

alberta_df = alberta_df.dropna(subset=['baseline_creatinine_raw'])
alberta_df = alberta_df.dropna(subset=['discharge_creatinine_raw'])

print("done dropping patients without baseline and discharge creatinine")

# --------------------------------------------------------
# -*- calculate alberta score 

alberta_df["alberta_score"] = alberta_df["sex_points"]+\
                              alberta_df["age_admit_points"]+\
                              alberta_df["highest_stage_points"]+\
                              alberta_df["baseline_creatinine_points"]+\
                              alberta_df["discharge_creatinine_points"]+\
                              alberta_df["albuminuria_status_points"]

print("done calculating alberta score")


# --------------------------------------------------------
# -*- standardize feature set patient list and order -*-

# Filter features_df to include only patients present in alberta_df
features_df = features_df[features_df['patient_id'].isin(alberta_df['patient_id'])]

# Filter alberta_df to include only patients present in features_df
alberta_df = alberta_df[alberta_df['patient_id'].isin(features_df['patient_id'])]

# Sort both DataFrames by patient_id
features_df = features_df.sort_values(by='patient_id').reset_index(drop=True)
alberta_df = alberta_df.sort_values(by='patient_id').reset_index(drop=True)

# Ensure they have the same shape
assert features_df.shape[0] == alberta_df.shape[0], "DataFrames do not have the same number of rows."

print("done standardizing feature set patient list and order")


# --------------------------------------------------------
# -*- create features_used, feature_names, attributes_used -*-
# Hing used "attribute" to refer to the target variable

alberta_score_feature = alberta_df[[
    "alberta_score"
]]

alberta_points_features = alberta_df[[
    "sex_points",
    "age_admit_points",
    "highest_stage_points",
    "baseline_creatinine_points",
    "discharge_creatinine_points",
    "albuminuria_status_points"
]]

if hing_features:  # use hing's features
    if alberta_features:
        # add the alberta score features to the feature set
        features_df = pd.concat([features_df, alberta_points_features], axis=1)
    if alberta_score:
        # add the alberta score to the feature set
        features_df = pd.concat([features_df, alberta_score_feature], axis=1)

    # Hing code
    title_list = list(features_df.dtypes.index)
    dataframe_values = features_df.values
    attribute_index = list(features_df.dtypes.index).index('ckd_stage45') # only 286 cases are true 
    attributes_used = dataframe_values[:, attribute_index]
    date_indexes = [i for i in range(len(title_list)) if ('_date' in title_list[i])]  # dates
    id_indexes = [i for i in range(len(title_list)) if ('patient_id' in title_list[i])]  # id numbers
    after_indexes = [i for i in range(len(title_list)) if ('after' in title_list[i])]  # other outcomes that aren't kidney disease progression
    feature_columns = [i for i in range(np.shape(dataframe_values)[1]) if not (i in date_indexes+id_indexes+after_indexes) and i != attribute_index]
    features_used = dataframe_values[:, feature_columns] # (4694, 467)
    feature_names = features_df.dtypes.index[feature_columns]

elif alberta_features:
    features_df = alberta_points_features.copy()  # use only the alberta points features
    
    if alberta_score:
        # add the alberta score to the feature set
        features_df = pd.concat([features_df, alberta_score_feature], axis=1)

    features_used = features_df.values
    feature_names = features_df.columns.values
    attributes_used = alberta_df["ckd_stage45"].values

    print(feature_names)

else:  # use only the alberta score
    features_used = alberta_score_feature.values
    feature_names = alberta_score_feature.columns.values
    attributes_used = alberta_df["ckd_stage45"].values

print("done creating features_used, feature_names, attributes_used")


# --------------------------------------------------------
# -*- run training loop and get results -*-

# Define the date and feature type for the subfolder name
current_date = datetime.now().strftime("%Y%m%d")
features_list = []
if hing_features: features_list.append("hing")
if alberta_features: features_list.append("abpoints")
if alberta_score: features_list.append("abscore")
extra = ""

features_string = "-".join(features_list)
subfolder_name = f"{current_date}_{features_string}_{extra}_fold_results"

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

if perform_cv:
	# Perform cross-validation
	for i, (train, test) in enumerate(cv.split(features_used, attributes_used)):
		if i < skip:  # Skip the first skip folds (for if training gets halted)
			continue
		
		t0 = time.time()

		print(f"Training fold {i + 1}...")

		X_train, y_train = features_used[train].astype(float), attributes_used[train].astype('int8')
		X_test, y_test = features_used[test].astype(float), attributes_used[test].astype('int8')

		attribute0_count = np.sum(y_train == 0)  # Count of attribute 0 in training set
		attribute1_count = np.sum(y_train == 1)  # Count of attribute 1 in training set

		classifier = XGBClassifier(random_state=1202)  # Initialize the classifier with a random seed
		# classifier = XGBClassifier(random_state=1202, scale_pos_weight = math.sqrt(attribute0_count/attribute1_count), learning_rate = 0.1)  # Initialize the classifier

		feature_names_fold = feature_names

		if hing_features:  # perform RFE on the training set
			rfe = RFE(estimator=classifier, n_features_to_select=240, step=0.1, verbose=1)  # 381 seconds
			rfe = rfe.fit(X_train, y_train) 

			X_train = rfe.transform(X_train)
			X_test = rfe.transform(X_test)

			feature_names_fold = feature_names[rfe.support_]

		# Train the classifier on the training set
		classifier.fit(X_train.astype(float), y_train)

		# Predict on the test set
		y_pred = classifier.predict(X_test)
		y_pred = y_pred.astype('int8')

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

		# save feature importances to a file
		importances = classifier.feature_importances_
		indices = np.argsort(importances)[::-1]
		with open(f"experiments/{subfolder_name}/fold_{i + 1}_feature_importances.txt", 'w') as f:
			for j in range(X_train.shape[1]):
				f.write('%d\t%d\t%s\t%.6f\n' % (j, indices[j], feature_names_fold[indices[j]], importances[indices[j]]))

		print("Fold {} completed in {:.2f} seconds.".format(i + 1, time.time() - t0))
		# i += 1  # Increment fold counter

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
if hing_features:
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
if hing_features:
	with open(f"experiments/{subfolder_name}/full_model_rfe.pkl", 'wb') as f:
		pickle.dump(rfe_full, f)

print(f"Full model training completed in {time.time() - t0:.2f} seconds.")
print(f"Model and feature importances saved to experiments/{subfolder_name}/")

# --------------------------------------------------------
# -*- perform SHAP analysis on the full model -*-

if perform_shapanalysis:
	print("Performing SHAP analysis on the full model...")
	
	# Create a DataFrame for SHAP analysis
	if hing_features:
		# Use the reduced feature set if RFE was applied
		shap_df = pd.DataFrame(X_full, columns=feature_names_full)
	else:
		shap_df = pd.DataFrame(X_full, columns=feature_names)
	
	# Initialize the SHAP explainer
	explainer = shap.TreeExplainer(full_classifier)
	shap_values = explainer(shap_df)
	
	# Generate and save the beeswarm plot
	plt.clf()
	figure = plt.gcf()
	
	# Standard size plot with top features
	shap.plots.beeswarm(shap_values, plot_size=(18, 14), max_display=60, 
						color=plt.get_cmap("cool"), show=False)
	figure.savefig(f"experiments/{subfolder_name}/shap_beeswarm_top60.png", dpi=300, bbox_inches='tight')
	
	# Larger plot with more features
	plt.clf()
	figure = plt.gcf()
	shap.plots.beeswarm(shap_values, plot_size=(18, 80), max_display=240, 
						color=plt.get_cmap("cool"), show=False)
	figure.savefig(f"experiments/{subfolder_name}/shap_beeswarm_top240.png", dpi=300, bbox_inches='tight')
	
	# Save SHAP values for future analysis
	with open(f"experiments/{subfolder_name}/shap_values.pkl", 'wb') as f:
		pickle.dump(shap_values, f)
	
	print(f"SHAP analysis completed and plots saved to experiments/{subfolder_name}/")