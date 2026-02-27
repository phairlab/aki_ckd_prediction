# Process for AKI->CKD project, 2023

# Import necessary libraries
import matplotlib.pyplot as plt
import pandas as pd

dataframe = pd.read_csv('features.csv') # [4694 rows x 487 columns] 
dataframe.drop(dataframe[dataframe['death_date'] <= dataframe['discharge_date']].index, inplace=True) # [4694 rows x 474 columns]
dataframe['admit_to_stage1'] = (pd.to_datetime(dataframe['stage1_date']) - pd.to_datetime(dataframe['admit_date'])).dt.days
dataframe['stage1_to_discharge'] = (pd.to_datetime(dataframe['discharge_date']) - pd.to_datetime(dataframe['stage1_date'])).dt.days
dataframe['admit_to_stage2'] = (pd.to_datetime(dataframe['stage2_date']) - pd.to_datetime(dataframe['admit_date'])).dt.days
dataframe['stage2_to_discharge'] = (pd.to_datetime(dataframe['discharge_date']) - pd.to_datetime(dataframe['stage2_date'])).dt.days
dataframe['admit_to_stage3'] = (pd.to_datetime(dataframe['stage3_date']) - pd.to_datetime(dataframe['admit_date'])).dt.days
dataframe['stage3_to_discharge'] = (pd.to_datetime(dataframe['discharge_date']) - pd.to_datetime(dataframe['stage3_date'])).dt.days
#dataframe = dataframe.fillna(0) # added but no good
title_list = list(dataframe.dtypes.index)
dataframe_values = dataframe.values
attribute_index = list(dataframe.dtypes.index).index('ckd_stage45') # only 286 cases are true 
attributes = dataframe_values[:, attribute_index]
date_indexes = [i for i in range(len(title_list)) if ('_date' in title_list[i])] # only 286 cases are true 
feature_columns = [i for i in range(np.shape(dataframe_values)[1]) if not (i in date_indexes) and i != attribute_index]
features = dataframe_values[:, feature_columns] # (4694, 467)
feature_names = (dataframe.dtypes.index.values)[feature_columns]
patient_count = attributes.shape[0] # 4694
attribute0_count = (attributes == 0).sum()  # 4408
attribute1_count = (attributes == 1).sum()  # 286
print('Number of features, total count, positive, scale_pos_weight: %d, %d, %d, %f' %
	(feature_names.shape[0], patient_count, attribute1_count, attribute0_count/attribute1_count))
# Number of features, total count, positive, scale_pos_weight: 480, 4694, 286, 15.412587

#------------
# Generate importances:

import time
start_time = time.time()
from xgboost import XGBClassifier
np.random.seed(0) 
classifier = XGBClassifier(scale_pos_weight = math.sqrt(attribute0_count/attribute1_count), learning_rate = 0.1)

df = pd.DataFrame(data = features, columns = feature_names)
classifier.fit(features, attributes)
importances = classifier.feature_importances_ 
indices = np.argsort(importances)[::-1]
for f in range(features.shape[1]):
	print('%d\t%d\t%s\t%f' % (f, indices[f], feature_names[indices[f]], importances[indices[f]]))

from sklearn.feature_selection import RFE
import time
start_time = time.time()
rfe = RFE(estimator = classifier, n_features_to_select = 240, step = 0.1, verbose=1) # 381 seconds
rfe = rfe.fit(features.astype(float), attributes.astype('int8')) # astype needed 
features_selected = rfe.transform(features)
feature_names_selected = feature_names[rfe.support_]
print(time.time() - start_time) # 364sec
classifier.fit(features_selected.astype(float), attributes) #, xgb_model = xgb_file)
importances = classifier.feature_importances_
indices = np.argsort(importances)[::-1]
for j in range(features_selected.shape[1]):
	print('%d\t%d\t%s\t%.6f' % (j, indices[j], feature_names_selected[indices[j]], importances[indices[j]]))

from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib
import matplotlib.pyplot as plt

cv = KFold(n_splits=10)
features_used = features_selected.copy()
attributes_used = attributes.copy()

# saving BEFORE shuffling for forming shap diagram:
from numpy import savetxt
attributes_used_expanded  = np.expand_dims(attributes_used, 1)
attributes_features = np.concatenate((attributes_used_expanded, features_used), axis=1)
savetxt('attributes_features_20231119.csv', attributes_features, delimiter=',')

# SHUFFLE columns and then rows:
np.random.shuffle(features_used.T)
attributes_used_expanded  = np.expand_dims(attributes_used, 1)
attributes_features = np.concatenate((attributes_used_expanded, features_used), axis=1)
np.random.shuffle(attributes_features)
features_used = attributes_features[:, 1:]
attributes_used = attributes_features[:, 0]

# CV:
tprs = []
aucs1 = []
aucs2 = []
precisions = []
mean_fpr1 = np.linspace(0, 1, 100)
mean_fpr2 = np.linspace(0, 1, 100)
i = 0
accuracy_sum = 0
sensitivity_sum = 0
specificity_sum = 0
ppv_sum = 0
npv_sum = 0
f1_sum = 0
y_real = []
y_proba = []
from numpy import argmax
plt.clf()
best_threshold_roc = 0
best_threshold_prc = 0
for train, test in cv.split(features_used, attributes_used):
	classifier.fit(features_used[train], attributes_used[train])
	y_pred = classifier.predict(features_used[test])
	attributes_used = attributes_used.astype('int8')
	y_pred = y_pred.astype('int8')
	cm = confusion_matrix(attributes_used[test], y_pred)
	TN = cm[0,0]
	FP = cm[0,1]
	FN = cm[1,0]
	TP = cm[1,1]
	P = TP+FN
	N = TN+FP
	accuracy = (TP+TN)/(P+N)
	sensitivity = TP/P
	specificity = TN/N
	ppv = TP/(TP+FP)
	npv = TN/(TN+FN)
	sensitivity = TP/(TP+FN)
	specificity = TN/(TN+FP)
	f1 = f1_score(attributes_used[test], y_pred)
	accuracy_sum += accuracy
	sensitivity_sum += sensitivity
	specificity_sum += specificity
	ppv_sum += ppv
	npv_sum += npv
	f1_sum += f1
	probas_ = classifier.fit(features_used[train], attributes_used[train]).predict_proba(features_used[test])
	fpr, tpr, roc_thresholds = roc_curve(attributes_used[test], probas_[:, 1])
	gmeans = tpr * (1-fpr) # math.sqrt fails 
	ix1 = argmax(gmeans) # G-Mean = sqrt(Sensitivity * Specificity)
	print('fpr=%f, tpr=%f, ROC Best Threshold=%f, G-Mean=%f' % (fpr[ix1], tpr[ix1], 
		roc_thresholds[ix1], math.sqrt(gmeans[ix1])))
	best_threshold_roc += roc_thresholds[ix1]
	precision, recall, prc_thresholds = precision_recall_curve(attributes_used[test], probas_[:, 1])
	fscore = (2 * precision * recall) / (precision + recall)
	ix2 = argmax(fscore)
	print('recall=%f, precision=%f, PRC Best Threshold=%f, fscore=%f' % (recall[ix2], precision[ix2], 
		prc_thresholds[ix2], fscore[ix2]))
	best_threshold_prc += prc_thresholds[ix2]
	tprs.append(np.interp(mean_fpr1, fpr, tpr))
	precisions.append(np.interp(mean_fpr2, recall, precision))
	y_real.append(attributes_used[test])
	y_proba.append(probas_[:, 1])
	tprs[-1][0] = 0.0
	precisions[-1][0] = 1.0
	roc_auc = auc(fpr, tpr)
	prc_auc = auc(recall, precision)
	print('ROC_AUC', roc_auc, 'PRC_AUC', prc_auc)
	aucs1.append(roc_auc)
	aucs2.append(prc_auc)
	plt.figure(1)
	plt.plot(fpr, tpr, lw=1, alpha=0.3, label='Fold %d: AUC=%0.4f' % (i, roc_auc))
	plt.scatter(fpr[ix1], tpr[ix1], marker='.')
	print('roc shape', roc_thresholds.shape[0])
	for k in range(roc_thresholds.shape[0]):
		if roc_thresholds[k] < 0.5:
			plt.scatter(fpr[k], tpr[k], marker='.', color='black')
			break
	plt.figure(2)
	plt.plot(recall, precision, lw=1, alpha=0.3, label='Fold %d: AUC=%0.4f' % (i, prc_auc))
	plt.scatter(recall[ix2], precision[ix2], marker='.')
	print('prc shape', prc_thresholds.shape[0])
	for k in range(prc_thresholds.shape[0]):
		if prc_thresholds[k] > 0.5:
			plt.scatter(recall[k], precision[k], marker='.', color='black')
			break
	i += 1

font = {'family' : 'normal', 'size'   : 10}
matplotlib.rc('font', **font)
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
auroc = mean_auc1 = auc(mean_fpr1, mean_tpr)
std_auc1 = np.std(aucs1)

plt.figure(1)
plt.plot(mean_fpr1, mean_tpr, color='b', label=r'Mean: AUC=%0.4f' % (mean_auc1), lw=1.5, alpha=.8)
plt.plot([0, 1], [0, 1], linestyle='--', lw=1.5, color='r', alpha=.8)
std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr1, tprs_lower, tprs_upper, color='grey', alpha=.2)
plt.tick_params(axis='both', which='major', labelsize=11)
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
plt.title('Receiver Operating Curve', fontsize=13)
plt.legend(loc="lower right")

plt.figure(2)
y_real = np.concatenate(y_real)
y_proba = np.concatenate(y_proba)
precision, recall, _ = precision_recall_curve(y_real, y_proba)
mean_auc2 = average_precision_score(y_real, y_proba)
plt.plot(recall, precision, color='b', label='Mean: AUC=%0.4f' % (mean_auc2), lw=1.5, alpha=.8)
plt.tick_params(axis='both', which='major', labelsize=11)
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('Recall (Sensitivity)', fontsize=12)
plt.ylabel('Precision (PPV)', fontsize=12)
plt.title('Precision-Recall Curve', fontsize=13)
plt.legend(loc='lower left')

print('Accuracy:', '%.4f' % (accuracy_sum/10), ' Sensitivity:','%.4f' % (sensitivity_sum/10), ' Specificity:','%.4f' % (specificity_sum/10),
	' PPV:','%.4f' % (ppv_sum/10), ' NPV:','%.4f' % (npv_sum/10), ' F1 Score:','%.4f' % (f1_sum/10), ' AUROC:','%.4f' % (auroc))
print('Best threshold: roc:', '%.4f' %(best_threshold_roc/10), '   prc:', '%.4f' %(best_threshold_prc/10))
plt.show()

#=====================================================================
from xgboost import XGBClassifier
import shap
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

attributes_features_selected = pd.read_csv('attributes_features_20230904.csv', header=None) # unshuffled
features_selected = np.array(attributes_features_selected)[:, 1:] # (158991, 250)
attributes = np.array(attributes_features_selected)[:, 0] # (158991,)
attribute0_count = (attributes == 0).sum()  # 4408
attribute1_count = (attributes == 1).sum()  # 286

classifier = XGBClassifier(scale_pos_weight = attribute0_count/attribute1_count, learning_rate = 0.1)
features_used = features_selected.copy()
feature_names_used = np.arange(0, 240)
df = pd.DataFrame(data = features_used) # [4694 rows x 240 columns]
df = df.set_axis(feature_names_used, axis=1, inplace=False)
classifier.fit(features_selected.astype(float), np.ravel(attributes))
explainer = shap.TreeExplainer(classifier)
shap_values = explainer(df)

# display on screen:
shap.plots.beeswarm(shap_values, plot_size=(18,14), max_display=60, color=plt.get_cmap("cool"))

# saved as png: from   https://github.com/slundberg/shap/issues/153:
plt.clf()
figure = plt.gcf()
shap.plots.beeswarm(shap_values, plot_size=(18,80), max_display=240, color=plt.get_cmap("cool"), show=False) #
figure.savefig('SHAP_Shuffled_240_plot_size=(18,80)_20230904.png')

# mean-absolute-shap-values
for i in range(shap_values.values.shape[1]):
	print('%d\t%.6f' % (i, np.mean(np.abs(shap_values.values[:, i]))))

#==============
# ALE:

# To create and save features_selected_with_attributes.csv
dataframe = pd.read_csv('features.csv') # [4694 rows x 481 columns] 
dataframe.drop(dataframe[dataframe['death_date'] <= dataframe['discharge_date']].index, inplace=True) # [4694 rows x 474 columns]
dataframe['admit_to_stage1'] = (pd.to_datetime(dataframe['stage1_date']) - pd.to_datetime(dataframe['admit_date'])).dt.days
dataframe['stage1_to_discharge'] = (pd.to_datetime(dataframe['discharge_date']) - pd.to_datetime(dataframe['stage1_date'])).dt.days
dataframe['admit_to_stage2'] = (pd.to_datetime(dataframe['stage2_date']) - pd.to_datetime(dataframe['admit_date'])).dt.days
dataframe['stage2_to_discharge'] = (pd.to_datetime(dataframe['discharge_date']) - pd.to_datetime(dataframe['stage2_date'])).dt.days
dataframe['admit_to_stage3'] = (pd.to_datetime(dataframe['stage3_date']) - pd.to_datetime(dataframe['admit_date'])).dt.days
dataframe['stage3_to_discharge'] = (pd.to_datetime(dataframe['discharge_date']) - pd.to_datetime(dataframe['stage3_date'])).dt.days
#dataframe = dataframe.fillna(0) # added but no good
title_list = list(dataframe.dtypes.index)
dataframe_values = dataframe.values
attribute_index = list(dataframe.dtypes.index).index('ckd_stage45') # only 286 cases are true 
attributes = dataframe_values[:, attribute_index]
date_indexes = [i for i in range(len(title_list)) if ('_date' in title_list[i])] 
feature_columns = [i for i in range(np.shape(dataframe_values)[1]) if not (i in date_indexes) and i != attribute_index]
features = dataframe_values[:, feature_columns]
feature_names = (dataframe.dtypes.index.values)[feature_columns]
patient_count = attributes.shape[0]
attribute0_count = (attributes == 0).sum()
attribute1_count = (attributes == 1).sum()
print('Number of features, total count, positive, scale_pos_weight: %d, %d, %d, %f' %
	(feature_names.shape[0], patient_count, attribute1_count, attribute0_count/attribute1_count))
# Number of features, total count, positive, scale_pos_weight: 480, 4694, 286, 15.412587

from xgboost import XGBClassifier
np.random.seed(0) 
classifier = XGBClassifier(scale_pos_weight = attribute0_count/attribute1_count, learning_rate = 0.1)
from sklearn.feature_selection import RFE
rfe = RFE(estimator = classifier, n_features_to_select = 240, step = 0.1, verbose=1)
rfe = rfe.fit(features.astype(float), attributes.astype('int8')) # astype needed 
features_selected = rfe.transform(features)
feature_names_selected = feature_names[rfe.support_]
classifier.fit(features_selected.astype(float), attributes) #, xgb_model = xgb_file)
importances = classifier.feature_importances_
indices = np.argsort(importances)[::-1]
for j in range(features_selected.shape[1]):
	print('%d\t%d\t%s\t%.6f' % (j, indices[j], feature_names_selected[indices[j]], importances[indices[j]]))

df = pd.DataFrame(features_selected)
df['attribute'] = attributes
header = list(feature_names_selected)
header.append('attribute')
df.to_csv('features_selected_with_attributes.csv', header=header, index=False)

""" Useful?
features_selected_sorted = features_selected[:,indices]
feature_names_selected_sorted = feature_names_selected[indices]
df = pd.DataFrame(features_selected_sorted)
df['attribute'] = attributes
header = list(feature_names_selected_sorted)
header.append('attribute')
df.to_csv('features_selected_sorted_with_attributes.csv', header=header, index=False)
"""

# Try to reduce number of features, to see if explain(X_train) still takes forever:
dataframe = pd.read_csv('features_selected_with_attributes.csv') # [158991 rows x 251 columns]
dataframe_values = dataframe.values # (158991, 251)
attribute_index = list(dataframe.dtypes.index).index('attribute') # 250
attributes = dataframe_values[:, attribute_index] # array([0., ..., 0.])
feature_columns = [i for i in range(np.shape(dataframe_values)[1]) if i != attribute_index]
features = dataframe_values[:, feature_columns] # (158991, 250)
feature_names = (dataframe.dtypes.index.values)[feature_columns]
admission_count = attributes.shape[0]
attribute0_count = (attributes == 0).sum()
attribute1_count = (attributes == 1).sum()
print('Number of features, total count, positive, scale_pos_weight: %d, %d, %d, %f' %	(feature_names.shape[0], admission_count, attribute1_count, attribute0_count/attribute1_count)) # 250, 158991, 28484, 4.581765
classifier = XGBClassifier(scale_pos_weight = attribute0_count/attribute1_count, learning_rate = 0.1)
from sklearn.feature_selection import RFE
#rfe = RFE(estimator = classifier, n_features_to_select = 10, step = 40, verbose=1)
rfe = RFE(estimator = classifier, n_features_to_select = 50, step = 20, verbose=1)
rfe = rfe.fit(features.astype(float), attributes.astype('int8')) # astype needed 
features_selected = rfe.transform(features) # (158991, 10)
feature_names_selected = feature_names[rfe.support_]
for j in range(features_selected.shape[1]):
	print('%d\t%s' % (j, feature_names_selected[j]))

df = pd.DataFrame(features_selected)
df['attribute'] = attributes
header = list(feature_names_selected)
header.append('attribute')
#df.to_csv('ten_features_selected_with_attributes.csv', header=header, index=False) # [158991 rows x 11 columns]
df.to_csv('fifty_features_selected_with_attributes.csv', header=header, index=False) # [158991 rows x 11 columns]

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from alibi.explainers import ALE, plot_ale

#dataframe = pd.read_csv('ten_features_selected_with_attributes.csv') # [158991 rows x 51 columns]
dataframe = pd.read_csv('fifty_features_selected_with_attributes.csv') # [158991 rows x 51 columns]
dataframe_values = dataframe.values # (158991, 51)
attribute_index = list(dataframe.dtypes.index).index('attribute') # 50
attributes = dataframe_values[:, attribute_index] # array([0., ..., 0.])
feature_columns = [i for i in range(np.shape(dataframe_values)[1]) if i != attribute_index]
features = dataframe_values[:, feature_columns] # (158991, 50)
feature_names = (dataframe.dtypes.index.values)[feature_columns]
admission_count = attributes.shape[0]
attribute0_count = (attributes == 0).sum() 
attribute1_count = (attributes == 1).sum()
print('Number of features, total count, positive, scale_pos_weight: %d, %d, %d, %f' %	(feature_names.shape[0], admission_count, attribute1_count, attribute0_count/attribute1_count)) # 50, 158991, 28484, 4.581765
X_train, X_test, y_train, y_test = train_test_split(features, attributes, test_size=0.25, random_state=42)
from xgboost import XGBClassifier
classifier = XGBClassifier(scale_pos_weight = attribute0_count/attribute1_count, learning_rate = 0.1)
classifier.fit(X_train, y_train)
accuracy_score(y_test, classifier.predict(X_test))
proba_fun_classifier = classifier.predict_proba
proba_ale_classifier = ALE(proba_fun_classifier)
proba_exp_classifier = proba_ale_classifier.explain(X_train)
plot_ale(proba_exp_classifier)
plt.show()  
# "Tight layout not applied. tight_layout cannot make axes height small enough to accommodate all axes decorations."
lot_ale(proba_exp_classifier)
plot_ale(proba_exp_classifier, features=[0,1,2,3,4,5,6,7,8,9,10,11])
plt.show()  
plot_ale(proba_exp_classifier, features=[12,13,14,15,16,17,18,19,20,21,22,23])
plt.show()  
plot_ale(proba_exp_classifier, features=[24,25,26,27,28,29,30,31,32,33,34,35])
plt.show()  
plot_ale(proba_exp_classifier, features=[36,37,38,39,40,41,42,43,44,45,46,47])
plt.show()  
plot_ale(proba_exp_classifier, features=[34])
plt.show()  
plot_ale(proba_exp_classifier, features=[35])
plt.show()  
plot_ale(proba_exp_classifier, features=[31])
plt.show()  