

# ETL for AKI->CKD project, 2023

# -----------------
# cohort and outcome.csv

cohort = pd.read_csv('cohort and outcome.csv') # [4694 rows x 21 columns]
# ['AdmitDt', 'DischDt', 'SEX', 'AGE_ADMIT', 'total_los', 'stage1_dt', 'stage1_value', 'stage1', 'stage2_dt', 'stage2_value', 'stage2', 'stage3_dt', 'stage3_value', 'stage3', 'higheststage', 'death_date', 'CKD_stage45', 'stroke_af', 'chf_af', 'mi_af', 'id']
cohort.rename(columns={'id':'patient_id', 'AGE_ADMIT':'age_admit', 'stage1_dt':'stage1_date', 'stage1_value':'stage1_creatinine',
	'stage2_dt':'stage2_date', 'stage2_value':'stage2_creatinine', 'stage3_dt':'stage3_date', 'stage3_value':'stage3_creatinine',
	'higheststage':'highest_stage', 'CKD_stage45':'ckd_stage45', 'stroke_af':'stroke_after', 'chf_af':'chf_after', 'mi_af':'mi_after'},
	inplace=True)
cohort['admit_date'] = pd.to_datetime(cohort['AdmitDt'], format='%Y-%m-%d')
cohort['discharge_date'] = pd.to_datetime(cohort['DischDt'], format='%Y-%m-%d')
cohort['sex'] = (cohort['SEX'] == 'F').astype(int)
cohort['death_date'] = pd.to_datetime(cohort['death_date'], format='%Y-%m-%d')
cohort = cohort[['patient_id', 'admit_date', 'discharge_date', 'sex', 'age_admit', 'total_los', 'stage1', 'stage1_date',
	'stage1_creatinine', 'stage2', 'stage2_date', 'stage2_creatinine', 'stage3', 'stage3_date', 'stage3_creatinine',
	'highest_stage', 'death_date', 'ckd_stage45', 'stroke_after', 'chf_after', 'mi_after']] # [4694 rows x 21 columns]
cohort.to_csv('cohort.csv', index=False)

# -----------------
# in-hosp vars

def goal(g):
	return 'perioperative' if 'Perioperative' in g else 'acute' if 'Acute' in g else 'community' if 'Community' in g\
else 'transition' if 'Transition' in g else 'alc' if 'ACL' in g else 'intensive' if 'Intensive' in g else 'mobility' if\
'Mobility' in g else 'assessment' if 'Assessment' in g else 'waiting' if 'Waiting' in g else np.nan if g == '?' else 'others' 

service_list = ['General Internal Medicine', 'Cardiac Surgery', 'Emergency Medicine', 'Nephrology', 'Neurology', 'Transplant',
	'Cardiology', 'Gastroenterology', 'Neurosurgery', 'Surgery', 'Family Medicine', 'Critical Care', 'Trauma', 'Urology',
	'General Surgery', 'Orthopedics', 'Specialized Geriatrics', 'Otolaryngology', 'Stroke', 'Ear Nose and Throat', 'ICU', 
	'Transplant', 'Plastic Surgery', 'Hematology']

index_vars = pd.read_csv('in-hosp vars.csv') # [4694 rows x 23 columns]
# ['AdmitDt', 'DischDt', 'cardiac_surgery', 'ami', 'chf', 'icu_inhosp', 'insulin', 'beta_blocker', 'covid_test_result', 'smoke', 'goals_of_care', 'admission_services', 'foley_catheter', 'OT_assessment', 'renal_ultralsound', 'PT_assessment', 'angiogram', 'sepsis', 'obstructive_uropathy', 'Cardiac_catheterization', 'Mechanical_ventilation', 'dialysis', 'id']
index_vars.rename(columns={'id':'patient_id', 'icu_inhosp':'icu', 'Cardiac_catheterization':'cardiac_catheterization',
	'Mechanical_ventilation':'mechanical_ventilation', 'OT_assessment':'ot_assessment', 'PT_assessment':'pt_assessment'}, inplace=True)
index_vars['goals_of_care'].fillna('?', inplace=True)
index_vars['goals_of_care'] = index_vars['goals_of_care'].apply(goal)
for g in ['acute', 'community', 'transition', 'others', 'intensive', 'mobility', 'assessment', 'perioperative']:
	index_vars['goal_' + g] = (index_vars['goals_of_care'] == g).astype('int')

index_vars['covid_test_result'] = (index_vars['covid_test_result'] == 'Positive').astype('int')
index_vars = index_vars[['patient_id', 'icu', 'goal_acute', 'goal_community', 'goal_transition', 'goal_others', 'goal_intensive',
	'goal_mobility', 'goal_assessment', 'goal_perioperative', 'cardiac_surgery', 'insulin', 'beta_blocker', 
	'covid_test_result', 'smoke', 'cardiac_catheterization', 'ami', 'chf', 'dialysis', 'mechanical_ventilation',
	'renal_ultralsound', 'angiogram', 'foley_catheter', 'ot_assessment', 'pt_assessment', 'obstructive_uropathy', 'sepsis',
	'admission_services']]
for service in service_list:
	index_vars[service.lower().replace(' ', '_')] = np.where(index_vars['admission_services']==service, 1, 0)

index_vars.drop('admission_services', inplace=True, axis=1) # [4694 rows x 48 columns]
index_vars = index_vars.add_prefix('index_vars:')
index_vars.rename(columns={'index_vars:patient_id' : 'patient_id'}, inplace=True)
index_vars.to_csv('index_vars.csv', index=False)

# -----------------
# in-hosp flowsheet records

def record_name(n):
	return 'arterial_line_bp' if 'ARTERIAL' in n else 'bp' if 'PRESSURE' in n else 'heart_rate_ecg' if 'ECG' in n\
else 'oxygen_therapy' if 'OXYGEN' in n else 'bar_bmi' if 'BAR BMI' in n else 'bmi' if 'BMI' in n else 'max_heart_rate' if\
'MAC HEART RATE' in n else 'aortic_heart_rate' if 'AORTIC' in n else 'aware' if 'AWARE' in n else np.nan if n == '?' else 'others'

def value(v):
	return '1' if (v == 'Supplemental oxygen' or 'AWARE' in v or 'Aware' in v or 'Yes' in v) else str(v)

index_records = pd.read_csv('in-hosp flowsheet records.csv') # [975935 rows x 10 columns]
# ['AdmitDt', 'DischDt', 'MEAS_VALUE', 'record_date', 'FLO_MEAS_NAME', 'oxygen', 'blood_pressure', 'heart_rate', 'bmi', 'id']
index_records['patient_id'] = index_records['id']
index_records['record_date'] = pd.to_datetime(index_records['record_date'], format='%Y-%m-%d')
index_records['name'] = index_records['FLO_MEAS_NAME'].apply(record_name)
index_records['value'] = index_records['MEAS_VALUE'].apply(value) # [975935 rows x 14 columns]
index_bp = index_records.loc[index_records['name'].isin(['bp', 'arterial_line_bp'])] # [643225 rows x 14 columns]
index_bp['systolic_bp'] = [x.split('/')[0] for x in index_bp['value']]
index_bp['systolic_bp'] = index_bp['systolic_bp'].astype(float)
index_bp['diastolic_bp'] = [x.split('/')[1] for x in index_bp['value']]
index_bp['diastolic_bp'] = index_bp['diastolic_bp'].astype(float)

index_bp_s = index_bp[['patient_id', 'record_date', 'name', 'systolic_bp']] # [643225 rows x 5 columns]
index_bp_s.rename(columns={'systolic_bp' : 'value'}, inplace=True)
index_bp_d = index_bp[['patient_id', 'record_date', 'name', 'diastolic_bp']] # [643225 rows x 5 columns]
index_bp_d.rename(columns={'diastolic_bp' : 'value'}, inplace=True)
index_non_bp = index_records.loc[~index_records['name'].isin(['bp', 'arterial_line_bp'])]
index_non_bp = index_non_bp[['patient_id', 'record_date', 'name', 'value']] # 332710 rows x 5 columns]
index_records = pd.concat([index_non_bp, index_bp_s, index_bp_d]).sort_values(['patient_id'])
index_records['value']=index_records['value'].astype(float) # [1619160 rows x 5 columns]

index_records['records_count'] = 'records_count:' + index_records['name']
index_records_count = pd.crosstab(index = [index_records['patient_id']], columns = index_records['records_count'], values = index_records['value'], aggfunc='count')
index_records['records_mean'] = 'records_mean:' + index_records['name']
index_records_mean = pd.crosstab(index = [index_records['patient_id']], columns = index_records['records_mean'], values = index_records['value'], aggfunc='mean')
index_records['records_min'] = 'records_min:' + index_records['name']
index_records_min = pd.crosstab(index = [index_records['patient_id']], columns = index_records['records_min'], values = index_records['value'], aggfunc='min')
index_records['records_max'] = 'records_max:' + index_records['name']
index_records_max = pd.crosstab(index = [index_records['patient_id']], columns = index_records['records_max'], values = index_records['value'], aggfunc='max') # each [3844 rows x 10 columns]

index_records_crosstab = index_records_count.merge(index_records_mean, on='patient_id', how='left')
index_records_crosstab = index_records_crosstab.merge(index_records_min, on='patient_id', how='left')
index_records_crosstab = index_records_crosstab.merge(index_records_max, on='patient_id', how='left')
index_records_crosstab.to_csv('index_records_crosstab.csv')  # No index=False
index_records_crosstab = pd.read_csv('index_records_crosstab.csv') # [3844 rows x 42 columns]
# ['patient_id', 'records_count:aortic_heart_rate', 'records_count:arterial_line_bp', 'records_count:aware', 'records_count:bar_bmi', 'records_count:bmi', 'records_count:bp', 'records_count:heart_rate_ecg', 'records_count:max_heart_rate', 'records_count:others', 'records_count:oxygen_therapy', etc. for mean, min, max. 

# -----------------
# in-hosp consultations

def description(d):
	start = d.lower().find('consult to ') + 11
	return d.lower()[start:]

def trim(crosstab, ratio):
	threshold = crosstab.shape[0]*ratio
	list_to_drop = []
	for c in list(crosstab.dtypes.index):
		if crosstab[c].isna().sum() > threshold:
			list_to_drop.append(c)
	crosstab.drop(list_to_drop, inplace=True, axis=1)

index_consultations = pd.read_csv('in-hosp consultations.csv') # [14434 rows x 6 columns]
# ['AdmitDt', 'DischDt', 'ORDERING_DATE', 'DESCRIPTION', 'DISPLAY_NAME', 'id']
index_consultations['patient_id'] = index_consultations['id']
index_consultations['ordering_date'] = pd.to_datetime(index_consultations['ORDERING_DATE'], format='%Y-%m-%d')
index_consultations['consultation'] = index_consultations['DESCRIPTION'].apply(description)
index_consultations['consult_count'] = 'consult_count:' + index_consultations['consultation']
index_consultations_crosstab = pd.crosstab(index = [index_consultations['patient_id']], columns = index_consultations['consult_count'], 
	values = index_consultations['consultation'], aggfunc='count')
trim(index_consultations_crosstab, 199/200)
index_consultations_crosstab.to_csv('index_consultations_crosstab.csv')  # No index=False
index_consultations_crosstab = pd.read_csv('index_consultations_crosstab.csv') # [3468 rows x 57 columns] originally 87 columns
# 'patient_id', 'consult_count:acute pain services', 'consult_count:adult acute pain services', 'consult_count:anesthesiology', 'consult_count:cardiac surgery', 'consult_count:cardiac surgery navigator', 'consult_count:cardiology', etc.

# -----------------
# in-hosp labs

def test_name(n):
	return 'C-Reactive Protein' if 'CRP' in n else 'Glucose' if n == 'Glucose (mmol/L)' else 'eGRF' if 'GFR' in n or 'Glomerular' in n else n

index_labs = pd.read_csv('in-hosp labs.csv') # [519785 rows x 8 columns]
# ['test_date', 'TEST_NM', 'TEST_RSLT', 'TEST_UOFM', 'lab_test_category', 'AdmitDt', 'DischDt', 'id']
list_selected_labs = list(index_labs['TEST_NM'].value_counts().index)[:34]
index_labs = index_labs[index_labs['TEST_NM'].isin(list_selected_labs)] # [519737 rows x 8 columns]
index_labs = index_labs[index_labs['TEST_RSLT'].astype(str).apply(lambda x: x.replace('.','',1).replace('-','',1).isnumeric())]
index_labs['patient_id'] = index_labs['id']
index_labs['test_date'] = pd.to_datetime(index_labs['test_date'], format='%Y-%m-%d')
index_labs['name'] = index_labs['TEST_NM'].apply(test_name)
index_labs['result'] = index_labs['TEST_RSLT'].astype(float) # str.split().str.get(0).astype(float) for first part

index_labs['labs_count'] = 'labs_count:' + index_labs['name']
index_labs_count = pd.crosstab(index = [index_labs['patient_id']], columns = index_labs['labs_count'], values = index_labs['result'], aggfunc='count')
index_labs['labs_mean'] = 'labs_mean:' + index_labs['name']
index_labs_mean = pd.crosstab(index = [index_labs['patient_id']], columns = index_labs['labs_mean'], values = index_labs['result'], aggfunc='mean')
index_labs['labs_min'] = 'labs_min:' + index_labs['name']
index_labs_min = pd.crosstab(index = [index_labs['patient_id']], columns = index_labs['labs_min'], values = index_labs['result'], aggfunc='min')
index_labs['labs_max'] = 'labs_max:' + index_labs['name']
index_labs_max = pd.crosstab(index = [index_labs['patient_id']], columns = index_labs['labs_max'], values = index_labs['result'], aggfunc='max') # each [4693 rows x 29 columns]

index_labs_crosstab = index_labs_count.merge(index_labs_mean, on='patient_id', how='left')
index_labs_crosstab = index_labs_crosstab.merge(index_labs_min, on='patient_id', how='left')
index_labs_crosstab = index_labs_crosstab.merge(index_labs_max, on='patient_id', how='left')
index_labs_crosstab.to_csv('index_labs_crosstab.csv')  # No index=False
index_labs_crosstab = pd.read_csv('index_labs_crosstab.csv') # [4693 rows x 118 columns]
# ['patient_id', 'labs_count:Albumin', 'labs_count:Albumin / Creatinine Ratio', 'labs_count:Bicarbonate, Arterial', 'labs_count:Bicarbonate, Bld', 'labs_count:Bicarbonate, Venous', 'labs_count:C Reactive Protein Quantitative', 'labs_count:C-Reactive Protein', 'labs_count:Cholesterol', 'labs_count:Cholesterol, Total', 'labs_count:Creatinine', labs_count:Creatinine Serum', 'labs_count:GLUCOSE RANDOM', 'labs_count:Glucose', 'labs_count:Glucose Meter', 'labs_count:Glucose, Bld', 'labs_count:Glucose, Random', 'labs_count:Glucose, random', 'labs_count:HCO3', 'labs_count:HCO3,ARTERIAL', 'labs_count:HCO3,VENOUS', 'labs_count:Hemoglobin', 'labs_count:Hemoglobin, Arterial', 'labs_count:Hemoglobin, Venous', 'labs_count:Phosphate', 'labs_count:Phosphorus', 'labs_count:Protein / Creatinine Ratio, Urine', 'labs_count:Protein Urine UA', 'labs_count:Urate', 'labs_count:eGRF', then labs_mean and so on.

# -----------------
# pre-hosp vars

pre_vars = pd.read_csv('pre-hosp vars.csv')
# ['chf_pre', 'pvd_pre', 'pud_pre', 'mild_liver_disease_pre', 'cancer_pre', 'mild_severe_liver_disease_pre', 'htn_pre', 'diab_pre', 'gout_pre', 'id', 'covid_test_result']
pre_vars['covid_test_result'] = pre_vars['covid_test_result'].replace('Positive', 1)
pre_vars.rename(columns={'id':'patient_id', 'chf_pre':'chf', 'pvd_pre':'pvd', 'pud_pre':'pud',
	'mild_liver_disease_pre':'mild_liver_disease', 'cancer_pre':'cancer', 'mild_severe_liver_disease_pre':'mild_severe_liver_disease',
	'htn_pre':'hypertension', 'diab_pre':'diabetes', 'gout_pre':'gout'}, inplace=True)
pre_vars = pre_vars[['patient_id', 'chf', 'pvd', 'pud', 'mild_liver_disease', 'cancer', 'mild_severe_liver_disease', 'hypertension', 'diabetes', 'gout', 'covid_test_result']]
pre_vars = pre_vars.add_prefix('pre-index_vars:')
pre_vars.rename(columns={'pre-index_vars:patient_id' : 'patient_id'}, inplace=True) # [4694 rows x 11 columns]
# ['patient_id', 'pre_vars:chf_pre', 'pre_vars:pvd_pre', 'pre_vars:pud_pre', 'pre_vars:mild_liver_disease_pre', 'pre_vars:cancer_pre', 'pre_vars:mild_severe_liver_disease_pre', 'pre_vars:hypertension_pre', 'pre_vars:diabetes_pre', 'pre_vars:gout_pre', 'pre_vars:covid_test_result_pre']
pre_vars.to_csv('pre_vars.csv', index=False)

# -----------------
# pre-hosp labs

def test_name(n):
	return 'C-Reactive Protein' if 'CRP' in n else 'Glucose' if n == 'Glucose (mmol/L)' else 'eGRF' if 'GFR' in n or 'Glomerular' in n else n

pre_labs = pd.read_csv('pre-hosp labs.csv') # [191034 rows x 8 columns]
# ['test_date', 'TEST_NM', 'TEST_RSLT', 'TEST_UOFM', 'lab_test_category', 'AdmitDt', 'DischDt', 'id']
list_selected_labs = list(pre_labs['TEST_NM'].value_counts().index)[:50]
pre_labs = pre_labs[pre_labs['TEST_NM'].isin(list_selected_labs)] # [167774 rows x 8 columns]
pre_labs = pre_labs[pre_labs['TEST_RSLT'].astype(str).apply(lambda x: x.replace('.','',1).replace('-','',1).isnumeric())]
pre_labs['patient_id'] = pre_labs['id']
pre_labs['test_date'] = pd.to_datetime(pre_labs['test_date'], format='%Y-%m-%d')
pre_labs['name'] = pre_labs['TEST_NM'].apply(test_name)
pre_labs['result'] = pre_labs['TEST_RSLT'].astype(float) # str.split().str.get(0).astype(float) for first part

pre_labs['labs_count'] = 'pre-index_labs_count:' + pre_labs['name']
pre_labs_count = pd.crosstab(index = [pre_labs['patient_id']], columns = pre_labs['labs_count'], values = pre_labs['result'], aggfunc='count')
pre_labs['labs_mean'] = 'pre-index_labs_mean:' + pre_labs['name']
pre_labs_mean = pd.crosstab(index = [pre_labs['patient_id']], columns = pre_labs['labs_mean'], values = pre_labs['result'], aggfunc='mean')
pre_labs['labs_min'] = 'pre-index_labs_min:' + pre_labs['name']
pre_labs_min = pd.crosstab(index = [pre_labs['patient_id']], columns = pre_labs['labs_min'], values = pre_labs['result'], aggfunc='min')
pre_labs['labs_max'] = 'pre-index_labs_max:' + pre_labs['name']
pre_labs_max = pd.crosstab(index = [pre_labs['patient_id']], columns = pre_labs['labs_max'], values = pre_labs['result'], aggfunc='max') # each [4693 rows x 29 columns]

pre_labs_crosstab = pre_labs_count.merge(pre_labs_mean, on='patient_id', how='left')
pre_labs_crosstab = pre_labs_crosstab.merge(pre_labs_min, on='patient_id', how='left')
pre_labs_crosstab = pre_labs_crosstab.merge(pre_labs_max, on='patient_id', how='left')
pre_labs_crosstab.to_csv('pre_labs_crosstab.csv')  # No index=False
pre_labs_crosstab = pd.read_csv('pre_labs_crosstab.csv') # [4242 rows x 177 columns]
# ['patient_id', 'pre_index_labs_count:ALBUMIN/CREATININE RATIO,URINE', 'pre_index_labs_count:Albumin', 'pre_index_labs_count:Albumin / Creatinine Ratio', etc., then labs_mean and so on.

# -----------------
# pre-hosp bmi

def bmi_name(n):
	return 'bar_bmi' if 'BAR' in n else 'bmi' 

pre_bmi = pd.read_csv('pre-hosp bmi.csv') # [2536 rows x 6 columns]
# ['AdmitDt', 'DischDt', 'MEAS_VALUE', 'record_date', 'FLO_MEAS_NAME', 'id']
pre_bmi['patient_id'] = pre_bmi['id']
pre_bmi['record_date'] = pd.to_datetime(pre_bmi['record_date'], format='%Y-%m-%d')
pre_bmi['name'] = pre_bmi['FLO_MEAS_NAME'].apply(bmi_name)
pre_bmi['value'] = pre_bmi['MEAS_VALUE']

pre_bmi['pre_count'] = 'pre-index_count:' + pre_bmi['name']
pre_bmi_count = pd.crosstab(index = [pre_bmi['patient_id']], columns = pre_bmi['pre_count'], values = pre_bmi['value'], aggfunc='count')
pre_bmi['pre_mean'] = 'pre-index_mean:' + pre_bmi['name']
pre_bmi_mean = pd.crosstab(index = [pre_bmi['patient_id']], columns = pre_bmi['pre_mean'], values = pre_bmi['value'], aggfunc='mean')
pre_bmi['pre_min'] = 'pre-index_min:' + pre_bmi['name']
pre_bmi_min = pd.crosstab(index = [pre_bmi['patient_id']], columns = pre_bmi['pre_min'], values = pre_bmi['value'], aggfunc='min')
pre_bmi['pre_max'] = 'pre-index_max:' + pre_bmi['name']
pre_bmi_max = pd.crosstab(index = [pre_bmi['patient_id']], columns = pre_bmi['pre_max'], values = pre_bmi['value'], aggfunc='max') # each [566 rows x 2 columns]

pre_bmi_crosstab = pre_bmi_count.merge(pre_bmi_mean, on='patient_id', how='left')
pre_bmi_crosstab = pre_bmi_crosstab.merge(pre_bmi_min, on='patient_id', how='left')
pre_bmi_crosstab = pre_bmi_crosstab.merge(pre_bmi_max, on='patient_id', how='left')
pre_bmi_crosstab.to_csv('pre_bmi_crosstab.csv')  # No index=False
pre_bmi_crosstab = pd.read_csv('pre_bmi_crosstab.csv') # [566 rows x 9 columns]
# ['patient_id', 'pre_count:bar_bmi', 'pre_count:bmi', 'pre_mean:bar_bmi', 'pre_mean:bmi', 'pre_min:bar_bmi', 'pre_min:bmi', pre_max:bar_bmi', 'pre_max:bmi']

# -----------------
# pre-hosp medication

pre_medication = pd.read_csv('pre-hosp medication.csv') # [4087 rows x 12 columns]
# 'AdmitDt', 'DischDt', 'ace_arb', 'AMINOGLYCOSIDE', 'AMPHOTERICIN_B', 'DIURETIC_K', 'DIURETIC_NONK', 'NSAIDS', 'PPI', 'SGLT2', 'VANCOMYCIN', 'id']
pre_medication = pre_medication.rename(columns={'id':'patient_id', 'AMINOGLYCOSIDE':'aminoglycoside', 
	'AMPHOTERICIN_B':'amphotericin_b', 'DIURETIC_K':'diuretic_k', 'DIURETIC_NONK':'diuretic_non_k', 'NSAIDS':'nsaids',
	'PPI':'ppi', 'SGLT2':'sglt2', 'VANCOMYCIN':'vancomycin'})
pre_medication = pre_medication[['patient_id', 'aminoglycoside', 'amphotericin_b', 'diuretic_k', 'diuretic_non_k', 'nsaids',
	'ppi', 'sglt2', 'vancomycin']]
pre_medication = pre_medication.add_prefix('pre-index_medication:')
pre_medication.rename(columns={'pre-index_medication:patient_id' : 'patient_id'}, inplace=True)
pre_medication.to_csv('pre_medication.csv', index=False)

#==================
cohort = pd.read_csv('cohort.csv')
index_vars = pd.read_csv('index_vars.csv')
index_records_crosstab = pd.read_csv('index_records_crosstab.csv')
index_consultations_crosstab = pd.read_csv('index_consultations_crosstab.csv')
index_labs_crosstab = pd.read_csv('index_labs_crosstab.csv')
pre_vars = pd.read_csv('pre_vars.csv')
pre_labs_crosstab = pd.read_csv('pre_labs_crosstab.csv')
pre_bmi_crosstab = pd.read_csv('pre_bmi_crosstab.csv')
pre_medication = pd.read_csv('pre_medication.csv')
features = cohort.merge(index_vars, on='patient_id', how='outer')
features = features.merge(index_records_crosstab, on='patient_id', how='outer')
features = features.merge(index_consultations_crosstab, on='patient_id', how='outer')
features = features.merge(index_labs_crosstab, on='patient_id', how='outer')
features = features.merge(pre_vars, on='patient_id', how='outer')
features = features.merge(pre_labs_crosstab, on='patient_id', how='outer')
features = features.merge(pre_bmi_crosstab, on='patient_id', how='outer')
features = features.merge(pre_medication, on='patient_id', how='outer')
features.to_csv('features.csv', index=False) # [4694 rows x 481 columns]