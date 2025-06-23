
import numpy as np
import pandas as pd
import sys

sys.path.append('/data/kidney/Sacha/aki_ckd_prediction/src')

from alberta_score_helpers import *


def preprocess_data(hing_features, 
                    alberta_features, 
                    alberta_score,
                    target):

    # --------------------------------------------------------
    # -*- load in and process the features.csv file -*-
    # --------------------------------------------------------

    # Construct the file path
    file_path = "/data/kidney/Hing/features.csv"

    # Load the CSV file into a DataFrame
    features_df = pd.read_csv(file_path)  # contains all the features that hing engineered

    # cast AdmitDt and DischDt to datetime
    features_df['admit_date'] = pd.to_datetime(features_df['admit_date'])
    features_df['discharge_date'] = pd.to_datetime(features_df['discharge_date'])

    # drop patients who died before they were discharged
    features_df.drop(features_df[features_df['death_date'] <= features_df['discharge_date']].index, inplace=True)

    if args.hing_features:
        # Calculate length of stay
        features_df['length_of_stay'] = (features_df['discharge_date'] - features_df['admit_date']).dt.days
        
        # For each stage, calculate time to either stage date or discharge (whichever comes first)
        features_df['admit_to_stage1_or_discharge'] = features_df.apply(
            lambda x: (pd.to_datetime(x['stage1_date']) - x['admit_date']).days 
            if pd.notna(x['stage1_date']) 
            else (x['discharge_date'] - x['admit_date']).days, 
            axis=1
        )
        
        features_df['admit_to_stage2_or_discharge'] = features_df.apply(
            lambda x: (pd.to_datetime(x['stage2_date']) - x['admit_date']).days 
            if pd.notna(x['stage2_date']) 
            else (x['discharge_date'] - x['admit_date']).days, 
            axis=1
        )
        
        features_df['admit_to_stage3_or_discharge'] = features_df.apply(
            lambda x: (pd.to_datetime(x['stage3_date']) - x['admit_date']).days 
            if pd.notna(x['stage3_date']) 
            else (x['discharge_date'] - x['admit_date']).days, 
            axis=1
        )

    # Calculate death within one year from discharge
    features_df['death_date'] = pd.to_datetime(features_df['death_date'])
    features_df['days_to_death'] = (features_df['death_date'] - features_df['discharge_date']).dt.days
    features_df['death_within_1yr'] = ((features_df['days_to_death'] <= 365) & (~pd.isna(features_df['death_date']))).fillna(False)
    features_df['ckd_or_death'] = features_df['ckd_stage45'] | features_df['death_within_1yr']

    # Delete death-related columns
    features_df = features_df.drop(['death_date', 'days_to_death', 'death_within_1yr'], axis=1)

    print("done loading features.csv")


    # --------------------------------------------------------
    # -*- load in the lab tests -*-
    # --------------------------------------------------------

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
    # -*- calculate alberta score features -*-
    # --------------------------------------------------------

    # sex
    features_df["sex_points"] = features_df.sex.map({1: 3, 0: 0})

    # age
    features_df["age_admit_points"] = features_df.age_admit.map(age_mapping)

    # highest stage
    features_df["highest_stage_points"] = features_df.highest_stage.map(stage_mapping)

    # baseline creatinine
    features_df['baseline_creatinine_raw'] = features_df.apply(
        lambda row: get_baseline_creatinine(row['patient_id'], all_labs_df, row['admit_date'], row['discharge_date']), axis=1
    )
    features_df["baseline_creatinine_points"] = features_df.baseline_creatinine_raw.map(baseline_creatinine_mapping)

    # discharge creatinine
    features_df['discharge_creatinine_raw'] = features_df.apply(
        lambda row: get_discharge_creatinine(row['patient_id'], all_labs_df, row['admit_date'], row['discharge_date']), axis=1
    )
    features_df["discharge_creatinine_points"] = features_df.discharge_creatinine_raw.map(discharge_creatinine_mapping)

    # albuminuria status
    features_df['albuminuria_status_raw'] = features_df.apply(
        lambda row: get_albuminuria_status(row['patient_id'], all_labs_df, row['admit_date'], row['discharge_date'])[0], axis=1
    )
    features_df["albuminuria_status_points"] = features_df.albuminuria_status_raw.map(albuminuria_status_mapping)

    print("done calculating alberta score points")


    # --------------------------------------------------------
    # -*- drop patients who don't have a baseline creatinine and discharge creatinine -*-
    # --------------------------------------------------------

    print(features_df.shape)

    features_df = features_df.dropna(subset=['baseline_creatinine_raw'])
    features_df = features_df.dropna(subset=['discharge_creatinine_raw'])

    print(features_df.shape)

    print("done dropping patients without baseline and discharge creatinine")


    # --------------------------------------------------------
    # -*- calculate alberta score 
    # --------------------------------------------------------

    features_df["alberta_score"] = features_df["sex_points"]+\
                                features_df["age_admit_points"]+\
                                features_df["highest_stage_points"]+\
                                features_df["baseline_creatinine_points"]+\
                                features_df["discharge_creatinine_points"]+\
                                features_df["albuminuria_status_points"]

    print("done calculating alberta score")


    # --------------------------------------------------------
    # -*- [OPTIONAL] remove columns with >90% null values -*-
    # --------------------------------------------------------

    remove_high_nulls = False

    if remove_high_nulls:
        # Print original shape
        print(f"Original features_df shape: {features_df.shape}")

        # Drop columns with extremely high null percentages (e.g., > 90%)
        high_null_cols = null_analysis[null_analysis['Null %'] > 90]['Feature'].tolist()
        if high_null_cols:
            print(f"\nRemoving {len(high_null_cols)} columns with >90% null values")
            # Make sure we only drop columns that actually exist in the DataFrame
            high_null_cols_existing = [col for col in high_null_cols if col in features_df.columns]
            if len(high_null_cols_existing) < len(high_null_cols):
                print(f"Note: Only {len(high_null_cols_existing)} out of {len(high_null_cols)} high-null columns exist in the DataFrame")
            features_df = features_df.drop(columns=high_null_cols_existing)

        # For remaining columns with moderate null percentages, we'll leave them as is
        # They will be handled during model training with imputation

        print(f"\nFinal features_df shape after cleaning: {features_df.shape}")


    # --------------------------------------------------------
    # -*- fill various missing values with appropriate defaults -*-
    # --------------------------------------------------------

    # ✅ fill nans in stage1 and stage2 with 0
    # fill stage1 and stage2 with 0
    features_df['stage1'] = features_df['stage1'].fillna(0)
    features_df['stage2'] = features_df['stage2'].fillna(0)
    features_df['stage3'] = features_df['stage3'].fillna(0)

    # ✅ for rows where stage1 is zero, impute missing stage1_creatinine values using alberta_df['baseline_creatinine_raw']
    # Converting from mg/dL to umol/L by multiplying by 88.4
    mask = features_df['stage1'] == 0
    features_df.loc[mask, 'stage1_creatinine'] = features_df.loc[mask, 'discharge_creatinine_raw'] * 88.4

    # ✅ for rows where stage2 is zero, impute stage2_creatinine values using stage1_creatinine
    mask = features_df['stage2'] == 0
    features_df.loc[mask, 'stage2_creatinine'] = features_df.loc[mask, 'stage1_creatinine']

    # ✅ for rows where stage3 is zero, impute stage2_creatinine values using stage1_creatinine
    mask = features_df['stage3'] == 0
    features_df.loc[mask, 'stage3_creatinine'] = features_df.loc[mask, 'stage2_creatinine']

    # ✅ fill record counts with 0
    record_columns = [col for col in features_df.columns if 'records_count' in col]
    features_df[record_columns] = features_df[record_columns].fillna(0)

    # ✅ fill means, mins, and maxes of record with the median of the column
    record_mean_cols = [col for col in features_df.columns if 'records_mean' in col]
    record_min_cols = [col for col in features_df.columns if 'records_min' in col]
    record_max_cols = [col for col in features_df.columns if 'records_max' in col]
    record_mean_medians = features_df[record_mean_cols].median()
    record_min_medians = features_df[record_min_cols].median()
    record_max_medians = features_df[record_max_cols].median()
    features_df[record_mean_cols] = features_df[record_mean_cols].fillna(record_mean_medians)
    features_df[record_min_cols] = features_df[record_min_cols].fillna(record_min_medians)
    features_df[record_max_cols] = features_df[record_max_cols].fillna(record_max_medians)

    # ✅ fill consult counts with 0
    consult_count_cols = [col for col in features_df.columns if 'consult_count' in col]
    features_df[consult_count_cols] = features_df[consult_count_cols].fillna(0)

    # ✅ fill lab counts with 0
    lab_count_cols = [col for col in features_df.columns if 'labs_count' in col]
    features_df[lab_count_cols] = features_df[lab_count_cols].fillna(0)

    # ✅ fill means, mins, and maxes of labs with the median of the column
    lab_mean_cols = [col for col in features_df.columns if 'labs_mean' in col]
    lab_min_cols = [col for col in features_df.columns if 'labs_min' in col]
    lab_max_cols = [col for col in features_df.columns if 'labs_max' in col]
    lab_mean_medians = features_df[lab_mean_cols].median()
    lab_min_medians = features_df[lab_min_cols].median()
    lab_max_medians = features_df[lab_max_cols].median()
    features_df[lab_mean_cols] = features_df[lab_mean_cols].fillna(lab_mean_medians)
    features_df[lab_min_cols] = features_df[lab_min_cols].fillna(lab_min_medians)
    features_df[lab_max_cols] = features_df[lab_max_cols].fillna(lab_max_medians)

    # ✅ fill pre-index lab counts with 0
    pre_index_labs_count_cols = [col for col in features_df.columns if 'pre-index_labs_count' in col]
    features_df[pre_index_labs_count_cols] = features_df[pre_index_labs_count_cols].fillna(0)

    # ✅ fill means, mins, and maxes of pre-index labs with the median of the column
    pre_index_labs_mean_cols = [col for col in features_df.columns if 'pre-index_labs_mean' in col]
    pre_index_labs_min_cols = [col for col in features_df.columns if 'pre-index_labs_min' in col]
    pre_index_labs_max_cols = [col for col in features_df.columns if 'pre-index_labs_max' in col]
    pre_index_labs_mean_medians = features_df[pre_index_labs_mean_cols].median()
    pre_index_labs_min_medians = features_df[pre_index_labs_min_cols].median()
    pre_index_labs_max_medians = features_df[pre_index_labs_max_cols].median()
    features_df[pre_index_labs_mean_cols] = features_df[pre_index_labs_mean_cols].fillna(pre_index_labs_mean_medians)
    features_df[pre_index_labs_min_cols] = features_df[pre_index_labs_min_cols].fillna(pre_index_labs_min_medians)
    features_df[pre_index_labs_max_cols] = features_df[pre_index_labs_max_cols].fillna(pre_index_labs_max_medians)

    # ✅ fill pre-index medication counts with 0
    pre_index_med_count_cols = [col for col in features_df.columns if 'pre-index_medication' in col]
    features_df[pre_index_med_count_cols] = features_df[pre_index_med_count_cols].fillna(0)

    # ✅ fill pre-index vars counts with 0
    pre_index_vars_count_cols = [col for col in features_df.columns if 'pre-index_vars' in col]
    features_df[pre_index_vars_count_cols] = features_df[pre_index_vars_count_cols].fillna(0)

    # ✅ fill pre-index counts with 0
    pre_index_count_cols = [col for col in features_df.columns if 'pre-index_count' in col]
    features_df[pre_index_count_cols] = features_df[pre_index_count_cols].fillna(0)

    # ✅ fill means, mins, and maxes of pre-index stuff with the median of the column
    pre_index_mean_cols = [col for col in features_df.columns if 'pre-index_mean' in col]
    pre_index_min_cols = [col for col in features_df.columns if 'pre-index_min' in col]
    pre_index_max_cols = [col for col in features_df.columns if 'pre-index_max' in col]
    pre_index_mean_medians = features_df[pre_index_mean_cols].median()
    pre_index_min_medians = features_df[pre_index_min_cols].median()
    pre_index_max_medians = features_df[pre_index_max_cols].median()
    features_df[pre_index_mean_cols] = features_df[pre_index_mean_cols].fillna(pre_index_mean_medians)
    features_df[pre_index_min_cols] = features_df[pre_index_min_cols].fillna(pre_index_min_medians)
    features_df[pre_index_max_cols] = features_df[pre_index_max_cols].fillna(pre_index_max_medians)


    # --------------------------------------------------------
    # -*- get final feature set -*-
    # --------------------------------------------------------

    # get list of feature columns to use
    if args.hing_features:  # use hing's features
        # Filter out columns we don't want to use as features
        # Define patterns to exclude
        excluded_patterns = ['_date', 'patient_id', 'after', 'ckd_or_death', 'ckd_stage45']
        
        # Add optional patterns based on args
        if not args.alberta_features: excluded_patterns.append('_points')
        if not args.alberta_score: excluded_patterns.append('alberta_score')

        # Get the categorical feature columns
        categorical_features = []

        # Inspect each column to identify categorical ones, excluding the ones with excluded patterns
        for col in features_df.columns:
            if not any(pattern in col for pattern in excluded_patterns):
                # Check if column contains categorical data or boolean values
                if features_df[col].dtype == 'object' or \
                features_df[col].dtype == 'bool' or \
                (features_df[col].dtype in ['int64', 'float64'] and len(features_df[col].unique()) < 10):
                    categorical_features.append(col)

        # One-hot encode the categorical features
        if categorical_features:
            features_df = pd.get_dummies(features_df, columns=categorical_features)
        
        # Get feature columns in one step using any() for pattern matching
        feature_columns = [col for col in features_df.columns 
                        if not any(pattern in col for pattern in excluded_patterns)]

    elif args.alberta_features:
        # Get features that have '_points' in their name
        feature_columns = [col for col in features_df.columns if '_points' in col]
        if args.alberta_score: feature_columns.append('alberta_score')

    else:  # use only the alberta score
        feature_columns = ['alberta_score']

    # Get index for the appropriate target attribute
    if args.target == 'ckdordeath': attributes_used = features_df['ckd_or_death'].values
    else: attributes_used = features_df['ckd_stage45'].values

    # create features_df with only the selected feature columns
    features_used = features_df[feature_columns].values
    feature_names = features_df[feature_columns].columns.values


    # --------------------------------------------------------
    # -*- return values -*-
    # --------------------------------------------------------
    return features_used, attributes_used, feature_names, features_df
