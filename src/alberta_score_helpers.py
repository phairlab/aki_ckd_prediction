'''
Alberta Score Calculator Helper Module
The Alberta Score is calculated using:
1. Age (0-3 points)
2. AKI stage during hospitalization (0-3 points)
3. Baseline serum creatinine (0-5 points)
4. Discharge serum creatinine (0-11 points)
5. Albuminuria status (0-3 points)
Higher scores indicate greater risk of kidney disease progression. The module 
provides functions to:
- Map individual risk factors to point values
- Calculate baseline and discharge creatinine from lab data
- Determine albuminuria status using ACR or dipstick measurements

Steps to calculate the Alberta Score are defined in the following paper:
"Derivation and External Validation of Prediction Models for Advanced 
Chronic Kidney Disease Following Acute Kidney Injury"
https://jamanetwork.com/journals/jama/fullarticle/2662889
'''

import pandas as pd
import numpy as np
import re

# alberta score points for age
def age_mapping(age):
    if age < 50:
        return 0
    elif age < 60:
        return 1
    elif age < 90:
        return 2
    else:  # ≥ 90
        return 3


# alberta score points for highest stage
def stage_mapping(stage):
    if stage == 1:
        return 0
    elif stage == 2:
        return 1
    else:
        return 3


# alberta score points for baseline_creatinine
def baseline_creatinine_mapping(baseline_cr):
    if baseline_cr is not None and isinstance(baseline_cr, (int, float)):
        if baseline_cr < 0.6:
            return 0
        elif baseline_cr < 0.8:
            return 1
        elif baseline_cr < 1.0:
            return 2
        elif baseline_cr < 1.2:
            return 3
        elif baseline_cr < 1.3:
            return 4
        else:  # ≥ 1.3
            return 5
    else:
        return None  # Handle None or invalid values


# alberta score points for discharge_creatinine
def discharge_creatinine_mapping(discharge_cr):
    if discharge_cr is not None and isinstance(discharge_cr, (int, float)):
        if discharge_cr < 1.0:
            return 0
        elif discharge_cr < 1.3:
            return 3
        elif discharge_cr < 1.6:
            return 6
        elif discharge_cr < 1.9:
            return 7
        else:  # ≥ 1.9
            return 11
    else:
        return None  # Handle None or invalid values


# alberta score points for albuminuria_status
def albuminuria_status_mapping(albuminuria_status):
    if albuminuria_status == 'normal':
        return 0
    elif albuminuria_status == 'mild':
        return 1
    elif albuminuria_status == 'heavy':
        return 3
    else:  # unmeasured
        return 1



# calculate baseline SCr, mg/dL
def get_baseline_creatinine(patient_id, all_labs_df, index_admit_date, index_discharge_date=None):
    """
    Get the baseline creatinine value for a patient.
    
    This function finds the most recent outpatient creatinine measurement 
    between 7 and 365 days prior to the index hospitalization.
    
    Parameters:
    -----------
    patient_id : int or str
        The unique identifier for the patient
    all_labs_df : pandas.DataFrame
        DataFrame containing lab test results
    index_admit_date : datetime
        The admission date for the index hospitalization
    
    Returns:
    --------
    float
        The baseline creatinine value in mg/dL
    """
    # Filter labs for the specific patient
    patient_labs = all_labs_df[all_labs_df['id'] == patient_id]
    
    # Filter for creatinine tests only
    creatinine_labs = patient_labs[patient_labs['TEST_NM'].str.contains('Creatinine', case=False) & 
                                   ~patient_labs['TEST_NM'].str.contains('Ratio|Protein|Albumin', case=False)]
    
    # Calculate the time window (7-365 days before admission)
    lower_bound = index_admit_date - pd.Timedelta(days=365)
    upper_bound = index_admit_date - pd.Timedelta(days=7)
    
    # Filter for tests within the time window
    window_labs = creatinine_labs[(creatinine_labs['test_date'] >= lower_bound) & 
                                  (creatinine_labs['test_date'] <= upper_bound)]
    
    # If no labs in window, return None
    if window_labs.empty:
        # Make sure discharge date is available
        if index_discharge_date is None:
            return None
            
        # Look for in-hospital creatinine values
        inpatient_labs = creatinine_labs[(creatinine_labs['test_date'] >= index_admit_date) & 
                                        (creatinine_labs['test_date'] <= index_discharge_date)]
        
        # If no in-hospital labs either, return None
        if inpatient_labs.empty:
            return None
            
        # Process all values to numeric
        numeric_values = []
        for _, row in inpatient_labs.iterrows():
            try:
                value = float(row['TEST_RSLT'])
                # Convert to mg/dL if in umol/L
                if 'umol/L' in row['TEST_UOFM']:
                    value = value / 88.4
                numeric_values.append(value)
            except (ValueError, TypeError):
                continue
        
        # If no valid numeric values, return None
        if not numeric_values:
            return None
            
        # Return the lowest in-hospital creatinine as baseline
        return min(numeric_values)
    
    # Get the most recent test before admission
    most_recent = window_labs.sort_values('test_date', ascending=False).iloc[0]

    try:
        # Convert the creatinine value to mg/dL if it's in umol/L
        if 'umol/L' in most_recent['TEST_UOFM']:
            most_recent['TEST_RSLT'] = float(most_recent['TEST_RSLT']) / 88.4  # Convert to mg/dL
        elif 'mg/dL' in most_recent['TEST_UOFM']:
            most_recent['TEST_RSLT'] = float(most_recent['TEST_RSLT'])
        else:
            return "Invalid unit"
    except ValueError:
        return "Invalid test result"

    # Return the creatinine value
    return most_recent['TEST_RSLT']


# calculate discharge SCr, mg/dL
def get_discharge_creatinine(patient_id, all_labs_df, index_admit_date, index_discharge_date):
    """
    Get the discharge creatinine value for a patient.
    
    This function finds the most recent inpatient creatinine measurement 
    before hospital discharge.
    
    Parameters:
    -----------
    patient_id : int or str
        The unique identifier for the patient
    all_labs_df : pandas.DataFrame
        DataFrame containing lab test results
    index_discharge_date : datetime
        The discharge date for the index hospitalization
    
    Returns:
    --------
    float
        The discharge creatinine value in mg/dL
    """
    # Filter labs for the specific patient
    patient_labs = all_labs_df[all_labs_df['id'] == patient_id]
    
    # Filter for creatinine tests only
    creatinine_labs = patient_labs[patient_labs['TEST_NM'].str.contains('Creatinine', case=False) & 
                                   ~patient_labs['TEST_NM'].str.contains('Ratio|Protein|Albumin', case=False)]
    
    # Filter for tests on or after admission and before discharge
    discharge_labs = creatinine_labs[(creatinine_labs['test_date'] >= index_admit_date) & 
                                     (creatinine_labs['test_date'] <= index_discharge_date)]
    
    # If no labs before discharge, return None
    if discharge_labs.empty:
        return None
    
    # Get the most recent test before discharge
    most_recent = discharge_labs.sort_values('test_date', ascending=False).iloc[0]

    try:
        # Convert the creatinine value to mg/dL if it's in umol/L
        if 'umol/L' in most_recent['TEST_UOFM']:
            most_recent['TEST_RSLT'] = float(most_recent['TEST_RSLT']) / 88.4  # Convert to mg/dL
        elif 'mg/dL' in most_recent['TEST_UOFM']:
            most_recent['TEST_RSLT'] = float(most_recent['TEST_RSLT'])
        else:
            return "Invalid unit"
    except ValueError:
        return "Invalid test result"

    # Return the creatinine value
    return most_recent['TEST_RSLT']


# calculate albuminuria values
def classify_single_dipstick_result(result):
    """
    Classify a single dipstick result into normal, mild, or heavy albuminuria.
    
    Parameters:
    -----------
    result : str or float
        The dipstick test result
    
    Returns:
    --------
    str
        Category: 'normal', 'mild', or 'heavy'
    int
        Numeric equivalent: 0 for normal, 1 for mild, 2 for heavy
    """

    # Convert to string and uppercase for consistent processing
    result_str = str(result).upper().strip()
    
    # Check for unmeasured or uninterpretable
    if "UNABLE TO INTERPRET" in result_str or result_str == "" or result_str == "NAN":
        return None, None
    # Check for negative results
    if any(term in result_str for term in ["NEGATIVE", "NEG", "NEGATIVE URINALYSIS"]):
        return 'normal', 0
        
    # Check for trace results
    if "TRACE" in result_str:
        return 'mild', 1
        
    # Check for plus notation
    if "1+" in result_str:
        return 'mild', 1
    if any(term in result_str for term in ["2+", "3+", "4+"]):
        return 'heavy', 2
        
    # Check for numeric values
    try:
        # Extract numeric value if it exists
        numeric_matches = re.findall(r'(\d+\.?\d*)', result_str)
        if numeric_matches:
            value = float(numeric_matches[0])
            
            # Apply thresholds
            if ">=" in result_str:
                # For values like ">=3.0" or ">=5.0"
                if value >= 3.0:
                    return 'heavy', 2
                elif value >= 0.3:
                    return 'mild', 1
                else:
                    return 'normal', 0
            else:
                # For regular numeric values
                if value >= 3.0:
                    return 'heavy', 2
                elif value >= 0.3:
                    return 'mild', 1
                else:
                    return 'normal', 0
    except:
        pass
        
    # If we couldn't determine a category
    return None, None


def get_median_dipstick_category(dipstick_results):
    """
    Calculate the median category for a series of dipstick test results.
    
    Parameters:
    -----------
    dipstick_results : list or pandas.Series
        A collection of dipstick test results
    
    Returns:
    --------
    str
        The median category: 'normal', 'mild', 'heavy', or 'unmeasured'
    """
    # Clean input - remove None, NaN, empty strings
    if isinstance(dipstick_results, pd.Series):
        dipstick_results = dipstick_results.dropna().tolist()
    else:
        dipstick_results = [r for r in dipstick_results if r is not None and r != "" and not (isinstance(r, float) and np.isnan(r))]
    

    # If no valid results, return unmeasured
    if not dipstick_results:
        return 'unmeasured'
    
    # Classify each result
    categories = []
    category_values = []
    
    for result in dipstick_results:
        category, value = classify_single_dipstick_result(result)

        if category is not None and value is not None:
            categories.append(category)
            category_values.append(value)
    
    # If no valid categories, return unmeasured
    if not category_values:
        return 'unmeasured'
    
    # Calculate the median category value
    median_value = int(np.median(category_values))
    
    # Map the median value back to a category
    if median_value == 0:
        return 'normal'
    elif median_value == 1:
        return 'mild'
    else:  # median_value == 2
        return 'heavy'


def get_median_acr_category(acr_results):
    """
    Determine the albuminuria category based on ACR results.
    
    Parameters:
    -----------
    acr_results : list or pandas.Series, optional
        ACR measurements
    dipstick_results : list or pandas.Series, optional
        Dipstick test results
        
    Returns:
    --------
    str
        The albuminuria category: 'normal', 'mild', 'heavy', or 'unmeasured'
    """

    # normal_threshold = 30   # < 30 mg/g is normal
    # mild_threshold = 300    # 30-300 mg/g is mild

    # Set thresholds based on unit - https://www.scymed.com/en/smnxps/psdjb222_c.htm
    normal_threshold = 3.39  # < 3.4 mg/mmol is normal
    mild_threshold = 33.9     # 3.4-34 mg/mmol is mild
    
    # Check if we have ACR results (prioritize these)
    if acr_results is not None and len(acr_results) > 0:
        # print("We got some ACR results")

        acr_values = acr_results['TEST_RSLT'].tolist()    
        # print("ACR values:", acr_values)

        # Convert to numeric values where possible
        numeric_values = []
        for val in acr_values:
            try:
                if isinstance(val, str):
                    # Try to extract numeric value if it's a string
                    matches = re.findall(r'(\d+\.?\d*)', val)
                    if matches:
                        numeric_values.append(float(matches[0]))
                else:
                    numeric_values.append(float(val))
            except:
                continue

        # print(numeric_values)
        
        # If we have valid ACR values, calculate the median
        if numeric_values:
            median_acr = np.median(numeric_values)
            
            # Apply the ACR thresholds based on the unit
            if median_acr < normal_threshold:
                return 'normal'
            elif median_acr <= mild_threshold:
                return 'mild'
            else:
                return 'heavy'
    
    # print('No ACR results found')

    # If we don't have either ACR or dipstick results
    return 'unmeasured'


# calculate albuminuria status
def get_albuminuria_status(patient_id, all_labs_df, index_admit_date, index_discharge_date):
    """
    Determine the albuminuria status for a patient.
    
    This function checks urine albumin:creatinine ratio (ACR) or urine dipstick 
    measurements during or 6 months prior to index admission.
    
    Categories:
    - Normal: ACR < 30 mg/g or dipstick negative
    - Mild: ACR 30-300 mg/g or dipstick trace or 1+
    - Heavy: ACR > 300 mg/g or dipstick positive ≥2+
    
    Parameters:
    -----------
    patient_id : int or str
        The unique identifier for the patient
    all_labs_df : pandas.DataFrame
        DataFrame containing lab test results
    index_admit_date : datetime
        The admission date for the index hospitalization
    
    Returns:
    --------
    str
        The albuminuria category: 'normal', 'mild', 'heavy', or 'unmeasured'
    """
    # Filter labs for the specific patient
    patient_labs = all_labs_df[all_labs_df['id'] == patient_id]
    
    # Calculate the time window (6 months before admission to admission date)
    lower_bound = index_admit_date - pd.Timedelta(days=180)
    
    # Filter for ACR or albumin tests within the time window
    acr_pattern = 'Albumin/Creatinine Ratio'
    dipstick_pattern = 'dipstick UA'
    
    acr_labs = patient_labs[
        patient_labs['lab_test_category'].str.contains(acr_pattern, case=False) &
        (patient_labs['test_date'] >= lower_bound) &
        (patient_labs['test_date'] <= index_discharge_date)
    ]

    dipstick_labs = patient_labs[
        patient_labs['lab_test_category'].str.contains(dipstick_pattern, case=False) &
        (patient_labs['test_date'] >= lower_bound) &
        (patient_labs['test_date'] <= index_discharge_date)
    ]

    # If no labs in window, return 'unmeasured'
    if acr_labs.empty and dipstick_labs.empty:
        return 'unmeasured', acr_labs, dipstick_labs
    
    # Prioritize ACR measurements over dipstick
    if not acr_labs.empty:
        # Get the median of multiple measurements
        result = get_median_acr_category(acr_labs)
        if result != 'unmeasured':  # if we have a result
            # print('Using ACR')
            return result, acr_labs, dipstick_labs
        else:
            # Use dipstick if ACR not available
            if not dipstick_labs.empty:
                # print("Using dipstick")
                return get_median_dipstick_category(dipstick_labs['TEST_RSLT']), acr_labs, dipstick_labs

    else: # Use dipstick if ACR not available
        # print("Using dipstick")
        return get_median_dipstick_category(dipstick_labs['TEST_RSLT']), acr_labs, dipstick_labs

    return 'unmeasured', acr_labs, dipstick_labs
