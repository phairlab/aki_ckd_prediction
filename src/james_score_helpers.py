'''
James Risk Score Calculator Helper Module

(Referred to as the "Alberta score" in earlier versions of this codebase.
The manuscript calls it the James score, after James et al. JAMA 2017.)
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
    # Explicit about stage 3 rather than folding every non-1, non-2 value into
    # 3 points: a missing or unexpected highest_stage previously scored 3, the
    # maximum, which silently inflated the James score for those patients.
    if stage == 1:
        return 0
    elif stage == 2:
        return 1
    elif stage == 3:
        return 3
    return np.nan


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



# ---------------------------------------------------------------------------
# Unit handling
# ---------------------------------------------------------------------------
# The original code checked only for 'umol/L' and otherwise assumed mg/dL, and
# get_discharge_creatinine returned the STRINGS "Invalid unit" /
# "Invalid test result" on failure.  Both were silent data-quality faults:
#
#   * a creatinine reported in any third unit was read as mg/dL, understating
#     it by ~88x and pushing the patient into the 0-point discharge band;
#   * a returned string is not NaN, so dropna(subset=["discharge_creatinine_raw"])
#     did NOT drop those patients.  They survived with a None points value,
#     which made the summed James score NaN for that patient.
#
# Conversions are now explicit, unknown units return None (so the patient is
# dropped by the existing dropna, matching the cohort diagram's "no valid
# creatinine measurement" exclusion), and every unrecognised unit is recorded
# in UNKNOWN_UNITS_SEEN so the count can be reported rather than discovered.

UNKNOWN_UNITS_SEEN = {}

# Creatinine molar mass factor: 1 mg/dL = 88.4 umol/L
_UMOL_PER_MGDL = 88.4

_UNIT_FACTORS_TO_MGDL = {
    "umol/l": 1.0 / _UMOL_PER_MGDL,
    "µmol/l": 1.0 / _UMOL_PER_MGDL,
    "mcmol/l": 1.0 / _UMOL_PER_MGDL,
    "umol/liter": 1.0 / _UMOL_PER_MGDL,
    "mg/dl": 1.0,
    "mg/100ml": 1.0,
    "mmol/l": 1000.0 / _UMOL_PER_MGDL,
}


def creatinine_to_mg_dl(value, unit):
    """Convert one creatinine result to mg/dL, or return None.

    Returns None (never a string, never a silently wrong number) when the
    value is non-numeric or the unit is not recognised.  Unrecognised units are
    tallied in UNKNOWN_UNITS_SEEN for reporting.
    """
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric):
        return None

    unit_key = str(unit).strip().lower() if unit is not None else ""
    if unit_key in _UNIT_FACTORS_TO_MGDL:
        return numeric * _UNIT_FACTORS_TO_MGDL[unit_key]

    # Tolerate suffixes such as 'umol/L (calc)' before giving up
    for known, factor in _UNIT_FACTORS_TO_MGDL.items():
        if unit_key.startswith(known):
            return numeric * factor

    UNKNOWN_UNITS_SEEN[unit_key] = UNKNOWN_UNITS_SEEN.get(unit_key, 0) + 1
    return None


def report_unknown_units():
    """Print and return the tally of unrecognised creatinine units."""
    if not UNKNOWN_UNITS_SEEN:
        print("[Units] All creatinine results carried a recognised unit.")
        return {}
    total = sum(UNKNOWN_UNITS_SEEN.values())
    print(f"[Units] {total} creatinine result(s) had an unrecognised unit and were "
          f"treated as missing:")
    for unit, n in sorted(UNKNOWN_UNITS_SEEN.items(), key=lambda kv: -kv[1]):
        print(f"    {unit!r}: {n}")
    return dict(UNKNOWN_UNITS_SEEN)


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
    
    # If no labs in window, look at the hospitalization
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
            value = creatinine_to_mg_dl(row['TEST_RSLT'], row['TEST_UOFM'])
            if value is not None:
                numeric_values.append(value)
        
        # If no valid numeric values, return None
        if not numeric_values:
            return None
            
        # Return the lowest in-hospital creatinine as baseline
        return min(numeric_values)
    
    # Sort window labs by test date (most recent first)
    sorted_window_labs = window_labs.sort_values('test_date', ascending=False)
    
    # Iterate through labs to find the first valid one
    for _, row in sorted_window_labs.iterrows():
        result_value = creatinine_to_mg_dl(row['TEST_RSLT'], row['TEST_UOFM'])
        if result_value is not None:
            return result_value
    
    # If no valid results found, return None
    return None


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
    
    # Most recent valid test before discharge.  Walk backwards rather than
    # taking only the single latest row: if that row carries an unparseable
    # value or an unrecognised unit, the next-most-recent measurement is a
    # better discharge creatinine than discarding the patient entirely.
    ordered = discharge_labs.sort_values('test_date', ascending=False)
    for _, row in ordered.iterrows():
        value = creatinine_to_mg_dl(row['TEST_RSLT'], row['TEST_UOFM'])
        if value is not None:
            return value

    return None



# calculate peak SCr, mg/dL
def get_peak_creatinine(patient_id, all_labs_df, index_admit_date, index_discharge_date):
    """
    Get the discharge creatinine value for a patient.
    
    This function finds the most recent inpatient creatinine measurement 
    after hospital admission and before hospital discharge.
    
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
        The peak creatinine value in mg/dL
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
    
    # Find the highest creatinine value before discharge
    # First, convert all values to mg/dL
    processed_values = []
    
    for _, row in discharge_labs.iterrows():
        result_value = creatinine_to_mg_dl(row['TEST_RSLT'], row['TEST_UOFM'])
        if result_value is not None:
            processed_values.append((row, result_value))
    
    # If no valid values, return None
    if not processed_values:
        return None
    
    # Find the row with the highest value
    highest_row, highest_value = max(processed_values, key=lambda x: x[1])
    
    # Return the highest creatinine value
    return highest_value



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


# ---------------------------------------------------------------------------
# Urine protein:creatinine ratio (sensitivity analysis only)
# ---------------------------------------------------------------------------
# The published James score defines the albuminuria component on ACR or urine
# dipstick, and the PRIMARY analysis keeps that definition -- it is what
# preserves comparability with James et al. and the Grampian validation.
#
# The audit found 229 patients (4.9% of the cohort, 6.4% of the unmeasured
# group) who are scored "unmeasured" -- worth 1 point, the same as mild -- while
# having a urine protein:creatinine ratio available. KDIGO accepts uPCR as an
# alternative when ACR is unavailable, so those patients hold albuminuria
# information the score discards. This supports a sensitivity analysis, enabled
# by config.ALBUMINURIA_INCLUDE_UPCR.
#
# Band equivalences are KDIGO's (2024 CKD guideline, albuminuria categories):
#
#     category                 ACR (mg/mmol)   PCR (mg/mmol)
#     A1 normal / mild             < 3             < 15
#     A2 moderately increased     3 - 30          15 - 50
#     A3 severely increased        > 30            > 50
#
# mapping onto the score's normal / mild / heavy. The ACR thresholds used
# elsewhere in this module are 3.39 and 33.9, which are 30 and 300 mg/g
# converted at 8.84; the PCR thresholds below are KDIGO's published values
# rather than a conversion of them.

PCR_NORMAL_THRESHOLD_MG_MMOL = 15.0
PCR_HEAVY_THRESHOLD_MG_MMOL = 50.0

# To mg/mmol. 1 mg/g = 1/8.84 mg/mmol; 1 mg/mg = 1 g/g = 1000 mg/g.
_PCR_UNIT_FACTORS = {
    "mg/mmol": 1.0,
    "g/mmol": 1000.0,
    "mg/g": 1.0 / 8.84,
    "mg/mg": 1000.0 / 8.84,
    "g/g": 1000.0 / 8.84,
    "mg/gcreat": 1.0 / 8.84,
}

UNKNOWN_PCR_UNITS_SEEN = {}


def pcr_to_mg_mmol(value, unit):
    """Convert one protein:creatinine result to mg/mmol, or return None."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric):
        return None

    key = str(unit).strip().lower() if unit is not None else ""
    if key in _PCR_UNIT_FACTORS:
        return numeric * _PCR_UNIT_FACTORS[key]
    for known, factor in _PCR_UNIT_FACTORS.items():
        if key.startswith(known):
            return numeric * factor

    UNKNOWN_PCR_UNITS_SEEN[key] = UNKNOWN_PCR_UNITS_SEEN.get(key, 0) + 1
    return None


def get_median_pcr_category(pcr_labs):
    """Albuminuria band from the median protein:creatinine ratio.

    Median rather than any single value, matching how ACR and dipstick results
    are already reduced in this module.
    """
    if pcr_labs is None or len(pcr_labs) == 0:
        return 'unmeasured'

    values = [pcr_to_mg_mmol(v, u)
              for v, u in zip(pcr_labs['TEST_RSLT'], pcr_labs['TEST_UOFM'])]
    values = [v for v in values if v is not None]
    if not values:
        return 'unmeasured'

    median = float(np.median(values))
    if median < PCR_NORMAL_THRESHOLD_MG_MMOL:
        return 'normal'
    if median <= PCR_HEAVY_THRESHOLD_MG_MMOL:
        return 'mild'
    return 'heavy'


def report_unknown_pcr_units():
    """Print and return the tally of unrecognised uPCR units."""
    if not UNKNOWN_PCR_UNITS_SEEN:
        return {}
    total = sum(UNKNOWN_PCR_UNITS_SEEN.values())
    print(f"[Units] {total} protein:creatinine result(s) had an unrecognised unit "
          f"and were treated as missing:")
    for unit, n in sorted(UNKNOWN_PCR_UNITS_SEEN.items(), key=lambda kv: -kv[1]):
        print(f"    {unit!r}: {n}")
    return dict(UNKNOWN_PCR_UNITS_SEEN)


# calculate albuminuria status
def get_albuminuria_status(patient_id, all_labs_df, index_admit_date,
                          index_discharge_date, include_upcr=False):
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

    # Protein:creatinine, used ONLY as a last resort and only when enabled.
    # Deliberately after dipstick rather than in KDIGO's own preference order
    # (ACR > PCR > reagent strip): placing it last confines the change to
    # patients who currently have no albuminuria measurement at all, which is
    # what makes the sensitivity analysis interpretable as "what if the
    # unmeasured group were rescued" rather than a different score.
    pcr_labs = patient_labs[
        patient_labs['lab_test_category'].str.contains(
            'Protein/creatinine', case=False, na=False)
        & (patient_labs['test_date'] >= lower_bound)
        & (patient_labs['test_date'] <= index_discharge_date)
    ] if include_upcr else patient_labs.iloc[0:0]

    # If no labs in window, return 'unmeasured'
    if acr_labs.empty and dipstick_labs.empty:
        if include_upcr and not pcr_labs.empty:
            return get_median_pcr_category(pcr_labs), acr_labs, dipstick_labs
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

    if include_upcr and not pcr_labs.empty:
        return get_median_pcr_category(pcr_labs), acr_labs, dipstick_labs
    return 'unmeasured', acr_labs, dipstick_labs
