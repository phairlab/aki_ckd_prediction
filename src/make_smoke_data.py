#!/usr/bin/env python3
"""
Generate an internally consistent synthetic dataset for end-to-end testing.

Why this exists
---------------
`nonsense_data/` was produced by shuffling each column of the real data
independently (see create_faux_data.py). That destroys every relationship,
including the join key: a patient's row in features.csv does not correspond to
their rows in the lab extract. Running the pipeline against it drops all but a
couple of patients at the "missing baseline creatinine" step, so it cannot test
anything downstream of preprocessing.

This generator instead builds a small cohort that is *coherent*: patient ids
link across files, dates are ordered, units are plausible, and the outcome is
generated from a latent risk model so the models have real signal to find. It
is still entirely synthetic and contains no patient information.

It deliberately includes the messiness the pipeline has to survive:
  * the same analyte under several name strings, so lab normalization is
    exercised (editor point 5);
  * creatinine reported in umol/L and mg/dL, plus a few unrecognised units, so
    the unit converter is exercised;
  * missing values at realistic rates, so the fold-local imputer is exercised;
  * a mortality pattern, so the competing-risk analysis is exercised;
  * post-discharge outpatient eGFR, so the ascertainment analysis is exercised.

Usage
-----
    python src/make_smoke_data.py                 # 800 patients -> smoke_data/
    python src/make_smoke_data.py --n 2000 --seed 7
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import config


# Name variants for one analyte, mirroring the redundancy the editor described.
LAB_VARIANTS = {
    "creatinine":  ["Creatinine", "Creatinine, Bld", "CREATININE"],
    "egfr":        ["eGFR", "Glomerular Filtration Rate Estimate", "GFR Estimated"],
    "glucose":     ["Glucose", "Glucose Random", "Glucose, random", "Glucose Meter",
                    "Glucose, Bld", "Random Glucose"],
    "bicarbonate": ["HCO3", "HCO3 Calc Arterial", "HCO3, Calculated, Arterial",
                    "Bicarbonate, Bld", "Bicarbonate, Venous"],
    "hemoglobin":  ["Hemoglobin", "Total HGB", "Hemoglobin, Arterial"],
    "potassium":   ["Potassium", "Potassium, Bld"],
    "albumin":     ["Albumin"],
    "urate":       ["Urate", "Uric Acid"],
    "phosphate":   ["Phosphate"],
    "crp":         ["C-Reactive Protein (CRP)", "CRP"],
}

LAB_CATEGORY = {
    "creatinine": "Creatinine", "egfr": "eGFR", "glucose": "glucose",
    "bicarbonate": "Bicarbonate", "hemoglobin": "Hemoglobin",
    "potassium": "Potassium", "albumin": "Albumin", "urate": "uric acid",
    "phosphate": "phosphate", "crp": "C-reactive protein",
}

LAB_RANGES = {   # (mean, sd, unit)
    "egfr": (65, 25, "mL/min/1.73m2"),
    "glucose": (7.5, 3.0, "mmol/L"),
    "bicarbonate": (24, 4, "mmol/L"),
    "hemoglobin": (115, 20, "g/L"),
    "potassium": (4.2, 0.6, "mmol/L"),
    "albumin": (33, 6, "g/L"),
    "urate": (350, 100, "umol/L"),
    "phosphate": (1.2, 0.35, "mmol/L"),
    "crp": (40, 55, "mg/L"),
}

SERVICES = ["General Internal Medicine", "Cardiac Surgery", "Cardiology",
            "Nephrology", "Neurology", "General Surgery", "Critical Care", "?"]

GOALS = ["Acute care", "Community", "Transition", "Intensive care",
         "Perioperative", "?"]


def _solve_intercept(linear_predictor, target_rate, lo=-20.0, hi=20.0, tol=1e-6):
    """Bisect for the intercept giving the requested mean event probability."""
    for _ in range(200):
        mid = (lo + hi) / 2
        rate = float(np.mean(1 / (1 + np.exp(-(mid + linear_predictor)))))
        if abs(rate - target_rate) < tol:
            return mid
        if rate < target_rate:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def generate(n_patients=800, seed=1202, out_dir=None, event_rate=0.08):
    rng = np.random.default_rng(seed)
    out_dir = out_dir or os.path.join(config.PROJECT_ROOT, "smoke_data")
    os.makedirs(out_dir, exist_ok=True)

    ids = np.arange(100000, 100000 + n_patients)

    # ---- cohort ------------------------------------------------------------
    age = np.clip(rng.normal(64, 17, n_patients), 18, 98).round().astype(int)
    sex = rng.choice(["F", "M"], n_patients, p=[0.58, 0.42])
    admit = (pd.Timestamp("2020-01-01")
             + pd.to_timedelta(rng.integers(0, 730, n_patients), unit="D"))
    los = np.clip(rng.gamma(2.0, 7.0, n_patients), 3, 200).round().astype(int)
    discharge = admit + pd.to_timedelta(los, unit="D")

    # AKI stage, skewed toward stage 1 as in the real cohort
    highest_stage = rng.choice([1, 2, 3], n_patients, p=[0.76, 0.16, 0.08])

    baseline_cr = np.clip(rng.lognormal(np.log(0.95), 0.35, n_patients), 0.3, 6.0)
    rise = 1 + rng.gamma(1.5, 0.25, n_patients) * highest_stage / 2
    discharge_cr = np.clip(baseline_cr * rise, 0.3, 9.0)

    albuminuria = rng.choice(["unmeasured", "normal", "mild", "heavy"],
                             n_patients, p=[0.76, 0.12, 0.085, 0.035])

    # ---- latent risk -> outcome -------------------------------------------
    # Coefficients chosen so discrimination lands near the real cohort's AUROC,
    # which makes the smoke test a realistic exercise of the tuning search.
    alb_effect = pd.Series(albuminuria).map(
        {"unmeasured": 0.0, "normal": -0.2, "mild": 0.5, "heavy": 1.1}).to_numpy()
    linear_predictor = (0.030 * (age - 64)
                        + 0.95 * np.log(baseline_cr)
                        + 1.25 * np.log(discharge_cr)
                        + 0.35 * (highest_stage - 1)
                        + alb_effect
                        + 0.30 * (sex == "F")
                        + rng.normal(0, 0.55, n_patients))

    # Solve for the intercept that yields the requested event rate rather than
    # hand-tuning it: the linear predictor's scale changes whenever a
    # coefficient or a covariate distribution is edited, and an event rate that
    # silently drifts to 1% makes the smoke test useless for stratified CV.
    intercept = _solve_intercept(linear_predictor, event_rate)
    p_event = 1 / (1 + np.exp(-(intercept + linear_predictor)))
    ckd = (rng.random(n_patients) < p_event).astype(int)

    # Guarantee enough events for 10-fold stratified CV even on an unlucky draw.
    min_events = max(30, int(round(event_rate * n_patients * 0.6)))
    if ckd.sum() < min_events:
        deficit = min_events - int(ckd.sum())
        candidates = np.argsort(-p_event)
        for idx in candidates:
            if deficit <= 0:
                break
            if ckd[idx] == 0:
                ckd[idx] = 1
                deficit -= 1

    # ---- mortality ---------------------------------------------------------
    # Higher in progressors, matching the real cohort's 23.8% vs 9.1% split.
    p_death = np.where(ckd == 1, 0.24, 0.09)
    died = rng.random(n_patients) < p_death
    days_to_death = rng.integers(5, 500, n_patients)
    death_date = pd.Series(pd.NaT, index=range(n_patients))
    death_date[died] = (discharge[died]
                        + pd.to_timedelta(days_to_death[died], unit="D"))

    stage_dates, stage_values, stage_flags = {}, {}, {}
    for s in (1, 2, 3):
        reached = highest_stage >= s
        stage_flags[s] = reached.astype(int)
        offset = rng.integers(0, np.maximum(los, 1))
        d = pd.Series(pd.NaT, index=range(n_patients))
        d[reached] = admit[reached] + pd.to_timedelta(offset[reached], unit="D")
        stage_dates[s] = d
        v = np.where(reached, discharge_cr * 88.4 * rng.uniform(0.8, 1.1, n_patients),
                     np.nan)
        stage_values[s] = v

    cohort = pd.DataFrame({
        "id": ids,
        "AdmitDt": admit, "DischDt": discharge,
        "SEX": sex, "AGE_ADMIT": age, "total_los": los,
        "stage1": stage_flags[1], "stage1_dt": stage_dates[1], "stage1_value": stage_values[1],
        "stage2": stage_flags[2], "stage2_dt": stage_dates[2], "stage2_value": stage_values[2],
        "stage3": stage_flags[3], "stage3_dt": stage_dates[3], "stage3_value": stage_values[3],
        "higheststage": highest_stage,
        "death_date": death_date.values,
        "CKD_stage45": ckd,
        "stroke_af": rng.integers(0, 2, n_patients),
        "chf_af": rng.integers(0, 2, n_patients),
        "mi_af": rng.integers(0, 2, n_patients),
    })
    cohort.to_csv(os.path.join(out_dir, "cohort and outcome.csv"), index=False)

    # ---- in-hospital variables ---------------------------------------------
    def flag(p):
        return (rng.random(n_patients) < p).astype(int)

    in_vars = pd.DataFrame({
        "id": ids,
        "goals_of_care": rng.choice(GOALS, n_patients),
        "admission_services": rng.choice(SERVICES, n_patients),
        "covid_test_result": rng.choice(["Positive", "Negative", None],
                                        n_patients, p=[0.05, 0.75, 0.20]),
        "icu_inhosp": flag(0.25), "icu": flag(0.25),
        "cardiac_surgery": flag(0.12), "insulin": flag(0.30),
        "beta_blocker": flag(0.35), "smoke": flag(0.20),
        "Cardiac_catheterization": flag(0.10), "ami": flag(0.08),
        "chf": flag(0.18), "dialysis": flag(0.03),
        "Mechanical_ventilation": flag(0.15), "renal_ultralsound": flag(0.22),
        "angiogram": flag(0.09), "foley_catheter": flag(0.40),
        "OT_assessment": flag(0.30), "PT_assessment": flag(0.35),
        "obstructive_uropathy": flag(0.04), "sepsis": flag(0.14),
    })
    in_vars.to_csv(os.path.join(out_dir, "in-hosp vars.csv"), index=False)

    # ---- pre-index variables and medication --------------------------------
    pre_vars = pd.DataFrame({
        "id": ids,
        "covid_test_result": rng.choice(["Positive", 0], n_patients, p=[0.05, 0.95]),
        "chf_pre": flag(0.18), "pvd_pre": flag(0.07), "pud_pre": flag(0.05),
        "mild_liver_disease_pre": flag(0.09),
        "mild_severe_liver_disease_pre": flag(0.03),
        "cancer_pre": flag(0.20), "htn_pre": flag(0.52),
        "diab_pre": flag(0.33), "gout_pre": flag(0.06),
    })
    pre_vars.to_csv(os.path.join(out_dir, "pre-hosp vars.csv"), index=False)

    pre_med = pd.DataFrame({
        "id": ids,
        "AMINOGLYCOSIDE": flag(0.05), "AMPHOTERICIN_B": flag(0.01),
        "DIURETIC_K": flag(0.12), "DIURETIC_NONK": flag(0.28),
        "NSAIDS": flag(0.22), "PPI": flag(0.34),
        "SGLT2": flag(0.08), "VANCOMYCIN": flag(0.09),
    })
    pre_med.to_csv(os.path.join(out_dir, "pre-hosp medication.csv"), index=False)

    # ---- labs ---------------------------------------------------------------
    in_rows, pre_rows, post_rows = [], [], []

    for i in range(n_patients):
        pid, a, d = ids[i], admit[i], discharge[i]

        def add(rows, entity, value, date, unit=None, missing_unit=False):
            variants = LAB_VARIANTS[entity]
            name = variants[rng.integers(0, len(variants))]
            if missing_unit:
                unit = "arb.unit"
            rows.append({
                "id": pid, "TEST_NM": name,
                "TEST_RSLT": round(float(value), 3),
                "TEST_UOFM": unit,
                "lab_test_category": LAB_CATEGORY[entity],
                "test_date": date, "AdmitDt": a, "DischDt": d,
            })

        # Pre-admission creatinine (the James baseline window: 7-365 days)
        if rng.random() < 0.85:
            for _ in range(rng.integers(1, 4)):
                offset = int(rng.integers(7, 365))
                in_umol = rng.random() < 0.7
                value = baseline_cr[i] * (88.4 if in_umol else 1.0)
                value *= rng.uniform(0.92, 1.08)
                add(pre_rows, "creatinine", value,
                    a - pd.Timedelta(days=offset),
                    "umol/L" if in_umol else "mg/dL")

        # In-hospital creatinine, ending at the discharge value
        n_inhosp = int(rng.integers(2, 8))
        for j in range(n_inhosp):
            day = int(rng.integers(0, max(los[i], 1)))
            frac = j / max(n_inhosp - 1, 1)
            value = baseline_cr[i] + frac * (discharge_cr[i] - baseline_cr[i])
            in_umol = rng.random() < 0.7
            # A small fraction carry an unrecognised unit, to exercise the
            # converter's tally rather than silently mis-scaling.
            bad_unit = rng.random() < 0.02
            add(in_rows, "creatinine",
                value * (88.4 if in_umol else 1.0) * rng.uniform(0.95, 1.05),
                a + pd.Timedelta(days=day),
                "umol/L" if in_umol else "mg/dL", missing_unit=bad_unit)

        # The last in-hospital creatinine is the discharge value
        add(in_rows, "creatinine", discharge_cr[i] * 88.4, d, "umol/L")

        # Other analytes, in and before hospital
        for entity, (mean, sd, unit) in LAB_RANGES.items():
            for rows, lo, hi, base_date, sign in (
                    (in_rows, 0, max(los[i], 1), a, 1),
                    (pre_rows, 7, 365, a, -1)):
                if rng.random() < 0.55:
                    for _ in range(rng.integers(1, 4)):
                        offset = int(rng.integers(lo, hi))
                        value = max(rng.normal(mean, sd), 0.1)
                        add(rows, entity, value,
                            base_date + sign * pd.Timedelta(days=offset), unit)

        # Albuminuria: ACR or dipstick, matching the observed 24% measured rate
        if albuminuria[i] != "unmeasured":
            if rng.random() < 0.5:
                acr = {"normal": rng.uniform(0.2, 3.0),
                       "mild": rng.uniform(3.5, 33.0),
                       "heavy": rng.uniform(35, 300)}[albuminuria[i]]
                in_rows.append({
                    "id": pid, "TEST_NM": "Albumin/Creatinine Ratio",
                    "TEST_RSLT": round(acr, 2), "TEST_UOFM": "mg/mmol",
                    "lab_test_category": "Albumin/Creatinine Ratio",
                    "test_date": a - pd.Timedelta(days=int(rng.integers(0, 170))),
                    "AdmitDt": a, "DischDt": d})
            else:
                dip = {"normal": "NEGATIVE", "mild": "1+", "heavy": "3+"}[albuminuria[i]]
                in_rows.append({
                    "id": pid, "TEST_NM": "Protein, dipstick UA",
                    "TEST_RSLT": dip, "TEST_UOFM": None,
                    "lab_test_category": "dipstick UA",
                    "test_date": a + pd.Timedelta(days=int(rng.integers(0, max(los[i], 1)))),
                    "AdmitDt": a, "DischDt": d})

        # Post-discharge outpatient eGFR, for the ascertainment analysis.
        # Testing is deliberately NON-random: progressors and older patients
        # are more likely to be tested, which is exactly the selection the
        # editor is asking to be quantified.
        p_tested = 0.30 + 0.35 * ckd[i] + 0.10 * (age[i] > 70)
        if rng.random() < p_tested:
            for _ in range(rng.integers(1, 5)):
                day = int(rng.integers(30, 365))
                egfr_value = (rng.uniform(8, 29) if ckd[i] else rng.uniform(30, 100))
                post_rows.append({
                    "id": pid, "TEST_NM": LAB_VARIANTS["egfr"][rng.integers(0, 3)],
                    "TEST_RSLT": round(egfr_value, 1),
                    "TEST_UOFM": "mL/min/1.73m2",
                    "lab_test_category": "eGFR",
                    "test_date": d + pd.Timedelta(days=day),
                    "AdmitDt": a, "DischDt": d})

    pd.DataFrame(in_rows).to_csv(os.path.join(out_dir, "in-hosp labs.csv"), index=False)
    pd.DataFrame(pre_rows).to_csv(os.path.join(out_dir, "pre-hosp labs.csv"), index=False)
    pd.DataFrame(post_rows).to_csv(
        os.path.join(out_dir, "post-discharge labs.csv"), index=False)

    print(f"\nSynthetic smoke dataset -> {out_dir}")
    print(f"  patients                 : {n_patients:,}")
    print(f"  target event rate        : {event_rate * 100:.1f}%")
    print(f"  CKD 4-5 events           : {int(ckd.sum()):,} ({ckd.mean() * 100:.1f}%)")
    print(f"  died in follow-up        : {int(died.sum()):,} ({died.mean() * 100:.1f}%)")
    print(f"  in-hospital lab rows     : {len(in_rows):,}")
    print(f"  pre-index lab rows       : {len(pre_rows):,}")
    print(f"  post-discharge lab rows  : {len(post_rows):,} "
          f"({pd.DataFrame(post_rows)['id'].nunique() if post_rows else 0} patients tested)")
    print(f"  distinct raw lab names   : "
          f"{pd.DataFrame(in_rows + pre_rows)['TEST_NM'].nunique()}")
    print("\nRun the pipeline against it with:")
    print("  python run_pipeline.py --smoke --etl --tuning smoke --gpus cpu")
    return out_dir


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Generate coherent synthetic test data")
    p.add_argument("--n", type=int, default=800, help="Number of patients")
    p.add_argument("--seed", type=int, default=1202)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--event-rate", type=float, default=0.08,
                   help="Target outcome prevalence (default 0.08, near the real 6.1%%)")
    a = p.parse_args()
    generate(a.n, a.seed, a.out, a.event_rate)
