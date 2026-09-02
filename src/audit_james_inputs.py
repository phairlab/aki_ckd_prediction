#!/usr/bin/env python3
"""
Audit the laboratory selection that feeds the James score.

Read-only. Changes nothing; measures whether two string-matching rules in
james_score_helpers are selecting the right tests on the real extract.

Why this exists
---------------
The server probe showed that the albumin:creatinine ratio is recorded under at
least four distinct TEST_NM strings ("ALBUMIN/CREATININE RATIO,URINE",
"Albumin / Creatinine Ratio", "Albumin Creatinine Ratio",
"Albumin/Creatinine, Urine"), and creatinine under several more. The James
score selects those tests by substring matching, which raises two questions
that matter for the reported numbers:

  1. ALBUMINURIA. `get_albuminuria_status` filters on `lab_test_category`
     containing "Albumin/Creatinine Ratio". Any ACR result whose CATEGORY is
     spelled differently is invisible, and that patient is scored "unmeasured"
     -- worth 1 point, the same as mild albuminuria. The manuscript reports
     76.0% unmeasured, which is high enough that a matching failure would be
     material to both Table 3 and every James score.

  2. CREATININE. `get_baseline_creatinine` and `get_discharge_creatinine` accept
     any TEST_NM containing "Creatinine" while excluding "Ratio|Protein|Albumin".
     A urine creatinine passes that filter. Urine creatinine runs roughly
     100x serum on the same molar scale, so even a handful leaking in would
     distort the baseline or discharge value for those patients and move them
     across James score bands.

Both questions are answered by counting, not by argument. If the counts are
zero, the submitted numbers stand and this can be cited in the response letter
as a check that was done.

Usage
-----
    python src/audit_james_inputs.py --server
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import config
from lab_normalization import add_canonical_name


ACR_CATEGORY_PATTERN = "Albumin/Creatinine Ratio"   # as used in get_albuminuria_status
DIPSTICK_CATEGORY_PATTERN = "dipstick UA"


def load_labs():
    in_hosp, pre_hosp = config.get_labs_paths()
    frames = []
    for label, path in (("in-hosp", in_hosp), ("pre-index", pre_hosp)):
        if path and os.path.exists(path):
            df = pd.read_csv(path, low_memory=False)
            df["__source"] = label
            frames.append(df)
    if not frames:
        raise SystemExit("No lab files found; check config paths.")
    labs = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(labs):,} lab rows from {len(frames)} file(s)")
    return add_canonical_name(labs)


# ---------------------------------------------------------------------------

def audit_albuminuria(labs, output_dir):
    """Compare what the category filter catches against what ACR actually is."""
    print(f"\n{'=' * 74}\n1. ALBUMINURIA SELECTION\n{'=' * 74}")

    cat = labs["lab_test_category"].fillna("")
    caught = cat.str.contains(ACR_CATEGORY_PATTERN, case=False, na=False)
    is_acr = labs["canonical_test"] == "albumin_creatinine_ratio"

    missed = labs[is_acr & ~caught]
    extra = labs[caught & ~is_acr]

    print(f"  rows the current category filter catches : {int(caught.sum()):,}")
    print(f"  rows that ARE an ACR by test name        : {int(is_acr.sum()):,}")
    print(f"  ACR rows the filter MISSES               : {len(missed):,}")
    print(f"  non-ACR rows the filter wrongly catches  : {len(extra):,}")

    if len(extra):
        # These carry an ACR category but are a different assay by name. They
        # matter because get_median_acr_category() takes the MEDIAN across
        # every row the category filter returns, so a contaminant on a
        # different scale moves the median and can flip a patient's
        # albuminuria band. The ACR thresholds are 3.39 and 33.9 mg/mmol;
        # serum albumin runs 30-50 g/L, which would read as "heavy".
        n_pat = extra["id"].nunique() if "id" in extra.columns else "?"
        print(f"\n  CONTAMINANTS: {len(extra):,} row(s) across {n_pat} patient(s) "
              f"carry an ACR category but are not an ACR by name:")
        summary = (extra.groupby(["TEST_NM", "canonical_test", "TEST_UOFM"],
                                 dropna=False)
                   .agg(n_rows=("TEST_NM", "size"))
                   .reset_index().sort_values("n_rows", ascending=False))
        for _, r in summary.iterrows():
            print(f"    {r['n_rows']:>7,}  TEST_NM={r['TEST_NM']!r}  "
                  f"-> {r['canonical_test']}  unit={r['TEST_UOFM']!r}")

        numeric = pd.to_numeric(extra["TEST_RSLT"], errors="coerce").dropna()
        acr_numeric = pd.to_numeric(
            labs.loc[is_acr, "TEST_RSLT"], errors="coerce").dropna()
        if len(numeric) and len(acr_numeric):
            print(f"\n  value distributions (ACR bands are 3.39 and 33.9 mg/mmol):")
            for label, s in (("true ACR", acr_numeric), ("contaminants", numeric)):
                print(f"    {label:<14s} n={len(s):>6,}  median={s.median():>9.2f}  "
                      f"IQR {s.quantile(.25):>8.2f}-{s.quantile(.75):<8.2f}  "
                      f"max={s.max():>10.2f}")
            share_heavy = float((numeric > 33.9).mean())
            print(f"\n    {share_heavy * 100:.1f}% of contaminant values exceed 33.9, "
                  f"the 'heavy albuminuria'\n    threshold worth 3 James points.")
            if share_heavy > 0.05:
                print("    That is high enough to change albuminuria bands. Restrict the")
                print("    ACR selection to canonical_test == 'albumin_creatinine_ratio'")
                print("    and re-check Table 3's albuminuria distribution.")
            else:
                print("    Low enough that the median is unlikely to shift a band, but")
                print("    confirm against the per-patient counts before relying on it.")
        summary.to_csv(os.path.join(output_dir, "james_audit_acr_contaminants.csv"),
                       index=False)

    if len(missed):
        n_pat = missed["id"].nunique() if "id" in missed.columns else "?"
        print(f"\n  MISSED ACR results affect {n_pat} patient(s). By name/category:")
        summary = (missed.groupby(["TEST_NM", "lab_test_category"], dropna=False)
                   .size().sort_values(ascending=False).reset_index(name="n_rows"))
        for _, r in summary.head(15).iterrows():
            print(f"    {r['n_rows']:>7,}  TEST_NM={r['TEST_NM']!r}  "
                  f"category={r['lab_test_category']!r}")
        print("\n  Each of these patients is currently scored 'unmeasured' (1 point)")
        print("  despite having an ACR result. That inflates the reported 76.0%")
        print("  unmeasured figure and mis-scores the James score for them.")
        summary.to_csv(os.path.join(output_dir, "james_audit_missed_acr.csv"),
                       index=False)
    else:
        print("\n  No ACR result is missed by the category filter. The 76.0%")
        print("  unmeasured figure is not a string-matching artifact.")

    # Dipstick, same question
    dip_caught = cat.str.contains(DIPSTICK_CATEGORY_PATTERN, case=False, na=False)
    is_dip = labs["canonical_test"] == "urine_dipstick"
    dip_missed = labs[is_dip & ~dip_caught]
    print(f"\n  dipstick rows caught by category filter  : {int(dip_caught.sum()):,}")
    print(f"  dipstick rows MISSED                     : {len(dip_missed):,}")
    if len(dip_missed):
        for name, n in (dip_missed.groupby("TEST_NM").size()
                        .sort_values(ascending=False).head(8).items()):
            print(f"    {n:>7,}  {name!r}")

    return {"acr_missed_rows": len(missed), "acr_extra_rows": len(extra),
            "dipstick_missed_rows": len(dip_missed)}


def audit_creatinine(labs, output_dir):
    """Check whether the serum-creatinine filter admits non-serum tests."""
    print(f"\n{'=' * 74}\n2. SERUM CREATININE SELECTION\n{'=' * 74}")

    name = labs["TEST_NM"].fillna("")
    # Exactly the filter used in james_score_helpers
    caught = (name.str.contains("Creatinine", case=False, na=False)
              & ~name.str.contains("Ratio|Protein|Albumin", case=False, na=False))
    is_serum = labs["canonical_test"] == "creatinine"

    contaminants = labs[caught & ~is_serum]

    print(f"  rows the current filter accepts as serum : {int(caught.sum()):,}")
    print(f"  rows that ARE serum creatinine           : {int(is_serum.sum()):,}")
    print(f"  NON-serum rows wrongly accepted          : {len(contaminants):,}")

    if len(contaminants):
        n_pat = contaminants["id"].nunique() if "id" in contaminants.columns else "?"
        print(f"\n  Wrongly accepted, affecting {n_pat} patient(s):")
        summary = (contaminants.groupby(["TEST_NM", "canonical_test", "TEST_UOFM"],
                                        dropna=False)
                   .size().sort_values(ascending=False).reset_index(name="n_rows"))
        for _, r in summary.head(15).iterrows():
            print(f"    {r['n_rows']:>7,}  {r['TEST_NM']!r}  "
                  f"-> {r['canonical_test']}  unit={r['TEST_UOFM']!r}")
        print("\n  These can be selected as a baseline or discharge creatinine.")
        print("  Urine creatinine runs far above serum on the same molar scale,")
        print("  so a single such row taken as the discharge value would push")
        print("  that patient into the top James score band.")
        summary.to_csv(os.path.join(output_dir, "james_audit_creatinine_contaminants.csv"),
                       index=False)
    else:
        print("\n  No non-serum test passes the filter. Baseline and discharge")
        print("  creatinine selection is clean.")

    missed = labs[is_serum & ~caught]
    print(f"\n  serum creatinine rows the filter misses  : {len(missed):,}")
    if len(missed):
        for nm, n in (missed.groupby("TEST_NM").size()
                      .sort_values(ascending=False).head(8).items()):
            print(f"    {n:>7,}  {nm!r}")

    return {"creatinine_contaminant_rows": len(contaminants),
            "creatinine_missed_rows": len(missed)}


def audit_units(labs):
    """Units seen on the tests that feed the score."""
    print(f"\n{'=' * 74}\n3. UNITS ON SCORE INPUTS\n{'=' * 74}")
    for entity in ("creatinine", "albumin_creatinine_ratio", "egfr"):
        sub = labs[labs["canonical_test"] == entity]
        if sub.empty:
            continue
        print(f"\n  {entity}  ({len(sub):,} rows)")
        for unit, n in (sub.groupby("TEST_UOFM", dropna=False).size()
                        .sort_values(ascending=False).head(8).items()):
            print(f"    {n:>8,}  {unit!r}")
    print("\n  creatinine_to_mg_dl() recognises umol/L, umol/L variants, mg/dL")
    print("  and mmol/L. Anything else is treated as missing and tallied.")


# ---------------------------------------------------------------------------
# 4. Completeness sweep
# ---------------------------------------------------------------------------
# Sections 1 and 2 compare two rules against each other. That catches a
# disagreement but it CANNOT prove completeness: a test recorded under a name
# neither rule recognises is invisible to both and shows up as zero.
#
# This section does not use the score's rules at all. It casts a deliberately
# over-broad net over every distinct test name in the extract, then reports
# which candidates the score's rules actually select. Over-broad is the point:
# false positives here cost a line of output, false negatives cost a wrong
# James score.

# Any of these makes a name a CANDIDATE for the albuminuria component.
BROAD_ALBUMINURIA_NET = (
    r"album",                    # albumin, microalbumin, albuminuria
    r"\balb\b|\balb\s*[/,]",    # ALB, ALB/CREAT
    r"\bmalb\b|\bualb\b|\bmau\b",
    r"/\s*creat",                # anything/creatinine, however spelled
    r"\ba\s*[/:]\s*c\b",        # A/C ratio, A:C
    r"\bacr\b",
    r"dipstick|urinalys|\bua\b",
    r"protein.*urine|urine.*protein",
)

# Units that essentially only appear on a ratio-type urine result.
ALBUMINURIA_UNIT_HINTS = ("mg/mmol", "mg/g", "g/mol", "mg/gcreat")


def completeness_sweep(labs, output_dir):
    """Enumerate every distinct test, then check what the score's rules select.

    Writes the full distinct-name inventory so it can be read by eye. That
    inventory is the actual evidence of completeness; everything else in this
    script is a consistency check.
    """
    print(f"\n{'=' * 74}\n4. COMPLETENESS SWEEP (does not use the score's rules)"
          f"\n{'=' * 74}")

    name = labs["TEST_NM"].fillna("").astype(str)
    cat = labs["lab_test_category"].fillna("").astype(str)
    unit = labs["TEST_UOFM"].fillna("").astype(str)
    haystack = name + " || " + cat

    candidate = pd.Series(False, index=labs.index)
    for pattern in BROAD_ALBUMINURIA_NET:
        candidate |= haystack.str.contains(pattern, case=False, regex=True, na=False)
    for hint in ALBUMINURIA_UNIT_HINTS:
        candidate |= unit.str.lower().str.contains(hint.lower(), regex=False, na=False)

    # What the score's own rules do with each candidate
    acr_selected = cat.str.contains(ACR_CATEGORY_PATTERN, case=False, na=False)
    dip_selected = cat.str.contains(DIPSTICK_CATEGORY_PATTERN, case=False, na=False)

    sub = labs[candidate].copy()
    sub["__acr_selected"] = acr_selected[candidate]
    sub["__dipstick_selected"] = dip_selected[candidate]

    grouped = (sub.groupby(["TEST_NM", "lab_test_category", "TEST_UOFM"], dropna=False)
               .agg(n_rows=("TEST_NM", "size"),
                    n_patients=("id", "nunique") if "id" in sub.columns
                               else ("TEST_NM", "size"),
                    canonical=("canonical_test", "first"),
                    used_as_acr=("__acr_selected", "max"),
                    used_as_dipstick=("__dipstick_selected", "max"))
               .reset_index()
               .sort_values("n_rows", ascending=False))

    grouped["used_by_score"] = grouped["used_as_acr"] | grouped["used_as_dipstick"]

    print(f"  distinct test names in the extract      : {name.nunique():,}")
    print(f"  candidates under the broad net          : {len(grouped):,} distinct "
          f"({int(candidate.sum()):,} rows)")
    print(f"  of those, USED by the albuminuria step  : "
          f"{int(grouped['used_by_score'].sum()):,} distinct")
    print(f"  of those, NOT used                      : "
          f"{int((~grouped['used_by_score']).sum()):,} distinct")

    print(f"\n  --- candidates the score DOES use ---")
    for _, r in grouped[grouped["used_by_score"]].iterrows():
        role = "ACR" if r["used_as_acr"] else "dipstick"
        print(f"    {r['n_rows']:>7,}  [{role:<8s}] {str(r['TEST_NM'])[:44]:<44s} "
              f"{str(r['TEST_UOFM'])[:12]:<12s} -> {r['canonical']}")

    not_used = grouped[~grouped["used_by_score"]]
    print(f"\n  --- candidates the score does NOT use (REVIEW THESE) ---")
    if not_used.empty:
        print("    none")
    else:
        for _, r in not_used.iterrows():
            print(f"    {r['n_rows']:>7,}  {str(r['TEST_NM'])[:44]:<44s} "
                  f"{str(r['TEST_UOFM'])[:12]:<12s} cat={str(r['lab_test_category'])[:24]:<24s} "
                  f"-> {r['canonical']}")
        print(f"\n    Each line is a test whose name or unit resembles an albuminuria")
        print(f"    measurement but which the score ignores. Most will be legitimately")
        print(f"    excluded -- serum albumin, urine protein, a protein:creatinine")
        print(f"    ratio. Any that is genuinely an albumin:creatinine ratio is a")
        print(f"    patient scored 'unmeasured' who should not have been.")

    path = os.path.join(output_dir, "james_audit_albuminuria_candidates.csv")
    grouped.to_csv(path, index=False)

    inventory = (labs.groupby(["TEST_NM", "lab_test_category", "TEST_UOFM"], dropna=False)
                 .size().reset_index(name="n_rows")
                 .sort_values("n_rows", ascending=False))
    inv_path = os.path.join(output_dir, "lab_test_inventory.csv")
    inventory.to_csv(inv_path, index=False)

    print(f"\n  candidates -> {path}")
    print(f"  FULL inventory of all {len(inventory):,} distinct "
          f"(name, category, unit) triples -> {inv_path}")
    print(f"  The inventory is the real completeness evidence: read it once and the")
    print(f"  question is settled for good.")

    return {"n_distinct_names": int(name.nunique()),
            "n_candidate_groups": int(len(grouped)),
            "n_candidates_used": int(grouped["used_by_score"].sum()),
            "n_candidates_unused": int((~grouped["used_by_score"]).sum())}


# ---------------------------------------------------------------------------
# 5. Would urine protein:creatinine change anyone's albuminuria band?
# ---------------------------------------------------------------------------
# The completeness sweep showed ~910 urine protein:creatinine (uPCR) results,
# which the score does not use. That exclusion is FAITHFUL to the published
# score -- James et al. define the albuminuria component on ACR or urine
# dipstick -- and keeping it is what preserves the like-for-like comparison the
# manuscript rests on.
#
# But KDIGO accepts uPCR as an alternative when ACR is unavailable, so a
# nephrology reviewer may reasonably ask why data on hand went unused. The
# answer should be a number, not a principle: how many patients are currently
# scored "unmeasured" -- worth 1 point, the same as mild -- while having a uPCR
# in the window?
#
# If that count is small the exclusion is immaterial and can be stated as such.
# If it is large, a sensitivity analysis adding uPCR is worth running.

ALBUMINURIA_WINDOW_DAYS_BEFORE_ADMIT = 180   # as in get_albuminuria_status


def audit_upcr_availability(labs, output_dir):
    """Cross-tabulate ACR / dipstick / uPCR availability per patient."""
    print(f"\n{'=' * 74}\n5. UNUSED URINE PROTEIN:CREATININE RESULTS\n{'=' * 74}")

    raw_dir = config.get_raw_data_dir()
    cohort_path = os.path.join(raw_dir, "cohort and outcome.csv")
    if not os.path.exists(cohort_path):
        cohort_path = os.path.join(raw_dir, "cohort.csv")
    if not os.path.exists(cohort_path):
        print("  SKIPPED: no cohort file found to supply admission dates.")
        return {}

    cohort = pd.read_csv(cohort_path, low_memory=False)
    id_col = "id" if "id" in cohort.columns else "patient_id"
    admit_col = "AdmitDt" if "AdmitDt" in cohort.columns else "admit_date"
    disch_col = "DischDt" if "DischDt" in cohort.columns else "discharge_date"

    cohort = cohort[[id_col, admit_col, disch_col]].copy()
    cohort.columns = ["id", "admit", "discharge"]
    cohort["id"] = cohort["id"].astype(str)
    cohort["admit"] = pd.to_datetime(cohort["admit"], errors="coerce")
    cohort["discharge"] = pd.to_datetime(cohort["discharge"], errors="coerce")

    work = labs.copy()
    work["id"] = work["id"].astype(str)
    work["test_date"] = pd.to_datetime(work["test_date"], errors="coerce")
    work = work.merge(cohort, on="id", how="inner")

    in_window = (
        (work["test_date"] >= work["admit"]
         - pd.Timedelta(days=ALBUMINURIA_WINDOW_DAYS_BEFORE_ADMIT))
        & (work["test_date"] <= work["discharge"])
    )
    win = work[in_window]

    def patients_with(entity):
        return set(win.loc[win["canonical_test"] == entity, "id"])

    acr = patients_with("albumin_creatinine_ratio")
    dip = patients_with("urine_dipstick")
    pcr = patients_with("protein_creatinine_ratio")
    everyone = set(cohort["id"])

    measured = acr | dip
    unmeasured = everyone - measured
    rescuable = unmeasured & pcr

    print(f"  cohort                                   : {len(everyone):,}")
    print(f"  with an ACR in the window                : {len(acr):,}")
    print(f"  with a dipstick in the window            : {len(dip):,}")
    print(f"  measured by the score (ACR or dipstick)  : {len(measured):,} "
          f"({len(measured) / len(everyone) * 100:.1f}%)")
    print(f"  scored 'unmeasured' (1 point)            : {len(unmeasured):,} "
          f"({len(unmeasured) / len(everyone) * 100:.1f}%)")
    print(f"  with a uPCR in the window                : {len(pcr):,}")
    print(f"\n  >>> scored 'unmeasured' BUT have a uPCR  : {len(rescuable):,} "
          f"({len(rescuable) / len(everyone) * 100:.2f}% of the cohort)")

    result = {
        "n_cohort": len(everyone),
        "n_with_acr": len(acr),
        "n_with_dipstick": len(dip),
        "n_measured_by_score": len(measured),
        "n_unmeasured": len(unmeasured),
        "n_with_upcr": len(pcr),
        "n_unmeasured_with_upcr": len(rescuable),
        "pct_unmeasured_with_upcr": round(
            100 * len(rescuable) / max(len(everyone), 1), 3),
    }

    if not rescuable:
        print("\n  No patient would gain albuminuria information from uPCR. The")
        print("  exclusion is immaterial and can be stated as such: uPCR results")
        print("  exist in the extract but only for patients who already had an ACR")
        print("  or a dipstick, so following the published score costs nothing.")
    else:
        share = len(rescuable) / max(len(unmeasured), 1)
        print(f"\n  That is {share * 100:.1f}% of the currently-unmeasured group.")
        if len(rescuable) / len(everyone) < 0.01:
            print("  Under 1% of the cohort. Report the number and note that the")
            print("  primary analysis follows the published score definition; a")
            print("  sensitivity analysis is unlikely to move anything.")
        else:
            print("  Large enough to be worth a sensitivity analysis. KDIGO accepts")
            print("  uPCR when ACR is unavailable, so these patients have albuminuria")
            print("  information that the score currently discards, and each is")
            print("  carrying the default 1 point instead of a measured band.")
            print("  Keep the published definition as primary -- it is what preserves")
            print("  comparability with James et al. and the Grampian validation --")
            print("  and report the uPCR-augmented version alongside it.")
        # Where would those patients actually land? Predicting this from the ACR
        # distribution is not good enough: a uPCR is ordered BECAUSE proteinuria
        # is suspected, so the tested group is enriched and the median may sit
        # well above the normal threshold.
        from james_score_helpers import (
            pcr_to_mg_mmol, PCR_NORMAL_THRESHOLD_MG_MMOL,
            PCR_HEAVY_THRESHOLD_MG_MMOL)

        pcr_rows = win[(win["canonical_test"] == "protein_creatinine_ratio")
                       & (win["id"].isin(rescuable))].copy()
        pcr_rows["__mg_mmol"] = [
            pcr_to_mg_mmol(v, u)
            for v, u in zip(pcr_rows["TEST_RSLT"], pcr_rows["TEST_UOFM"])]
        usable = pcr_rows.dropna(subset=["__mg_mmol"])

        per_patient = usable.groupby("id")["__mg_mmol"].median()
        n_unusable = len(rescuable) - len(per_patient)

        def band(v):
            if v < PCR_NORMAL_THRESHOLD_MG_MMOL:
                return "normal"
            if v <= PCR_HEAVY_THRESHOLD_MG_MMOL:
                return "mild"
            return "heavy"

        bands = per_patient.map(band)
        points = {"normal": 0, "mild": 1, "heavy": 3}

        print(f"\n  Where those patients would land (median uPCR per patient):")
        if len(per_patient):
            print(f"    median of medians : {per_patient.median():.2f} mg/mmol")
            print(f"    IQR               : {per_patient.quantile(.25):.2f} - "
                  f"{per_patient.quantile(.75):.2f}")
            print(f"    thresholds        : normal < "
                  f"{PCR_NORMAL_THRESHOLD_MG_MMOL:g}, heavy > "
                  f"{PCR_HEAVY_THRESHOLD_MG_MMOL:g} mg/mmol")
            print(f"\n    {'band':<8s} {'n':>6s} {'%':>7s}  points  "
                  f"James score change")
            for b in ("normal", "mild", "heavy"):
                n = int((bands == b).sum())
                delta = points[b] - 1     # 'unmeasured' is worth 1 point
                arrow = f"{delta:+d}" if delta else " 0 (no change)"
                print(f"    {b:<8s} {n:>6,} {n / len(bands) * 100:>6.1f}%  "
                      f"{points[b]:>6d}  {arrow}")
            if n_unusable:
                print(f"    {'(unusable units)':<8s} {n_unusable:>6,}")

            n_changed = int((bands != "mild").sum())
            print(f"\n    {n_changed:,} of {len(rescuable):,} would actually change "
                  f"score; the rest land in\n    'mild', which carries the same 1 point "
                  f"as 'unmeasured'.")
            net = float((bands.map(points) - 1).mean())
            print(f"    mean James score change across the 229: {net:+.2f} points")
        else:
            print("    none had a usable uPCR value after unit conversion")

        out = pd.DataFrame({"patient_id": per_patient.index,
                            "median_upcr_mg_mmol": per_patient.values,
                            "would_be_band": bands.values})
        out["james_points_change"] = out["would_be_band"].map(points) - 1
        out.to_csv(os.path.join(output_dir,
                                "james_audit_unmeasured_with_upcr.csv"), index=False)
        result["upcr_band_normal"] = int((bands == "normal").sum())
        result["upcr_band_mild"] = int((bands == "mild").sum())
        result["upcr_band_heavy"] = int((bands == "heavy").sum())
        result["upcr_median_mg_mmol"] = (float(per_patient.median())
                                         if len(per_patient) else None)

    return result


def main():
    p = argparse.ArgumentParser(description="Audit the James score's lab selection")
    source = p.add_mutually_exclusive_group()
    source.add_argument("--server", action="store_true")
    source.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    if args.smoke:
        config.USE_SMOKE_DATA, config.USE_NONSENSE_DATA = True, False
    elif args.server:
        config.USE_SMOKE_DATA, config.USE_NONSENSE_DATA = False, False

    output_dir = config.get_reports_dir()
    os.makedirs(output_dir, exist_ok=True)

    labs = load_labs()
    result = {}
    result.update(audit_albuminuria(labs, output_dir))
    result.update(audit_creatinine(labs, output_dir))
    audit_units(labs)
    result.update(completeness_sweep(labs, output_dir))
    result.update(audit_upcr_availability(labs, output_dir))

    print(f"\n{'=' * 74}\nSUMMARY\n{'=' * 74}")
    clean = (result["acr_missed_rows"] == 0
             and result["acr_extra_rows"] == 0
             and result["creatinine_contaminant_rows"] == 0)
    if clean:
        print("  Both selections are internally consistent. Note what that does and")
        print("  does not establish: sections 1 and 2 compare two rules against each")
        print("  other, so they detect disagreement, not omission. Section 4's")
        print("  candidate list and lab_test_inventory.csv are the completeness")
        print("  evidence -- read the 'does NOT use' list before citing this.")
    else:
        print("  At least one selection is wrong. The James score, Table 3's")
        print("  albuminuria distribution and every model using it are affected.")
        print("  Fix before running the re-analysis, not after.")
    for k, v in result.items():
        print(f"    {k:<32s} {v:,}")
    print(f"{'=' * 74}\n")


if __name__ == "__main__":
    main()
