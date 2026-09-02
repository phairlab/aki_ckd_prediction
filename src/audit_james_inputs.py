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

    print(f"\n{'=' * 74}\nSUMMARY\n{'=' * 74}")
    clean = (result["acr_missed_rows"] == 0
             and result["creatinine_contaminant_rows"] == 0)
    if clean:
        print("  Both selections are clean. The submitted James scores stand, and")
        print("  this check can be cited in the response letter.")
    else:
        print("  At least one selection is wrong. The James score, Table 3's")
        print("  albuminuria distribution and every model using it are affected.")
        print("  Fix before running the re-analysis, not after.")
    for k, v in result.items():
        print(f"    {k:<32s} {v:,}")
    print(f"{'=' * 74}\n")


if __name__ == "__main__":
    main()
