#!/usr/bin/env python3
"""
Inventory the raw data on the secure server.

Run this FIRST, before the re-analysis. It answers three questions that
determine how much of the editor's letter can actually be addressed with the
data already in hand:

  1. Is there any post-discharge laboratory data?  (editor's point 3 --
     outcome ascertainment and loss to follow-up)  The nine files the ETL
     currently reads are all index-admission or pre-index, so if no follow-up
     eGFR exists anywhere on disk, a new AHS extract has to be requested, and
     that has a long lead time.

  2. How badly is the candidate feature set split across redundant lab name
     strings?  (editor's point 5)  Produces the exact counts for the response
     letter.

  3. Can the primary outcome be re-derived from lab-level data, or does it
     only exist as the precomputed `CKD_stage45` column?

Nothing is modified. Output is printed and written to reports/.

Usage
-----
    python src/probe_server_data.py --server
    python src/probe_server_data.py --nonsense       # smoke test locally
    python src/probe_server_data.py --server --raw-dir /some/other/path
"""

from __future__ import annotations

import os
import sys
import glob
import json
import argparse

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import config
from lab_normalization import add_canonical_name, audit_lab_names


def _preview(items, limit=12):
    """Short, readable rendering of a long column list.

    features.csv has 478 columns and the crosstab files have hundreds; printing
    them in full buries the numbers this script exists to report. The complete
    lists are in reports/server_data_probe.json and reports/feature_inventory.csv.
    """
    items = [str(i) for i in items]
    if len(items) <= limit:
        return str(items)
    shown = ", ".join(repr(i) for i in items[:limit])
    return f"[{shown}, ... +{len(items) - limit} more]"


DATE_HINTS = ("date", "dt", "_at")
EGFR_HINTS = ("egfr", "gfr", "glomerular")


def _is_datelike(col: str) -> bool:
    c = col.lower()
    return any(h in c for h in DATE_HINTS)


def _safe_read(path: str, nrows: int | None = None) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path, nrows=nrows, low_memory=False)
    except Exception as exc:                                  # noqa: BLE001
        print(f"    !! could not read: {type(exc).__name__}: {exc}")
        return None


# ---------------------------------------------------------------------------
# 1. File inventory
# ---------------------------------------------------------------------------

def inventory_files(raw_dir: str) -> list[dict]:
    """List every CSV in the raw directory with shape, columns and date span."""
    print(f"\n{'='*74}\n1. FILE INVENTORY: {raw_dir}\n{'='*74}")
    paths = sorted(glob.glob(os.path.join(raw_dir, "*.csv")))
    if not paths:
        print(f"  No CSV files found in {raw_dir}")
        return []

    records = []
    for path in paths:
        name = os.path.basename(path)
        size_mb = os.path.getsize(path) / 1e6
        print(f"\n  {name}  ({size_mb:,.1f} MB)")

        head = _safe_read(path, nrows=5)
        if head is None:
            continue

        # Row count without loading the whole file into memory
        with open(path, "rb") as f:
            n_rows = sum(1 for _ in f) - 1

        cols = list(head.columns)
        print(f"    rows: {n_rows:,}   cols: {len(cols)}")
        print(f"    columns: {_preview(cols)}")

        rec = {"file": name, "n_rows": n_rows, "n_cols": len(cols), "columns": cols,
               "size_mb": round(size_mb, 1)}

        date_cols = [c for c in cols if _is_datelike(c)]
        if date_cols:
            full = _safe_read(path)
            if full is not None:
                rec["date_ranges"] = {}
                for c in date_cols:
                    parsed = pd.to_datetime(full[c], errors="coerce")
                    if parsed.notna().any():
                        lo, hi = parsed.min(), parsed.max()
                        print(f"    {c}: {lo.date()} .. {hi.date()} "
                              f"({parsed.notna().sum():,} parsed / {len(parsed):,})")
                        rec["date_ranges"][c] = [str(lo.date()), str(hi.date())]
        records.append(rec)

    return records


# ---------------------------------------------------------------------------
# 2. Post-discharge lab availability  (editor's point 3)
# ---------------------------------------------------------------------------

def probe_followup_labs(raw_dir: str) -> dict:
    """Check every lab-like file for test dates after discharge.

    The outcome needs two outpatient eGFR values <30, the first drawn between
    30 days and one year post-discharge. If any file carries eGFR results in
    that window, the ascertainment analysis can be run on existing data.
    """
    print(f"\n{'='*74}\n2. POST-DISCHARGE LAB AVAILABILITY  (editor point 3)\n{'='*74}")

    lab_paths = [p for p in sorted(glob.glob(os.path.join(raw_dir, "*.csv")))
                 if "lab" in os.path.basename(p).lower()]
    if not lab_paths:
        print("  No files with 'lab' in the name found.")
        return {"any_post_discharge_egfr": False, "files": {}}

    out: dict = {"files": {}}
    any_post_egfr = False

    for path in lab_paths:
        name = os.path.basename(path)
        print(f"\n  {name}")
        df = _safe_read(path)
        if df is None:
            continue

        needed = {"test_date", "DischDt"}
        if not needed.issubset(df.columns):
            print(f"    skipped: needs {sorted(needed)}, has {_preview(df.columns, 8)}")
            continue

        test_date = pd.to_datetime(df["test_date"], errors="coerce")
        disch = pd.to_datetime(df["DischDt"], errors="coerce")
        admit = pd.to_datetime(df.get("AdmitDt"), errors="coerce")

        valid = test_date.notna() & disch.notna()
        days_post = (test_date - disch).dt.days

        post = valid & (days_post > 0)
        in_window = valid & (days_post >= 30) & (days_post <= 365)
        pre_admit = valid & admit.notna() & (test_date < admit)

        print(f"    total rows            : {len(df):,}")
        print(f"    before admission      : {int(pre_admit.sum()):,}")
        print(f"    after discharge       : {int(post.sum()):,}")
        print(f"    30-365 d post-discharge: {int(in_window.sum()):,}   <-- outcome window")

        file_rec = {
            "n_rows": len(df),
            "n_post_discharge": int(post.sum()),
            "n_in_outcome_window": int(in_window.sum()),
        }

        if "TEST_NM" in df.columns and int(in_window.sum()):
            sub = df.loc[in_window].copy()
            sub = add_canonical_name(sub)
            egfr = sub[sub["canonical_test"].isin(["egfr", "creatinine"])]
            n_pat = egfr["id"].nunique() if "id" in egfr.columns else None
            print(f"    of those, eGFR/creatinine: {len(egfr):,} rows"
                  + (f" across {n_pat:,} patients" if n_pat is not None else ""))
            file_rec["n_egfr_creatinine_in_window"] = int(len(egfr))
            file_rec["n_patients_with_egfr_in_window"] = int(n_pat) if n_pat else 0
            if len(egfr):
                any_post_egfr = True

        out["files"][name] = file_rec

    out["any_post_discharge_egfr"] = any_post_egfr

    print(f"\n  {'-'*70}")
    if any_post_egfr:
        print("  VERDICT: post-discharge eGFR/creatinine IS present.")
        print("  -> Set FOLLOWUP_LABS_PATH in config.py and run the ascertainment")
        print("     analysis (src/analysis/ascertainment.py). No new extract needed.")
    else:
        print("  VERDICT: NO post-discharge eGFR found in the current extracts.")
        print("  -> Editor point 3 cannot be answered from data on disk. Request an")
        print("     AHS extract of outpatient eGFR/creatinine for these patients,")
        print("     discharge date to +365 days, with test_date, value, unit and id.")
        print("     Start this request now; it is the long pole in the resubmission.")
    print(f"  {'-'*70}")

    return out


# ---------------------------------------------------------------------------
# 3. Lab name redundancy  (editor's point 5)
# ---------------------------------------------------------------------------

def probe_lab_redundancy(raw_dir: str, output_dir: str) -> dict:
    """Quantify how many raw name strings collapse into each clinical entity."""
    print(f"\n{'='*74}\n3. LAB NAME REDUNDANCY  (editor point 5)\n{'='*74}")

    frames = {}
    for label, fname in (("in-hosp", "in-hosp labs.csv"),
                         ("pre-index", "pre-hosp labs.csv")):
        path = os.path.join(raw_dir, fname)
        if not os.path.exists(path):                      # nonsense-data naming
            path = os.path.join(raw_dir, fname.replace(" ", "_"))
        if os.path.exists(path):
            df = _safe_read(path)
            if df is not None and "TEST_NM" in df.columns:
                frames[label] = df

    if not frames:
        print("  No lab files with a TEST_NM column found.")
        return {}

    os.makedirs(output_dir, exist_ok=True)
    summary = {}
    audits = []

    for label, df in frames.items():
        # Two different cuts, and reporting the wrong one gives the wrong
        # number for the response letter.
        #
        # BEFORE normalization the ETL kept the top N by RAW NAME (34 / 50), so
        # the same analyte under many spellings consumed several of those slots.
        # AFTER normalization it keeps the top N by ENTITY (config.TOP_LABS_*),
        # and each retained entity pulls in all of its name variants -- which is
        # why the entity-first cut sees MORE raw names, not fewer.
        old_top_n = 34 if label == "in-hosp" else 50
        new_top_n = (config.TOP_LABS_IN_HOSPITAL if label == "in-hosp"
                     else config.TOP_LABS_PRE_INDEX)

        old_kept = df[df["TEST_NM"].isin(
            list(df["TEST_NM"].value_counts().index)[:old_top_n])]
        n_raw_old = old_kept["TEST_NM"].nunique()

        with_entity = add_canonical_name(df)
        top_entities = list(
            with_entity["canonical_test"].value_counts().index)[:new_top_n]
        new_kept = with_entity[with_entity["canonical_test"].isin(top_entities)]
        n_raw_new = new_kept["TEST_NM"].nunique()
        n_entity = new_kept["canonical_test"].nunique()

        audit = audit_lab_names(new_kept, source_label=label)
        audits.append(audit)

        # features.csv holds four aggregations per column (count/mean/min/max)
        cols_before, cols_after = n_raw_old * 4, n_entity * 4

        print(f"\n  {label}")
        print(f"    BEFORE: top {old_top_n} by raw name -> {n_raw_old} names, "
              f"{cols_before} columns")
        print(f"    AFTER : top {new_top_n} by entity   -> {n_entity} entities "
              f"(spanning {n_raw_new} raw names), {cols_after} columns")
        print(f"    redundant columns removed    : {cols_before - cols_after}")
        print(f"    (the entity cut spans MORE raw names because each retained")
        print(f"     entity brings all its spellings with it)")
        n_raw, kept = n_raw_new, new_kept

        split = (audit[audit["n_raw_names_in_entity"] > 1]
                 .groupby("canonical_test")["TEST_NM"].nunique().sort_values(ascending=False))
        if len(split):
            print(f"    entities split across >1 string:")
            for entity, n in split.items():
                names = sorted(audit.loc[audit["canonical_test"] == entity, "TEST_NM"])
                print(f"      {entity:<24s} {n:>2d} strings: {_preview(names, 8)}")

        summary[label] = {
            "n_raw_names": int(n_raw),
            "n_entities": int(n_entity),
            "feature_cols_before": int(cols_before),
            "feature_cols_after": int(cols_after),
            "cols_removed": int(cols_before - cols_after),
            "entities_split": {k: int(v) for k, v in split.items()},
        }

    path = os.path.join(output_dir, "lab_normalization_audit.csv")
    pd.concat(audits, ignore_index=True).to_csv(path, index=False)
    print(f"\n  Full audit written to {path}")
    print("  REVIEW THIS FILE before refitting: a wrong merge propagates into")
    print("  every expanded-feature result.")

    return summary


# ---------------------------------------------------------------------------
# 4. Outcome derivability
# ---------------------------------------------------------------------------

def probe_outcome(raw_dir: str) -> dict:
    """Report how the primary outcome is represented in the cohort file."""
    print(f"\n{'='*74}\n4. OUTCOME REPRESENTATION\n{'='*74}")

    path = os.path.join(raw_dir, "cohort and outcome.csv")
    if not os.path.exists(path):
        path = os.path.join(raw_dir, "cohort.csv")
    if not os.path.exists(path):
        print("  No cohort file found.")
        return {}

    df = _safe_read(path)
    if df is None:
        return {}

    print(f"  {os.path.basename(path)}: {len(df):,} rows")
    outcome_cols = [c for c in df.columns
                    if "ckd" in c.lower() or "outcome" in c.lower() or "stage45" in c.lower()]
    print(f"  outcome-like columns: {outcome_cols}")

    rec: dict = {"file": os.path.basename(path), "n_rows": len(df),
                 "outcome_columns": outcome_cols}

    for c in outcome_cols:
        counts = df[c].value_counts(dropna=False).to_dict()
        print(f"    {c}: {counts}")
        rec[f"{c}_counts"] = {str(k): int(v) for k, v in counts.items()}

    egfr_cols = [c for c in df.columns if any(h in c.lower() for h in EGFR_HINTS)]
    date_cols = [c for c in df.columns if _is_datelike(c)]
    print(f"  eGFR-like columns : {_preview(egfr_cols) if egfr_cols else 'NONE'}")
    print(f"  date columns      : {_preview(date_cols)}")

    if "death_date" in df.columns:
        dd = pd.to_datetime(df["death_date"], errors="coerce")
        print(f"  death_date present: {dd.notna().sum():,} of {len(df):,} have a death date")
        rec["n_with_death_date"] = int(dd.notna().sum())
        print("  -> competing-risk analysis (editor point 2) is feasible from this file.")
    else:
        print("  death_date NOT present -- editor point 2 needs a mortality source.")

    if not egfr_cols:
        print("\n  NOTE: no per-test eGFR columns here, so the outcome exists only as the")
        print("  precomputed flag. Re-deriving it (and counting who was never tested)")
        print("  requires the follow-up lab extract probed in section 2.")

    return rec


# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Inventory raw data before the re-analysis")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--server", action="store_true", help="Use the secure-server paths")
    source.add_argument("--smoke", action="store_true",
                        help="Use the coherent synthetic dataset (src/make_smoke_data.py)")
    source.add_argument("--nonsense", action="store_true",
                        help="Use the legacy column-shuffled data")
    parser.add_argument("--raw-dir", type=str, default=None, help="Override the raw data directory")
    parser.add_argument("--output-dir", type=str, default=None, help="Where to write reports")
    args = parser.parse_args()

    if args.smoke:
        config.USE_SMOKE_DATA, config.USE_NONSENSE_DATA = True, False
    elif args.nonsense:
        config.USE_SMOKE_DATA, config.USE_NONSENSE_DATA = False, True
    elif args.server:
        config.USE_SMOKE_DATA, config.USE_NONSENSE_DATA = False, False

    raw_dir = args.raw_dir or config.get_raw_data_dir()
    output_dir = args.output_dir or config.get_etl_reports_dir()

    print(f"\nAKI-CKD data probe")
    print(f"raw data : {raw_dir}")
    print(f"reports  : {output_dir}")
    if not os.path.isdir(raw_dir):
        print(f"\nERROR: {raw_dir} is not a directory.")
        sys.exit(1)

    result = {
        "raw_dir": raw_dir,
        "inventory": inventory_files(raw_dir),
        "followup_labs": probe_followup_labs(raw_dir),
        "lab_redundancy": probe_lab_redundancy(raw_dir, output_dir),
        "outcome": probe_outcome(raw_dir),
    }

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "server_data_probe.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    print(f"\n{'='*74}")
    print(f"Probe complete. Machine-readable summary: {out_path}")
    print(f"{'='*74}\n")


if __name__ == "__main__":
    main()
