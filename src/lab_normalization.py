"""
Clinical entity normalization for laboratory test names.

Motivation
----------
Routinely collected lab data records the same analyte under many name strings.
In this cohort, random glucose enters the candidate feature set under six
distinct strings for in-hospital measurements and ten for pre-index ones, and
bicarbonate is split across "HCO3", "HCO3 Calc Arterial", "Bicarbonate, Bld"
and others.  Because features.csv is built by crosstabbing on the raw name,
each string becomes its own column, replicated across the four temporal
aggregations (count / mean / min / max).

Two things go wrong.  Gain-based importance is split across the duplicates, so
a genuinely important analyte can rank below an unimportant one.  And both
feature-selection steps (RFE for XGBoost, SelectKBest for the transformer) are
degraded: the 100 selected columns can hold several copies of one test while
excluding a distinct analyte entirely.

Strategy
--------
The raw lab extracts already carry a `lab_test_category` column that groups
name variants -- 'Glucose Meter', 'Glucose' and 'Glucose, Random' all map to
'glucose'.  `alberta_score_helpers.get_albuminuria_status` already relies on
this column, so it is trusted elsewhere in the pipeline.  We use it as the
primary source of truth and fall back to explicit regex rules when it is
missing or blank.

Nothing here is silent.  `audit_lab_names()` writes a reviewable table of every
raw name and where it landed, so the mapping can be checked against the real
data on the server before any model is refitted.
"""

from __future__ import annotations

import os
import re
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Entities that must never be merged into a base analyte
# ---------------------------------------------------------------------------
# Serum creatinine, urine creatinine and the albumin:creatinine ratio all match
# /creatinine/, but they are different measurements on different scales.  Any
# name matching one of these is given its own canonical entity rather than
# being folded into the base analyte.

PROTECTED_RULES: tuple[tuple[str, str], ...] = (
    # The literal word "ratio" is NOT required. The AHS extract records the
    # albumin:creatinine ratio as "Albumin/Creatinine, Urine" 326 times, and
    # requiring "ratio" let that fall through to the urine-creatinine rule
    # below -- splitting one assay across two feature columns and inventing a
    # spurious creatinine_urine column. Same trap for "Protein/creatinine",
    # which was only rescued by its lab_test_category.
    # "creat" rather than "creatinine": the extract contains
    # "PROTEIN/CREAT RATIO,UR RANDOM", which requiring the full word sent to
    # protein_urine instead, splitting one assay across two feature columns.
    # "creat" is a prefix of "creatinine" so this covers both spellings.
    (r"(?:microalbumin|albumin|\balb)\s*[/,]?\s*creat",  "albumin_creatinine_ratio"),
    (r"\bacr\b",                                        "albumin_creatinine_ratio"),
    (r"protein\s*[/,]?\s*creat",                        "protein_creatinine_ratio"),
    (r"creatinine.*urine|urine.*creatinine",           "creatinine_urine"),
    (r"creatinine\s*clearance",                        "creatinine_clearance"),
    (r"dipstick|urinalysis|\bua\b",                    "urine_dipstick"),

    # Any other analyte measured on urine. Found in the real extract as
    # "EXT Glucose, Ur", which was being folded into blood glucose -- a urine
    # glucose is a different measurement on a different scale and does not
    # belong in the same feature column. ", Ur" and ", Urine" are the two
    # suffixes the warehouse uses.
    (r"(^|[\s,(])ur(ine)?([\s,)]|$)",                  "URINE_SPECIMEN"),
)


# ---------------------------------------------------------------------------
# Canonical entity rules, applied in order; first match wins
# ---------------------------------------------------------------------------
# Written against lowercased, whitespace-collapsed names.  Order matters:
# the more specific pattern must precede the more general one.

CANONICAL_RULES: tuple[tuple[str, str], ...] = (
    # --- kidney function -------------------------------------------------
    (r"glomerular\s*filtration|(^|[^a-z])egfr([^a-z]|$)|(^|[^a-z])gfr([^a-z]|$)", "egfr"),
    (r"creatinine",                                    "creatinine"),
    (r"\burea\b|blood\s*urea|\bbun\b",              "urea"),

    # --- acid-base -------------------------------------------------------
    (r"bicarb|hco3|co2\s*(content|total)",             "bicarbonate"),
    (r"anion\s*gap",                                   "anion_gap"),
    (r"\bpco2\b|carbon\s*dioxide\s*partial",         "pco2"),
    (r"\bpo2\b|oxygen\s*partial",                     "po2"),
    (r"lactate|lactic",                                "lactate"),

    # --- metabolic -------------------------------------------------------
    (r"hba1c|glycated|glycosylated\s*h",               "hba1c"),
    (r"glucose|glucometer|\bbgl\b",                    "glucose"),
    (r"\burate\b|uric\s*acid",                        "urate"),

    # --- inflammatory / cardiac ------------------------------------------
    # These precede the generic protein rule below: "C-Reactive Protein"
    # would otherwise be swallowed by /protein/.
    # High-sensitivity CRP before the general rule. They are different assays
    # with different dynamic ranges (hs-CRP is calibrated for cardiovascular
    # risk at 0-10 mg/L; standard CRP reports well past 300), so averaging them
    # into one column produces a number that is neither.
    (r"high\s*sensitivit|\bhs[\s-]*crp\b",             "crp_high_sensitivity"),
    (r"c[\s-]*reactive|\bcrp\b",                      "crp"),
    (r"procalcitonin",                                 "procalcitonin"),
    (r"troponin",                                      "troponin"),
    (r"\bbnp\b|natriuretic",                          "bnp"),
    (r"ferritin",                                      "ferritin"),

    # --- liver / protein --------------------------------------------------
    (r"\balbumin\b",                                   "albumin"),
    (r"bilirubin",                                     "bilirubin"),
    (r"\balt\b|alanine\s*amino",                      "alt"),
    (r"\bast\b|aspartate\s*amino",                    "ast"),
    (r"alkaline\s*phosphatase|\balp\b",               "alkaline_phosphatase"),
    (r"total\s*protein|\bprotein\b",                  "protein"),

    # --- haematology ------------------------------------------------------
    (r"ha?ematocrit|\bhct\b",                        "hematocrit"),
    (r"ha?emoglobin|\bhgb\b",                        "hemoglobin"),
    (r"platelet|\bplt\b",                              "platelets"),
    (r"white\s*(blood)?\s*cell|\bwbc\b|leu[ck]ocyte", "wbc"),
    (r"red\s*(blood)?\s*cell|\brbc\b",               "rbc"),
    (r"neutrophil",                                    "neutrophils"),
    (r"lymphocyte",                                    "lymphocytes"),
    (r"\binr\b|international\s*normali",             "inr"),
    (r"\bptt\b|thromboplastin",                       "ptt"),

    # --- electrolytes -----------------------------------------------------
    # Full words only.  Bare one- and two-letter abbreviations ("Na", "K",
    # "Cl") are handled by ABBREVIATION_RULES below, which require the whole
    # cleaned name to be the abbreviation -- matching them anywhere in a free
    # text string produces false hits ("NA" for not-available, the K in
    # "Vitamin K", the Cl in "Cl- Whole Blood" is fine but "Chloride, CSF"
    # is not the serum analyte).
    (r"\bsodium\b",                                    "sodium"),
    (r"potassium",                                     "potassium"),
    (r"chloride",                                      "chloride"),
    (r"calcium",                                       "calcium"),
    (r"magnesium",                                     "magnesium"),
    (r"phosph",                                        "phosphate"),

    # --- lipids -----------------------------------------------------------
    (r"\bhdl\b|cholesterol.*hdl",                     "hdl_cholesterol"),
    (r"\bldl\b|cholesterol.*ldl",                     "ldl_cholesterol"),
    (r"triglyceride",                                  "triglycerides"),
    (r"cholesterol",                                   "cholesterol_total"),

    # --- endocrine --------------------------------------------------------
    (r"\btsh\b|thyroid\s*stim",                      "tsh"),
    (r"\bpth\b|parathyroid",                          "pth"),

    # --- blood gas pH (last: /ph/ is a common substring) -------------------
    (r"^ph$|^ph[\s,(]|\bph\b.*(blood|arterial|venous|gas)", "ph"),
)


# Bare abbreviations, matched only when they constitute the ENTIRE cleaned
# name (optionally with a specimen qualifier).  Applied after CANONICAL_RULES.
ABBREVIATION_RULES: tuple[tuple[str, str], ...] = (
    (r"^na$|^na[\s,+-]", "sodium"),
    (r"^k$|^k[\s,+-]",   "potassium"),
    (r"^cl$|^cl[\s,+-]", "chloride"),
    (r"^ca$|^ca[\s,+-]", "calcium"),
    (r"^mg$|^mg[\s,+-]", "magnesium"),
    (r"^hb$|^hb[\s,]",   "hemoglobin"),
    (r"^po4$|^po4[\s,]", "phosphate"),
)


_WS = re.compile(r"\s+")


def _clean(text: object) -> str:
    """Lowercase, collapse whitespace, drop surrounding punctuation."""
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""
    return _WS.sub(" ", str(text).strip().lower())


def _slug(text: str) -> str:
    """Turn a free-text category into a stable identifier."""
    out = re.sub(r"[^a-z0-9]+", "_", _clean(text)).strip("_")
    return out or "unmapped"


def _base_entity(raw):
    """Canonical analyte for a name, ignoring the protected-specimen rules."""
    for pattern, entity in CANONICAL_RULES:
        if re.search(pattern, raw):
            return entity
    for pattern, entity in ABBREVIATION_RULES:
        if re.search(pattern, raw):
            return entity
    return None


def normalize_test_name(test_nm: object,
                        lab_test_category: object = None,
                        use_category: bool = True) -> str:
    """Map one raw lab test name to a canonical clinical entity.

    Parameters
    ----------
    test_nm : the raw TEST_NM string from the lab extract.
    lab_test_category : the extract's own grouping column, when present.
    use_category : if True (default) and a category is supplied, the category
        drives the result.  Set False to force the regex rules, which is useful
        for auditing whether the two agree.

    Returns
    -------
    A lowercase snake_case entity name.  Names that match nothing fall back to
    a slug of the raw string, so an unrecognised test becomes its own entity
    rather than being silently dropped or merged.

    Protected entities (urine creatinine, ACR, protein:creatinine ratio,
    creatinine clearance, dipstick urinalysis) are resolved before anything
    else and are never folded into a base analyte, whichever source is used.
    """
    raw = _clean(test_nm)

    # Protected checks run against the raw name in both modes: the extract's
    # category column is coarser than we need for the creatinine family, and
    # merging urine creatinine into serum creatinine would corrupt the
    # James score inputs as well as the expanded feature set.
    for pattern, entity in PROTECTED_RULES:
        if re.search(pattern, raw):
            if entity != "URINE_SPECIMEN":
                return entity
            # A urine specimen of some other analyte: resolve the analyte, then
            # keep it separate from its blood counterpart.
            base = _base_entity(raw)
            return f"{base}_urine" if base else "urine_other"

    if use_category:
        cat = _clean(lab_test_category)
        if cat:
            for pattern, entity in PROTECTED_RULES:
                if re.search(pattern, cat):
                    return entity
            for pattern, entity in CANONICAL_RULES:
                if re.search(pattern, cat):
                    return entity
            for pattern, entity in ABBREVIATION_RULES:
                if re.search(pattern, cat):
                    return entity
            return _slug(cat)

    if not raw:
        return "unmapped"

    for pattern, entity in CANONICAL_RULES:
        if re.search(pattern, raw):
            return entity

    for pattern, entity in ABBREVIATION_RULES:
        if re.search(pattern, raw):
            return entity

    return _slug(raw)


def add_canonical_name(labs_df: pd.DataFrame,
                       name_col: str = "TEST_NM",
                       category_col: str = "lab_test_category",
                       out_col: str = "canonical_test",
                       use_category: bool = True) -> pd.DataFrame:
    """Attach a canonical entity column to a lab DataFrame.

    Vectorised over the *distinct* (name, category) pairs rather than the rows,
    since a lab extract has millions of rows and a few hundred distinct names.
    """
    df = labs_df.copy()
    cat_series = (df[category_col] if category_col in df.columns
                  else pd.Series([None] * len(df), index=df.index))

    pairs = pd.DataFrame({"_n": df[name_col], "_c": cat_series}).drop_duplicates()
    lookup = {
        (n, c): normalize_test_name(n, c, use_category=use_category)
        for n, c in zip(pairs["_n"], pairs["_c"])
    }
    df[out_col] = [lookup[(n, c)] for n, c in zip(df[name_col], cat_series)]
    return df


# ---------------------------------------------------------------------------
# Unit consistency within an entity
# ---------------------------------------------------------------------------
# Collapsing name variants into one entity is only safe when they share a unit.
# The real extract contains two counter-examples, both small but both wrong to
# merge:
#
#   creatinine    34,770 rows in umol/L and 2 in mmol/L -- same quantity, but a
#                 1000x scale difference, and _prepare_labs does no conversion
#   urate_urine   mmol/d (24-hour excretion) and mmol/L (spot concentration) --
#                 not the same quantity at all, so no factor exists
#
# Before normalization each spelling had its own column, so nothing mixed; the
# merge is what introduced the hazard. Rather than build a conversion table for
# every analyte, an entity whose members disagree on units is SPLIT by unit.
# That costs a few sparse columns, which feature selection will ignore, and it
# cannot silently average incompatible scales.

def unit_slug(unit):
    """Stable, filename-safe token for a unit string."""
    text = _clean(unit)
    if not text:
        return "nounit"
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_") or "nounit"


def entity_unit_consistency(labs_df, name_col="TEST_NM",
                            unit_col="TEST_UOFM",
                            entity_col="canonical_test"):
    """Report which entities span more than one unit.

    Returns {entity: [unit, ...]} for entities with more than one distinct unit.
    """
    if unit_col not in labs_df.columns or entity_col not in labs_df.columns:
        return {}
    grouped = (labs_df.groupby(entity_col)[unit_col]
               .agg(lambda s: sorted({_clean(v) for v in s.dropna()} - {""})))
    return {entity: units for entity, units in grouped.items() if len(units) > 1}


def add_unit_aware_entity(labs_df, name_col="TEST_NM", unit_col="TEST_UOFM",
                          entity_col="canonical_test",
                          out_col="canonical_test_unit", verbose=True):
    """Attach an entity name that is split by unit only where units disagree.

    An entity reported in a single unit keeps its plain name, so the common
    case stays readable. One reported in several gets a suffix per unit.
    """
    df = labs_df.copy()
    mixed = entity_unit_consistency(df, name_col, unit_col, entity_col)

    if not mixed:
        df[out_col] = df[entity_col]
        return df, {}

    if verbose:
        noun = "entity spans" if len(mixed) == 1 else "entities span"
        print(f"[LabNorm] {len(mixed)} {noun} more than one unit and will be "
              f"SPLIT rather than averaged:")
        for entity, units in mixed.items():
            counts = (df.loc[df[entity_col] == entity, unit_col]
                      .fillna("").map(_clean).value_counts())
            detail = ", ".join(f"{u or 'nounit'}={n:,}" for u, n in counts.items())
            print(f"    {entity:<28s} {detail}")

    def label(row):
        entity = row[entity_col]
        if entity not in mixed:
            return entity
        return f"{entity}__{unit_slug(row[unit_col])}"

    df[out_col] = df.apply(label, axis=1)
    return df, mixed


# ---------------------------------------------------------------------------
# Unit harmonisation
# ---------------------------------------------------------------------------
# Splitting an entity by unit is safe but blunt: on the real extract it created
# 28 extra columns holding between 1 and 7 rows each. Where a conversion factor
# exists, converting is strictly better -- the measurement is kept and the
# column stays whole.
#
# Every factor below converts INTO the canonical unit and covers only units
# actually observed in this extract. Nothing is invented: an unrecognised unit
# for a listed entity is dropped and counted, never assumed.
#
#   entity -> (canonical unit, {unit: multiply by})

LAB_UNIT_CONVERSIONS = {
    # 1 mg/dL = 88.4 umol/L; 1 mmol/L = 1000 umol/L
    "creatinine": ("umol/l", {"umol/l": 1.0, "mmol/l": 1000.0, "mg/dl": 88.4}),
    # g(Hb)/dL and g/dL are the same thing; 1 g/dL = 10 g/L
    "hemoglobin": ("g/l", {"g/l": 1.0, "g/dl": 10.0, "g(hb)/dl": 10.0}),
    # 1 g/mmol = 1000 mg/mmol; 1 mg/mg = 1 g/g = 1000 mg/g = 1000/8.84 mg/mmol
    "protein_creatinine_ratio": ("mg/mmol", {"mg/mmol": 1.0, "g/mmol": 1000.0,
                                             "mg/mg": 1000.0 / 8.84,
                                             "mg/g": 1.0 / 8.84}),
    "albumin_creatinine_ratio": ("mg/mmol", {"mg/mmol": 1.0, "mg/g": 1.0 / 8.84}),
    # 1 mg/dL urate = 59.48 umol/L
    "urate": ("umol/l", {"umol/l": 1.0, "mg/dl": 59.48}),
    # 1 mg/dL glucose = 1/18.016 mmol/L
    "glucose": ("mmol/l", {"mmol/l": 1.0, "mg/dl": 1.0 / 18.016}),
    "bicarbonate": ("mmol/l", {"mmol/l": 1.0, "meq/l": 1.0}),
    "albumin": ("g/l", {"g/l": 1.0, "g/dl": 10.0}),
}

# Entities whose differing units measure DIFFERENT QUANTITIES, where no factor
# exists and splitting is the only correct answer. urate_urine appears as both
# mmol/d (24-hour excretion) and mmol/L (spot concentration).
NEVER_CONVERT = {"urate_urine", "creatinine_urine", "protein_urine",
                 "albumin_urine", "glucose_urine"}


def harmonise_units(labs_df, entity_col="canonical_test",
                    unit_col="TEST_UOFM", value_col="TEST_RSLT",
                    out_entity_col="canonical_test_unit", verbose=True):
    """Convert to a canonical unit per entity; split or drop when we cannot.

    Three outcomes per entity:
      * listed in LAB_UNIT_CONVERSIONS -- values converted into the canonical
        unit, and rows whose unit is not in its table are DROPPED and counted
      * listed in NEVER_CONVERT, or unlisted with several units -- SPLIT, with
        the unit appended to the entity name
      * unlisted with a single unit -- untouched

    Returns (dataframe, report).
    """
    df = labs_df.copy()
    df["__unit_clean"] = df[unit_col].map(_clean)
    df["__value"] = pd.to_numeric(df[value_col], errors="coerce")
    df[out_entity_col] = df[entity_col]

    report = {"converted": {}, "dropped": {}, "split": {}}
    drop_mask = pd.Series(False, index=df.index)

    for entity, group in df.groupby(entity_col):
        units = sorted(set(group["__unit_clean"]) - {""}) +                 (["<none>"] if (group["__unit_clean"] == "").any() else [])

        if entity in LAB_UNIT_CONVERSIONS and entity not in NEVER_CONVERT:
            canonical, factors = LAB_UNIT_CONVERSIONS[entity]
            for unit in set(group["__unit_clean"]):
                rows = group.index[group["__unit_clean"] == unit]
                factor = factors.get(unit)
                if factor is None:
                    drop_mask.loc[rows] = True
                    report["dropped"].setdefault(entity, {})[unit or "<none>"] = len(rows)
                elif factor != 1.0:
                    df.loc[rows, "__value"] = df.loc[rows, "__value"] * factor
                    report["converted"].setdefault(entity, {})[unit] = len(rows)
            continue

        if len(units) > 1:
            for unit in set(group["__unit_clean"]):
                rows = group.index[group["__unit_clean"] == unit]
                df.loc[rows, out_entity_col] = f"{entity}__{unit_slug(unit)}"
            report["split"][entity] = units

    if drop_mask.any():
        df = df[~drop_mask]

    df[value_col] = df["__value"]
    df = df.drop(columns=["__unit_clean", "__value"])

    if verbose:
        for entity, units in report["converted"].items():
            canonical = LAB_UNIT_CONVERSIONS[entity][0]
            detail = ", ".join(f"{u}={n:,}" for u, n in units.items())
            print(f"[LabNorm] {entity}: converted {detail} -> {canonical}")
        for entity, units in report["dropped"].items():
            detail = ", ".join(f"{u}={n:,}" for u, n in units.items())
            print(f"[LabNorm] {entity}: DROPPED {detail} "
                  f"(unit not interpretable for this analyte)")
        for entity, units in report["split"].items():
            print(f"[LabNorm] {entity}: SPLIT across {units} "
                  f"(different quantities, no conversion exists)")

    return df, report


def audit_lab_names(labs_df: pd.DataFrame,
                    name_col: str = "TEST_NM",
                    category_col: str = "lab_test_category",
                    unit_col: str = "TEST_UOFM",
                    id_col: str = "id",
                    source_label: str = "") -> pd.DataFrame:
    """Build a reviewable raw-name -> canonical-entity table.

    One row per distinct raw test name, carrying the volume behind it, the
    units it was reported in, and the entity it collapses into.  Review this on
    the server before refitting: a wrong merge here propagates into every
    expanded-feature result.

    The `n_raw_names_in_entity` column is the payload for the editor's point 5
    -- it says directly how many strings each analyte was split across.
    """
    df = add_canonical_name(labs_df, name_col, category_col)

    def _units(s):
        vals = sorted({str(v) for v in s.dropna().unique()})
        return "; ".join(vals[:6]) + (" ..." if len(vals) > 6 else "")

    agg = {"n_rows": (name_col, "size")}
    if id_col in df.columns:
        agg["n_patients"] = (id_col, "nunique")
    if unit_col in df.columns:
        agg["units"] = (unit_col, _units)

    grouped = (df.groupby(["canonical_test", name_col], dropna=False)
                 .agg(**agg)
                 .reset_index())

    if category_col in df.columns:
        cats = (df.groupby([ "canonical_test", name_col], dropna=False)[category_col]
                  .agg(lambda s: "; ".join(sorted({str(v) for v in s.dropna().unique()})[:4]))
                  .reset_index()
                  .rename(columns={category_col: "source_category"}))
        grouped = grouped.merge(cats, on=["canonical_test", name_col], how="left")

    sizes = grouped.groupby("canonical_test")[name_col].nunique()
    grouped["n_raw_names_in_entity"] = grouped["canonical_test"].map(sizes)
    if source_label:
        grouped.insert(0, "source", source_label)

    return grouped.sort_values(
        ["n_raw_names_in_entity", "canonical_test", "n_rows"],
        ascending=[False, True, False],
    ).reset_index(drop=True)


def write_lab_normalization_report(labs_frames: dict,
                                   output_dir: str,
                                   filename: str = "lab_normalization_audit.csv") -> str:
    """Write the audit table for one or more lab extracts.

    Parameters
    ----------
    labs_frames : {source_label: DataFrame}, e.g.
        {"in-hosp": index_labs_df, "pre-index": pre_labs_df}
    output_dir : where to write.

    Returns the path written.  Also prints the entities that were split across
    the most raw strings, which is the number the response letter needs.
    """
    os.makedirs(output_dir, exist_ok=True)
    audits = [audit_lab_names(df, source_label=label)
              for label, df in labs_frames.items() if df is not None and len(df)]
    if not audits:
        raise ValueError("No non-empty lab frames supplied to the audit")

    report = pd.concat(audits, ignore_index=True)
    path = os.path.join(output_dir, filename)
    report.to_csv(path, index=False)

    print(f"\n[LabNorm] Audit written to {path}")
    split = (report[report["n_raw_names_in_entity"] > 1]
             .groupby(["source", "canonical_test"])["TEST_NM"]
             .nunique().sort_values(ascending=False))
    if len(split):
        print("[LabNorm] Entities split across multiple raw name strings:")
        for (source, entity), n in split.head(20).items():
            print(f"    {source:<12s} {entity:<28s} {n} raw names")
        print(f"[LabNorm] {len(split)} entity/source pairs affected; "
              f"{int(split.sum() - len(split))} redundant columns removed by normalization.")
    else:
        print("[LabNorm] No entity was split across more than one raw name string.")

    return path


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
# The rules are ordered and regex-based, so a new rule can silently shadow an
# existing one (an early /protein/ swallowing "C-Reactive Protein", a
# /h(ae)?moglobin/ that matches "haemoglobin" but not "hemoglobin").  These
# cases are cheap to assert and expensive to discover in a refitted model.

SELF_TEST_CASES: tuple[tuple[str, str], ...] = (
    # spelling variants must converge
    ("Hemoglobin", "hemoglobin"),
    ("Haemoglobin", "hemoglobin"),
    ("Total HGB", "hemoglobin"),
    ("Hemoglobin, Arterial", "hemoglobin"),
    ("Hematocrit", "hematocrit"),
    ("Haematocrit", "hematocrit"),
    # the editor's two named examples
    ("Glucose Random", "glucose"),
    ("Glucose, random", "glucose"),
    ("Glucose Meter", "glucose"),
    ("Glucose, Bld", "glucose"),
    ("HCO3", "bicarbonate"),
    ("HCO3 Calc Arterial", "bicarbonate"),
    ("HCO3, Calculated, Arterial", "bicarbonate"),
    ("Bicarbonate, Bld", "bicarbonate"),
    ("Bicarbonate, Venous", "bicarbonate"),
    # eGFR variants
    ("eGFR", "egfr"),
    ("Glomerular Filtration Rate Estimate", "egfr"),
    # the creatinine family must stay separated
    ("Creatinine", "creatinine"),
    ("Creatinine, Urine", "creatinine_urine"),
    ("Albumin / Creatinine Ratio", "albumin_creatinine_ratio"),
    ("Protein / Creatinine Ratio, Urine", "protein_creatinine_ratio"),
    ("Creatinine Clearance", "creatinine_clearance"),
    # ordering hazards
    ("C-Reactive Protein (CRP)", "crp"),
    ("Total Protein", "protein"),
    ("HbA1c", "hba1c"),
    ("Albumin", "albumin"),
    ("dipstick UA", "urine_dipstick"),
    # Cases found in the real AHS extract, 2026-09-01
    ("EXT Glucose, Ur", "glucose_urine"),
    ("Glucose, Bld", "glucose"),
    ("EXT Glucose-Random-mmol/L", "glucose"),
    ("GLUCOSE RANDOM", "glucose"),
    ("C Reactive Protein High Sensitivity", "crp_high_sensitivity"),
    ("C Reactive Protein Quantitative", "crp"),
    ("Creatinine Serum", "creatinine"),
    ("EXT Creatinine", "creatinine"),
    ("ALBUMIN/CREATININE RATIO,URINE", "albumin_creatinine_ratio"),
    ("Albumin Creatinine Ratio", "albumin_creatinine_ratio"),
    ("Protein/Creatinine Ratio Conc, Urine", "protein_creatinine_ratio"),
    ("Phosphorus", "phosphate"),
    # An ACR without the word "ratio" in its name. 326 rows in the real
    # extract, unit mg/mmol, value distribution indistinguishable from the
    # named ACR results -- it is the same assay.
    ("Albumin/Creatinine, Urine", "albumin_creatinine_ratio"),
    ("Albumin/Creatinine Ratio", "albumin_creatinine_ratio"),
    ("Protein/creatinine", "protein_creatinine_ratio"),
    # Abbreviated forms present in the real extract
    ("PROTEIN/CREAT RATIO,UR RANDOM", "protein_creatinine_ratio"),
    ("EXT Microalbumin/Creatinine-mg/mmol", "albumin_creatinine_ratio"),
    ("Albumin/Creatinine Ratio, Urine", "albumin_creatinine_ratio"),
    ("Protein Urine UA", "urine_dipstick"),
    ("Protein/Creatinine, Urine, 24h", "protein_creatinine_ratio"),
    # ...and these must still stay distinct from it
    ("Creatinine, Urine", "creatinine_urine"),
    ("Creatinine", "creatinine"),
    ("Albumin", "albumin"),
    ("HCO3,VENOUS", "bicarbonate"),
    ("GFR ESTIMATED", "egfr"),
    ("EXT Hemoglobin (Hgb)-g/L", "hemoglobin"),
)


def run_self_test(verbose: bool = True) -> None:
    """Assert the ordering-sensitive cases above. Raises on the first failure."""
    failures = []
    for raw, expected in SELF_TEST_CASES:
        got = normalize_test_name(raw, None)
        if got != expected:
            failures.append(f"  {raw!r} -> {got!r}, expected {expected!r}")
    if failures:
        raise AssertionError(
            "Lab normalization self-test failed:\n" + "\n".join(failures) +
            "\n\nA rule is shadowing another. Check CANONICAL_RULES ordering."
        )
    if verbose:
        print(f"[LabNorm] Self-test passed ({len(SELF_TEST_CASES)} cases).")


if __name__ == "__main__":
    run_self_test()
