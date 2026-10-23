#!/usr/bin/env python3
"""
Pathology underperformance / information-density audit for completed outer_split_* artifacts.

This script mirrors the MRI underperformance-audit logic from the resume evaluator,
but applies it to pathology reports and frozen pathology recode provenance files.
It is API-free: it never calls SecureGPT, never rediscovers lexical features, and
never relearns split-specific lexicons. It only consumes completed split artifacts
and the original raw CSV containing pathology report text.

Core outputs
------------
1. pathology_audit_case_table_all_resumed_splits.csv
   Split-level pathology audit rows for every retained/deduplicated split.
2. pathology_audit_case_table_case_level.csv
   Case-level collapsed audit rows for plotting/statistics, to avoid repeatedly
   counting the same case when it appears in multiple split roles.
3. pathology_audit_density_summary_all_resumed_splits.csv
   Summary table stratified by dispersion_true_high_low, relapse_true, and split_role.
4. pathology_audit_high_low_metric_stats.csv
   High-vs-low dispersion statistical tests for every audit metric.
5. plots/
   One high-vs-low dispersion plot per metric plus a p-value overview plot.

Example
-------
python pathology_underperformance_audit.py \
  --split-root /Users/lukezhao/projects/onc/out/securegpt_dispersion_nested_path_informed_eval7 \
  --csv-path /Users/lukezhao/projects/onc/PROCESSED_TRIMMED_path_cases_MRI_status.csv

Optional, if auto-detection cannot find your pathology text column:
python pathology_underperformance_audit.py \
  --split-root /path/to/splits \
  --csv-path /path/to/raw.csv \
  --path-text-col pathology_report_text
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

DISPERSION_TRUE_HIGH_THRESHOLD = 85.0
RANDOM_SEED = 17
EPS = 1e-12

# Same shared concept ontology families used by the MRI audit, with pathology-relevant
# concepts retained. Non-mass enhancement remains here only so the code is robust if
# the shared ontology includes it in pathology recode outputs.
SPATIAL_MORPH_RESPONSE_GROUPS = {
    "extent_span",
    "multiplicity",
    "multicentricity_separate_sites",
    "distribution_linear_segmental_regional",
    "fragmentation_scattered_patchy_discontinuous",
    "residual_tumor_presence",
    "non_mass_enhancement",
    "invasive_disease",
    "in_situ_disease_dcis",
    "treatment_response",
    "treatment_effect_tumor_bed",
    "localized_compact_residual",
    "diffuse_scattered_residual",
    "lymphovascular_invasion",
    "margin_proximity",
}
DISTRIBUTION_GROUPS = {
    "multiplicity",
    "multicentricity_separate_sites",
    "distribution_linear_segmental_regional",
    "fragmentation_scattered_patchy_discontinuous",
    "diffuse_scattered_residual",
    "non_mass_enhancement",
}
AMBIGUITY_GROUPS = {
    "benign_or_nonspecific_enhancement",
}

# Pathology names are preferred, but the script can also write MRI-compatible aliases
# if you need downstream code expecting mri_report_words / mri_extraction_density.
PATH_AUDIT_METRICS = [
    "path_report_chars",
    "path_report_words",
    "path_report_missing",
    "has_findings_section",
    "has_impression_section",
    "has_comparison_section",
    "has_response_language",
    "has_measurement_language",
    "has_final_diagnosis_section",
    "has_synoptic_section",
    "has_gross_description_section",
    "has_microscopic_section",
    "has_margin_language",
    "has_lymph_node_language",
    "n_phrase_features",
    "n_phrase_present",
    "n_phrase_uncertain",
    "n_phrase_negated",
    "n_group_features",
    "n_group_present",
    "n_spatial_morph_response_phrases",
    "n_distribution_phrases",
    "n_ambiguity_phrases",
    "n_uncertain_group_mentions",
    "n_negated_group_mentions",
    "path_extraction_density",
    "distribution_density",
    "ambiguity_density",
    "uncertainty_density",
    "negation_density",
]

MRI_COMPATIBLE_ALIAS_MAP = {
    "path_report_chars": "mri_report_chars",
    "path_report_words": "mri_report_words",
    "path_report_missing": "mri_report_missing",
    "path_extraction_density": "mri_extraction_density",
}

RAW_PATH_TEXT_COLUMN_CANDIDATES = [
    "pathology_report_text",
    "pathology_text",
    "path_report_text",
    "path_text",
    "preop_pathology_text",
    "preop_path_text",
    "surgical_pathology_text",
    "surgical_path_text",
    "final_pathology_text",
    "pathology_report",
    "path_report",
    "pathology",
    "path",
    "trimmed_pathology_text",
    "trimmed_path_text",
]


@dataclass
class SplitArtifacts:
    split_dir: str
    split_id: str
    source_split_id: str
    mri_group_prov: Optional[pd.DataFrame]
    path_group_prov: Optional[pd.DataFrame]
    path_phrase_prov: Optional[pd.DataFrame]
    role_table: pd.DataFrame
    inventory: Dict[str, Any] = field(default_factory=dict)


# -----------------------------------------------------------------------------
# Generic utilities
# -----------------------------------------------------------------------------

def safe_read_csv(path: Optional[str]) -> Optional[pd.DataFrame]:
    if not path or not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        print(f"[WARN] Failed reading CSV: {path}. Error={exc}")
        return None


def find_one(patterns: Sequence[str], required: bool = False, label: str = "file") -> Optional[str]:
    matches: List[str] = []
    for pattern in patterns:
        matches.extend(glob.glob(pattern))
    matches = sorted(set(matches))
    if not matches:
        if required:
            raise FileNotFoundError(f"Required {label} not found. Tried patterns: {patterns}")
        return None
    matches = sorted(matches, key=lambda p: (os.path.getmtime(p), p), reverse=True)
    return matches[0]


def normalize_text(s: Any) -> str:
    s = str(s or "").lower()
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def word_count(s: Any) -> int:
    return len([w for w in re.split(r"\s+", str(s or "").strip()) if w])


def stable_case_hash(case_ids: Sequence[str]) -> str:
    payload = "\n".join(sorted(map(str, case_ids))).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:16]


def infer_source_split_id(split_dir: str, csv_paths: Sequence[Optional[str]]) -> str:
    for path in csv_paths:
        if not path:
            continue
        base = os.path.basename(path)
        m = re.search(r"(outer_split_\d{3}(?:_\d+)?)", base)
        if m:
            return m.group(1)
    return os.path.basename(os.path.abspath(split_dir))


def coerce_numeric_targets(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "case_id" in out.columns:
        out["case_id"] = out["case_id"].astype(str)
    for col in ["row_index", "dispersion_true", "dispersion_true_high_low", "relapse_true"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    if "dispersion_true_high_low" in out.columns:
        out["dispersion_true_high_low"] = out["dispersion_true_high_low"].round().astype("Int64")
    return out


def numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce")


def sanitize_filename(s: str, max_len: int = 160) -> str:
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s)).strip("_")
    return s[:max_len] if len(s) > max_len else s


def first_existing_col(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    lower_to_original = {str(c).lower(): c for c in df.columns}
    for cand in candidates:
        if cand in df.columns:
            return cand
        if cand.lower() in lower_to_original:
            return lower_to_original[cand.lower()]
    return None


# -----------------------------------------------------------------------------
# Split artifact reading and deduplication
# -----------------------------------------------------------------------------

def build_role_table_from_provenance(
    mri_prov: Optional[pd.DataFrame],
    path_prov: Optional[pd.DataFrame],
    split_id: str,
) -> pd.DataFrame:
    """Recover train/test membership from frozen recode provenance.

    The same principle as the MRI resume evaluator is used here: split_role in
    frozen provenance is authoritative. Pathology provenance is sufficient, but
    MRI provenance is also consulted when present to detect incompatible mixing.
    """
    tables: List[pd.DataFrame] = []
    for modality, prov in [("mri", mri_prov), ("path", path_prov)]:
        if prov is None or len(prov) == 0:
            continue
        required_cols = {"case_id", "row_index", "split_role"}
        missing = required_cols - set(prov.columns)
        if missing:
            print(f"[WARN] {split_id} {modality} provenance missing columns: {sorted(missing)}")
            continue
        keep_cols = ["case_id", "row_index", "split_role"]
        optional = [c for c in ["dispersion_true", "dispersion_true_high_low", "relapse_true"] if c in prov.columns]
        tmp = prov[keep_cols + optional].drop_duplicates().copy()
        tmp["role_source_modality"] = modality
        tables.append(tmp)
    if not tables:
        raise ValueError(
            f"{split_id}: cannot recover train/test membership. Expected recode provenance with split_role."
        )
    role_long = pd.concat(tables, ignore_index=True)
    role_long["case_id"] = role_long["case_id"].astype(str)
    role_long["split_role"] = role_long["split_role"].astype(str).str.lower()
    role_long = role_long[role_long["split_role"].isin(["train", "test"])].copy()
    if len(role_long) == 0:
        raise ValueError(f"{split_id}: no valid train/test split_role values found.")

    conflict = role_long.groupby(["case_id", "row_index"])["split_role"].nunique().reset_index(name="n_roles")
    conflict = conflict[conflict["n_roles"] > 1]
    if len(conflict):
        raise ValueError(
            f"{split_id}: conflicting split_role assignments for {len(conflict)} cases. "
            "This suggests artifacts from incompatible split definitions were mixed."
        )

    role = (
        role_long.sort_values(["case_id", "row_index", "role_source_modality"])
        .groupby(["case_id", "row_index"], as_index=False)
        .first()
    )
    role["split_id"] = split_id

    # If the provenance did not carry the binary label but has continuous dispersion, derive it.
    if "dispersion_true_high_low" not in role.columns and "dispersion_true" in role.columns:
        role["dispersion_true_high_low"] = (
            pd.to_numeric(role["dispersion_true"], errors="coerce") >= DISPERSION_TRUE_HIGH_THRESHOLD
        ).astype("Int64")
    return coerce_numeric_targets(role)


def discover_split_dirs(split_root: Optional[str], explicit_split_dirs: Optional[Sequence[str]]) -> List[str]:
    dirs: List[str] = []
    if split_root:
        root = os.path.abspath(split_root)
        dirs.extend(sorted(glob.glob(os.path.join(root, "outer_split*"))))
    if explicit_split_dirs:
        dirs.extend([os.path.abspath(d) for d in explicit_split_dirs])
    dirs = [d for d in sorted(set(dirs)) if os.path.isdir(d)]
    if not dirs:
        raise ValueError("No split directories found. Provide --split-root or --split-dirs.")
    return dirs


def read_completed_split(split_dir: str) -> SplitArtifacts:
    split_dir = os.path.abspath(split_dir)
    split_id = os.path.basename(split_dir.rstrip(os.sep))
    mri_dir = os.path.join(split_dir, "mri")
    path_dir = os.path.join(split_dir, "path")

    paths = {
        "mri_group_prov": find_one([os.path.join(mri_dir, "*_group_recode_provenance_mri.csv")]),
        "path_group_prov": find_one([os.path.join(path_dir, "*_group_recode_provenance_path.csv")]),
        "path_phrase_prov": find_one([os.path.join(path_dir, "*_phrase_recode_provenance_path.csv")]),
    }
    source_split_id = infer_source_split_id(split_dir, list(paths.values()))
    print("-" * 100)
    print(f"[SPLIT_READ] directory={split_dir}")
    print(f"[SPLIT_READ] split_id={split_id} source_split_id={source_split_id}")
    for key, path in paths.items():
        print(f"[SPLIT_READ] {key}: {path if path else 'MISSING'}")

    mri_group_prov = safe_read_csv(paths["mri_group_prov"])
    path_group_prov = safe_read_csv(paths["path_group_prov"])
    path_phrase_prov = safe_read_csv(paths["path_phrase_prov"])
    role_table = build_role_table_from_provenance(mri_group_prov, path_group_prov, split_id)

    train_ids = role_table.loc[role_table["split_role"] == "train", "case_id"].astype(str).tolist()
    test_ids = role_table.loc[role_table["split_role"] == "test", "case_id"].astype(str).tolist()
    inventory = {
        "split_dir": split_dir,
        "split_id": split_id,
        "source_split_id": source_split_id,
        "n_train_cases": len(set(train_ids)),
        "n_test_cases": len(set(test_ids)),
        "train_case_hash": stable_case_hash(train_ids),
        "test_case_hash": stable_case_hash(test_ids),
        "has_path_group_prov": path_group_prov is not None and len(path_group_prov) > 0,
        "has_path_phrase_prov": path_phrase_prov is not None and len(path_phrase_prov) > 0,
        "has_mri_group_prov": mri_group_prov is not None and len(mri_group_prov) > 0,
    }
    print(
        f"[SPLIT_READ] role summary: n_train={inventory['n_train_cases']} "
        f"n_test={inventory['n_test_cases']} test_hash={inventory['test_case_hash']}"
    )
    if inventory["n_train_cases"] == 0 or inventory["n_test_cases"] == 0:
        raise ValueError(f"{split_id}: split has n_train=0 or n_test=0.")
    if path_group_prov is None or len(path_group_prov) == 0:
        raise ValueError(f"{split_id}: pathology group recode provenance is missing; cannot audit pathology.")

    return SplitArtifacts(
        split_dir=split_dir,
        split_id=split_id,
        source_split_id=source_split_id,
        mri_group_prov=mri_group_prov,
        path_group_prov=path_group_prov,
        path_phrase_prov=path_phrase_prov,
        role_table=role_table,
        inventory=inventory,
    )


def choose_splits_after_dedupe(splits: List[SplitArtifacts], dedupe_policy: str) -> Tuple[List[SplitArtifacts], pd.DataFrame]:
    if dedupe_policy == "keep_all":
        return splits, pd.DataFrame()
    if dedupe_policy not in {"drop_duplicate_test_sets", "drop_duplicate_source_ids"}:
        raise ValueError(f"Unsupported dedupe policy: {dedupe_policy}")
    buckets: Dict[str, List[SplitArtifacts]] = {}
    for split in splits:
        key = str(split.inventory["test_case_hash"] if dedupe_policy == "drop_duplicate_test_sets" else split.source_split_id)
        buckets.setdefault(key, []).append(split)

    chosen: List[SplitArtifacts] = []
    drop_rows: List[Dict[str, Any]] = []
    for key, group in buckets.items():
        if len(group) == 1:
            chosen.append(group[0])
            continue

        def completeness_score(s: SplitArtifacts) -> Tuple[int, float, str]:
            inv = s.inventory
            score = sum(int(bool(inv.get(k))) for k in [
                "has_path_group_prov",
                "has_path_phrase_prov",
                "has_mri_group_prov",
            ])
            csvs = glob.glob(os.path.join(s.split_dir, "**", "*.csv"), recursive=True)
            latest_mtime = max([os.path.getmtime(p) for p in csvs], default=0.0)
            return score, latest_mtime, s.split_id

        selected = sorted(group, key=completeness_score, reverse=True)[0]
        chosen.append(selected)
        for dropped in group:
            if dropped is selected:
                continue
            drop_rows.append({
                "dedupe_key": key,
                "dropped_split_id": dropped.split_id,
                "dropped_split_dir": dropped.split_dir,
                "selected_split_id": selected.split_id,
                "selected_split_dir": selected.split_dir,
                "dedupe_policy": dedupe_policy,
            })
        print(
            f"[DEDUPE] key={key} selected={selected.split_id}; "
            f"dropped={','.join([s.split_id for s in group if s is not selected])} policy={dedupe_policy}"
        )
    return sorted(chosen, key=lambda s: s.split_id), pd.DataFrame(drop_rows)


# -----------------------------------------------------------------------------
# Raw pathology text and audit regeneration
# -----------------------------------------------------------------------------

def detect_path_text_col(raw_df: pd.DataFrame, user_col: Optional[str] = None) -> str:
    if user_col:
        if user_col not in raw_df.columns:
            raise ValueError(
                f"Requested --path-text-col '{user_col}' not found in raw CSV. Available columns include: "
                f"{list(raw_df.columns)[:50]}"
            )
        return user_col

    col = first_existing_col(raw_df, RAW_PATH_TEXT_COLUMN_CANDIDATES)
    if col is not None:
        return col

    # Heuristic fallback: among text-like columns containing path/pathology, choose the one
    # with the largest average text length.
    candidates = []
    for c in raw_df.columns:
        name = str(c).lower()
        if ("path" in name or "pathology" in name) and raw_df[c].dtype == object:
            avg_len = raw_df[c].dropna().astype(str).str.len().mean()
            candidates.append((float(avg_len) if np.isfinite(avg_len) else 0.0, c))
    if candidates:
        return sorted(candidates, reverse=True)[0][1]

    raise ValueError(
        "Could not auto-detect pathology report text column. Pass --path-text-col explicitly. "
        f"Available columns include: {list(raw_df.columns)[:80]}"
    )


def selected_path_text_from_raw(raw_df: pd.DataFrame, row_index: int, path_text_col: str) -> str:
    if raw_df is None or path_text_col not in raw_df.columns:
        return ""
    if int(row_index) < 0 or int(row_index) >= len(raw_df):
        return ""
    val = raw_df.iloc[int(row_index)].get(path_text_col, "")
    return "" if pd.isna(val) else str(val)


def report_section_flags(text: str) -> Dict[str, int]:
    """Same core flags as MRI audit, with extra pathology-specific flags.

    The first five flags mirror the MRI audit exactly. The additional pathology
    flags help interpret pathology report structure without changing the primary
    MRI-compatible metrics.
    """
    t = normalize_text(text)
    return {
        # MRI-compatible flags
        "has_findings_section": int(bool(re.search(r"\bfindings\b", t))),
        "has_impression_section": int(bool(re.search(r"\bimpression\b", t))),
        "has_comparison_section": int(bool(re.search(r"\bcomparison\b|\bcompared with\b|\bprior\b|\bprevious\b", t))),
        "has_response_language": int(bool(re.search(
            r"\bresponse\b|\bdecreased\b|\bresolved\b|\bpersistent\b|\bresidual\b|"
            r"\btreatment effect\b|\btherapy effect\b|\bnear complete\b|\bcomplete response\b|"
            r"\bpartial response\b|\bviable tumor\b|\btumor bed\b",
            t,
        ))),
        "has_measurement_language": int(bool(re.search(r"\b\d+(?:\.\d+)?\s*(?:cm|mm)\b", t))),
        # Pathology-specific structure/context flags
        "has_final_diagnosis_section": int(bool(re.search(r"\bfinal diagnosis\b|\bdiagnosis\b", t))),
        "has_synoptic_section": int(bool(re.search(r"\bsynoptic\b|\bcase summary\b|\bcancer case summary\b", t))),
        "has_gross_description_section": int(bool(re.search(r"\bgross description\b|\bgross\b", t))),
        "has_microscopic_section": int(bool(re.search(r"\bmicroscopic\b|\bmicroscopy\b", t))),
        "has_margin_language": int(bool(re.search(r"\bmargin\b|\bmargins\b|\binked\b|\btransected\b", t))),
        "has_lymph_node_language": int(bool(re.search(r"\blymph node\b|\blymph nodes\b|\bnode-positive\b|\bnode negative\b|\bsentinel\b", t))),
    }


def prepare_provenance_numeric(prov: pd.DataFrame) -> pd.DataFrame:
    out = prov.copy()
    out["case_id"] = out["case_id"].astype(str)
    if "feature_slug" not in out.columns:
        # Defensive fallback for future provenance schemas.
        for alt in ["concept", "feature", "canonical_concept", "feature_name"]:
            if alt in out.columns:
                out["feature_slug"] = out[alt].astype(str)
                break
    if "feature_slug" not in out.columns:
        raise ValueError("Provenance table lacks feature_slug or a recognizable feature/concept column.")
    for col in ["count", "present", "uncertain_count", "negated_count"]:
        if col not in out.columns:
            out[col] = 0.0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    return out


def regenerate_pathology_audit_from_frozen_outputs(
    split: SplitArtifacts,
    raw_df: pd.DataFrame,
    path_text_col: str,
    include_mri_compatible_aliases: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if split.path_group_prov is None or len(split.path_group_prov) == 0:
        raise ValueError("Pathology group recode provenance is missing; cannot regenerate pathology audit.")

    role_cols = ["case_id", "row_index", "split_role"]
    for col in ["dispersion_true", "dispersion_true_high_low", "relapse_true"]:
        if col in split.role_table.columns:
            role_cols.append(col)
    role = split.role_table[role_cols].copy()
    role["case_id"] = role["case_id"].astype(str)

    phrase_counts = pd.DataFrame()
    if split.path_phrase_prov is not None and len(split.path_phrase_prov):
        phr = prepare_provenance_numeric(split.path_phrase_prov)
        phrase_counts = phr.groupby(["case_id", "row_index"]).agg(
            n_phrase_features=("feature_slug", "nunique"),
            n_phrase_present=("present", "sum"),
            n_phrase_uncertain=("uncertain_count", "sum"),
            n_phrase_negated=("negated_count", "sum"),
        ).reset_index()

    grp = prepare_provenance_numeric(split.path_group_prov)
    grp["is_spatial_morph_response"] = grp["feature_slug"].astype(str).isin(SPATIAL_MORPH_RESPONSE_GROUPS).astype(int)
    grp["is_distribution"] = grp["feature_slug"].astype(str).isin(DISTRIBUTION_GROUPS).astype(int)
    grp["is_ambiguity"] = grp["feature_slug"].astype(str).isin(AMBIGUITY_GROUPS).astype(int)
    grp["spatial_count"] = grp["count"] * grp["is_spatial_morph_response"]
    grp["distribution_count"] = grp["count"] * grp["is_distribution"]
    grp["ambiguity_count"] = grp["count"] * grp["is_ambiguity"]
    group_counts = grp.groupby(["case_id", "row_index"]).agg(
        n_group_features=("feature_slug", "nunique"),
        n_group_present=("present", "sum"),
        n_spatial_morph_response_phrases=("spatial_count", "sum"),
        n_distribution_phrases=("distribution_count", "sum"),
        n_ambiguity_phrases=("ambiguity_count", "sum"),
        n_uncertain_group_mentions=("uncertain_count", "sum"),
        n_negated_group_mentions=("negated_count", "sum"),
    ).reset_index()

    audit = role.copy()
    audit["split_id"] = split.split_id
    audit["source_split_id"] = split.source_split_id
    audit["path_report_text"] = audit["row_index"].astype(int).apply(
        lambda i: selected_path_text_from_raw(raw_df, i, path_text_col)
    )
    audit["path_text_column"] = path_text_col
    audit["path_report_missing"] = audit["path_report_text"].apply(lambda x: int(str(x).strip() == ""))
    audit["path_report_chars"] = audit["path_report_text"].apply(lambda x: len(str(x or "")))
    audit["path_report_words"] = audit["path_report_text"].apply(word_count)

    flags_df = pd.DataFrame([report_section_flags(x) for x in audit["path_report_text"].tolist()])
    audit = pd.concat([audit.reset_index(drop=True), flags_df.reset_index(drop=True)], axis=1)
    audit = audit.drop(columns=["path_report_text"])

    if len(phrase_counts):
        audit = audit.merge(phrase_counts, on=["case_id", "row_index"], how="left")
    audit = audit.merge(group_counts, on=["case_id", "row_index"], how="left")

    count_cols = [c for c in audit.columns if c.startswith("n_") or c.startswith("has_")]
    audit[count_cols] = audit[count_cols].fillna(0.0)
    denom = audit["path_report_words"].replace(0, np.nan)
    audit["path_extraction_density"] = audit.get("n_spatial_morph_response_phrases", 0.0) / denom
    audit["distribution_density"] = audit.get("n_distribution_phrases", 0.0) / denom
    audit["ambiguity_density"] = audit.get("n_ambiguity_phrases", 0.0) / denom
    audit["uncertainty_density"] = audit.get("n_uncertain_group_mentions", 0.0) / denom
    audit["negation_density"] = audit.get("n_negated_group_mentions", 0.0) / denom
    density_cols = ["path_extraction_density", "distribution_density", "ambiguity_density", "uncertainty_density", "negation_density"]
    audit[density_cols] = audit[density_cols].fillna(0.0)

    if include_mri_compatible_aliases:
        for src, dst in MRI_COMPATIBLE_ALIAS_MAP.items():
            audit[dst] = audit[src]

    summary = summarize_audit_by_groups(audit)
    return audit, summary


def summarize_audit_by_groups(audit_df: pd.DataFrame) -> pd.DataFrame:
    if len(audit_df) == 0:
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    for label_col in ["dispersion_true_high_low", "relapse_true", "split_role"]:
        if label_col not in audit_df.columns:
            continue
        sub_df = audit_df[audit_df[label_col].notna()].copy()
        if len(sub_df) == 0 or sub_df[label_col].nunique() < 2:
            continue
        for label_value, sub in sub_df.groupby(label_col, dropna=True):
            rec: Dict[str, Any] = {
                "comparison": label_col,
                "group_value": label_value,
                "n_cases_or_rows": len(sub),
            }
            for m in PATH_AUDIT_METRICS:
                if m in sub.columns:
                    vals = pd.to_numeric(sub[m], errors="coerce")
                    rec[f"{m}_mean"] = float(vals.mean()) if vals.notna().any() else np.nan
                    rec[f"{m}_median"] = float(vals.median()) if vals.notna().any() else np.nan
                    rec[f"{m}_sd"] = float(vals.std(ddof=1)) if vals.notna().sum() > 1 else np.nan
            rows.append(rec)
    return pd.DataFrame(rows)


def collect_pathology_audits(
    splits: List[SplitArtifacts],
    raw_df: pd.DataFrame,
    path_text_col: str,
    include_mri_compatible_aliases: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    case_tables: List[pd.DataFrame] = []
    summary_tables: List[pd.DataFrame] = []
    log_rows: List[Dict[str, Any]] = []
    for split in splits:
        status = "skipped"
        detail = ""
        audit_case = pd.DataFrame()
        audit_summary = pd.DataFrame()
        try:
            audit_case, audit_summary = regenerate_pathology_audit_from_frozen_outputs(
                split=split,
                raw_df=raw_df,
                path_text_col=path_text_col,
                include_mri_compatible_aliases=include_mri_compatible_aliases,
            )
            status = "regenerated_from_frozen_pathology_recode"
            detail = "regenerated using pathology recode provenance and raw pathology text; no API calls"
            case_tables.append(audit_case)
            if len(audit_summary):
                audit_summary = audit_summary.copy()
                audit_summary["split_id"] = split.split_id
                audit_summary["source_split_id"] = split.source_split_id
                summary_tables.append(audit_summary)
        except Exception as exc:
            status = "regeneration_failed"
            detail = str(exc)
            print(f"[AUDIT] {split.split_id}: regeneration failed: {detail}")
        log_rows.append({
            "split_id": split.split_id,
            "source_split_id": split.source_split_id,
            "audit_status": status,
            "detail": detail,
            "n_audit_case_rows": int(len(audit_case)),
            "n_audit_summary_rows": int(len(audit_summary)),
        })
    audit_all = pd.concat(case_tables, ignore_index=True) if case_tables else pd.DataFrame()
    summary_all = pd.concat(summary_tables, ignore_index=True) if summary_tables else pd.DataFrame()
    audit_log = pd.DataFrame(log_rows)
    return audit_all, summary_all, audit_log


# -----------------------------------------------------------------------------
# Case-level collapse and statistics
# -----------------------------------------------------------------------------

def collapse_to_case_level(audit_all: pd.DataFrame) -> pd.DataFrame:
    """Collapse duplicated split-level rows to one row per case_id,row_index.

    For binary/label columns, take the first non-missing value. For metric columns,
    take the median across repeated rows. This prevents high/low comparisons from
    over-weighting cases that appear in multiple split artifacts.
    """
    if len(audit_all) == 0:
        return pd.DataFrame()
    df = audit_all.copy()
    keys = ["case_id", "row_index"]
    for k in keys:
        if k not in df.columns:
            raise ValueError(f"Cannot collapse audit table; missing key column {k}")

    metric_cols = [m for m in PATH_AUDIT_METRICS if m in df.columns]
    first_cols = [c for c in ["dispersion_true", "dispersion_true_high_low", "relapse_true", "path_text_column"] if c in df.columns]
    rows: List[Dict[str, Any]] = []
    for (case_id, row_index), sub in df.groupby(keys, dropna=False):
        rec: Dict[str, Any] = {"case_id": case_id, "row_index": row_index, "n_split_rows_collapsed": len(sub)}
        for c in first_cols:
            non_missing = sub[c].dropna()
            rec[c] = non_missing.iloc[0] if len(non_missing) else np.nan
        rec["split_ids_collapsed"] = ";".join(sorted(set(map(str, sub.get("split_id", pd.Series(dtype=str)).dropna()))))
        rec["source_split_ids_collapsed"] = ";".join(sorted(set(map(str, sub.get("source_split_id", pd.Series(dtype=str)).dropna()))))
        for m in metric_cols:
            vals = pd.to_numeric(sub[m], errors="coerce")
            rec[m] = float(vals.median()) if vals.notna().any() else np.nan
        rows.append(rec)
    out = pd.DataFrame(rows)
    return coerce_numeric_targets(out)


def is_binary_metric(vals: pd.Series) -> bool:
    non_na = pd.to_numeric(vals, errors="coerce").dropna()
    if len(non_na) == 0:
        return False
    unique = set(non_na.astype(float).unique())
    return unique.issubset({0.0, 1.0}) and len(unique) <= 2


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Cliff's delta for x vs y. Positive means values in y tend to be larger if called with low, high below."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) == 0 or len(y) == 0:
        return np.nan
    # For small clinical datasets, the simple O(n*m) implementation is fine and transparent.
    greater = 0
    less = 0
    for xi in x:
        greater += int(np.sum(xi > y))
        less += int(np.sum(xi < y))
    return float((greater - less) / max(len(x) * len(y), 1))


def benjamini_hochberg(p_values: Sequence[float]) -> List[float]:
    p = np.asarray([np.nan if v is None else float(v) for v in p_values], dtype=float)
    q = np.full_like(p, np.nan, dtype=float)
    valid = np.where(np.isfinite(p))[0]
    if len(valid) == 0:
        return q.tolist()
    order = valid[np.argsort(p[valid])]
    ranked = p[order]
    m = len(ranked)
    adjusted = ranked * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)
    q[order] = adjusted
    return q.tolist()


def compare_high_low(df: pd.DataFrame, metrics: Sequence[str]) -> pd.DataFrame:
    if "dispersion_true_high_low" not in df.columns:
        raise ValueError("Audit table lacks dispersion_true_high_low; cannot compare high vs low dispersion.")
    work = df[df["dispersion_true_high_low"].notna()].copy()
    work["dispersion_true_high_low"] = pd.to_numeric(work["dispersion_true_high_low"], errors="coerce").round().astype("Int64")
    rows: List[Dict[str, Any]] = []
    for metric in metrics:
        if metric not in work.columns:
            continue
        vals = pd.to_numeric(work[metric], errors="coerce")
        label = work["dispersion_true_high_low"]
        low = vals[(label == 0) & vals.notna()].astype(float).values
        high = vals[(label == 1) & vals.notna()].astype(float).values
        rec: Dict[str, Any] = {
            "metric": metric,
            "n_low": int(len(low)),
            "n_high": int(len(high)),
            "mean_low": float(np.mean(low)) if len(low) else np.nan,
            "mean_high": float(np.mean(high)) if len(high) else np.nan,
            "median_low": float(np.median(low)) if len(low) else np.nan,
            "median_high": float(np.median(high)) if len(high) else np.nan,
            "sd_low": float(np.std(low, ddof=1)) if len(low) > 1 else np.nan,
            "sd_high": float(np.std(high, ddof=1)) if len(high) > 1 else np.nan,
            "mean_diff_high_minus_low": float(np.mean(high) - np.mean(low)) if len(low) and len(high) else np.nan,
            "median_diff_high_minus_low": float(np.median(high) - np.median(low)) if len(low) and len(high) else np.nan,
            "primary_test": "not_tested",
            "primary_p_value": np.nan,
            "welch_t_p_value": np.nan,
            "mannwhitney_p_value": np.nan,
            "fisher_exact_p_value": np.nan,
            "effect_size": np.nan,
            "effect_size_name": "",
            "odds_ratio_high_vs_low": np.nan,
        }
        if len(low) > 0 and len(high) > 0:
            metric_vals = vals[vals.notna()]
            if is_binary_metric(metric_vals):
                low_success = int(np.sum(low == 1))
                low_fail = int(np.sum(low == 0))
                high_success = int(np.sum(high == 1))
                high_fail = int(np.sum(high == 0))
                table = np.array([[high_success, high_fail], [low_success, low_fail]], dtype=int)
                try:
                    odds_ratio, p_fisher = stats.fisher_exact(table, alternative="two-sided")
                except Exception:
                    odds_ratio, p_fisher = np.nan, np.nan
                rec["primary_test"] = "Fisher exact test"
                rec["primary_p_value"] = float(p_fisher) if np.isfinite(p_fisher) else np.nan
                rec["fisher_exact_p_value"] = rec["primary_p_value"]
                rec["effect_size"] = float(np.mean(high) - np.mean(low))
                rec["effect_size_name"] = "risk_difference_high_minus_low"
                rec["odds_ratio_high_vs_low"] = float(odds_ratio) if np.isfinite(odds_ratio) else np.nan
            else:
                if len(low) > 1 and len(high) > 1:
                    try:
                        _, p_welch = stats.ttest_ind(high, low, equal_var=False, nan_policy="omit")
                    except Exception:
                        p_welch = np.nan
                    rec["welch_t_p_value"] = float(p_welch) if np.isfinite(p_welch) else np.nan
                if len(low) >= 1 and len(high) >= 1:
                    try:
                        _, p_mwu = stats.mannwhitneyu(high, low, alternative="two-sided")
                    except Exception:
                        p_mwu = np.nan
                    rec["mannwhitney_p_value"] = float(p_mwu) if np.isfinite(p_mwu) else np.nan
                    rec["primary_test"] = "Mann-Whitney U test"
                    rec["primary_p_value"] = rec["mannwhitney_p_value"]
                # Cliff's delta called as low vs high, then negated so positive means high > low.
                cd_low_vs_high = cliffs_delta(low, high)
                rec["effect_size"] = float(-cd_low_vs_high) if np.isfinite(cd_low_vs_high) else np.nan
                rec["effect_size_name"] = "cliffs_delta_positive_high_larger"
        rows.append(rec)
    out = pd.DataFrame(rows)
    if len(out):
        out["primary_p_value_fdr_bh"] = benjamini_hochberg(out["primary_p_value"].tolist())
        out = out.sort_values(["primary_p_value_fdr_bh", "primary_p_value", "metric"], na_position="last")
    return out


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def metric_label(metric: str) -> str:
    return metric.replace("_", " ")


def add_pvalue_annotation(ax: Any, p_value: float, q_value: Optional[float] = None) -> None:
    if not np.isfinite(p_value):
        text = "p = NA"
    elif p_value < 1e-4:
        text = f"p = {p_value:.1e}"
    else:
        text = f"p = {p_value:.4f}"
    if q_value is not None and np.isfinite(q_value):
        text += f"\nFDR q = {q_value:.4f}" if q_value >= 1e-4 else f"\nFDR q = {q_value:.1e}"
    ax.text(0.5, 0.98, text, transform=ax.transAxes, ha="center", va="top", fontsize=9)


def plot_continuous_metric(df: pd.DataFrame, metric: str, stat_row: pd.Series, out_path: str, jitter_seed: int = 17) -> None:
    sub = df[["dispersion_true_high_low", metric]].copy()
    sub[metric] = pd.to_numeric(sub[metric], errors="coerce")
    sub = sub[sub["dispersion_true_high_low"].notna() & sub[metric].notna()]
    if len(sub) == 0:
        return
    groups = []
    labels = []
    for label in [0, 1]:
        vals = sub.loc[sub["dispersion_true_high_low"].astype(int) == label, metric].astype(float).values
        groups.append(vals)
        labels.append("Low dispersion\n(0)" if label == 0 else "High dispersion\n(1)")
    if sum(len(g) > 0 for g in groups) < 2:
        return

    rng = np.random.RandomState(jitter_seed)
    plt.figure(figsize=(6.2, 4.8))
    ax = plt.gca()
    ax.boxplot(groups, tick_labels=labels, showfliers=True)
    for i, vals in enumerate(groups, start=1):
        if len(vals):
            jitter = rng.normal(0, 0.035, size=len(vals))
            ax.scatter(np.full(len(vals), i) + jitter, vals, alpha=0.55, s=18)
            ax.scatter([i], [np.mean(vals)], marker="D", s=42, zorder=3)
    ax.set_xlabel("True high dispersion label")
    ax.set_ylabel(metric_label(metric))
    ax.set_title(f"Pathology audit metric by dispersion: {metric}")
    add_pvalue_annotation(
        ax,
        float(stat_row.get("primary_p_value", np.nan)),
        float(stat_row.get("primary_p_value_fdr_bh", np.nan)),
    )
    ax.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


def binomial_se_ci(values: np.ndarray) -> Tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    p = float(np.mean(values))
    se = math.sqrt(max(p * (1.0 - p), 0.0) / max(len(values), 1))
    return p, max(0.0, p - 1.96 * se), min(1.0, p + 1.96 * se)


def plot_binary_metric(df: pd.DataFrame, metric: str, stat_row: pd.Series, out_path: str) -> None:
    sub = df[["dispersion_true_high_low", metric]].copy()
    sub[metric] = pd.to_numeric(sub[metric], errors="coerce")
    sub = sub[sub["dispersion_true_high_low"].notna() & sub[metric].notna()]
    if len(sub) == 0:
        return
    rates = []
    err_low = []
    err_high = []
    labels = ["Low dispersion\n(0)", "High dispersion\n(1)"]
    for label in [0, 1]:
        vals = sub.loc[sub["dispersion_true_high_low"].astype(int) == label, metric].astype(float).values
        p, lo, hi = binomial_se_ci(vals)
        rates.append(p)
        err_low.append(p - lo if np.isfinite(p) and np.isfinite(lo) else 0)
        err_high.append(hi - p if np.isfinite(p) and np.isfinite(hi) else 0)
    if not all(np.isfinite(rates)):
        return

    plt.figure(figsize=(5.8, 4.6))
    ax = plt.gca()
    ax.bar([0, 1], rates, yerr=[err_low, err_high], capsize=4)
    ax.set_xticks([0, 1], labels)
    ax.set_ylim(0, 1)
    ax.set_xlabel("True high dispersion label")
    ax.set_ylabel(f"Fraction with {metric}")
    ax.set_title(f"Pathology audit flag by dispersion: {metric}")
    add_pvalue_annotation(
        ax,
        float(stat_row.get("primary_p_value", np.nan)),
        float(stat_row.get("primary_p_value_fdr_bh", np.nan)),
    )
    ax.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_pvalue_overview(stats_df: pd.DataFrame, out_path: str, top_n: int = 30) -> None:
    if len(stats_df) == 0 or "primary_p_value" not in stats_df.columns:
        return
    df = stats_df.copy()
    df = df[np.isfinite(pd.to_numeric(df["primary_p_value"], errors="coerce"))].copy()
    if len(df) == 0:
        return
    df["neg_log10_p"] = -np.log10(np.maximum(pd.to_numeric(df["primary_p_value"], errors="coerce"), 1e-300))
    df = df.sort_values("primary_p_value").head(top_n).iloc[::-1]
    plt.figure(figsize=(9, max(4.5, 0.28 * len(df) + 1.5)))
    ax = plt.gca()
    ax.barh(df["metric"].astype(str), df["neg_log10_p"].astype(float))
    ax.axvline(-math.log10(0.05), linestyle="--", linewidth=1)
    ax.set_xlabel("-log10(primary p-value)")
    ax.set_ylabel("Metric")
    ax.set_title("Pathology audit high-vs-low dispersion differences")
    ax.grid(axis="x", alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


def make_metric_plots(df: pd.DataFrame, stats_df: pd.DataFrame, plots_dir: str) -> None:
    os.makedirs(plots_dir, exist_ok=True)
    if len(stats_df) == 0:
        return
    stat_lookup = {str(row["metric"]): row for _, row in stats_df.iterrows()}
    for metric, row in stat_lookup.items():
        if metric not in df.columns:
            continue
        vals = pd.to_numeric(df[metric], errors="coerce")
        if vals.notna().sum() == 0:
            continue
        out_path = os.path.join(plots_dir, f"pathology_audit_{sanitize_filename(metric)}_by_high_low.png")
        if is_binary_metric(vals):
            plot_binary_metric(df, metric, row, out_path)
        else:
            plot_continuous_metric(df, metric, row, out_path)
    plot_pvalue_overview(stats_df, os.path.join(plots_dir, "pathology_audit_high_low_pvalue_overview.png"))


# -----------------------------------------------------------------------------
# Output and CLI
# -----------------------------------------------------------------------------

def default_out_dir(split_root: Optional[str], split_dirs: Optional[Sequence[str]]) -> str:
    if split_root:
        parent = os.path.dirname(os.path.abspath(split_root.rstrip(os.sep)))
    elif split_dirs:
        parent = os.path.dirname(os.path.abspath(split_dirs[0].rstrip(os.sep)))
    else:
        parent = os.getcwd()
    return os.path.join(parent, "pathology_underperformance_audit")


def write_summary_text(
    out_path: str,
    args: argparse.Namespace,
    path_text_col: str,
    audit_all: pd.DataFrame,
    audit_case: pd.DataFrame,
    stats_df: pd.DataFrame,
    audit_log: pd.DataFrame,
) -> None:
    lines: List[str] = []
    lines.append("Pathology underperformance audit summary")
    lines.append("========================================")
    lines.append("")
    lines.append(f"Split root: {args.split_root}")
    lines.append(f"Raw CSV: {args.csv_path}")
    lines.append(f"Pathology text column: {path_text_col}")
    lines.append(f"Output directory: {args.out_dir}")
    lines.append(f"Split-level audit rows: {len(audit_all)}")
    lines.append(f"Case-level collapsed rows: {len(audit_case)}")
    if len(audit_case) and "dispersion_true_high_low" in audit_case.columns:
        counts = audit_case["dispersion_true_high_low"].value_counts(dropna=False).to_dict()
        lines.append(f"Case-level high/low counts: {counts}")
    lines.append("")
    lines.append("Audit status counts:")
    if len(audit_log):
        for status, n in audit_log["audit_status"].value_counts(dropna=False).items():
            lines.append(f"- {status}: {n}")
    lines.append("")
    lines.append("Top high-vs-low dispersion metric differences by FDR-adjusted p-value:")
    if len(stats_df):
        top = stats_df.sort_values(["primary_p_value_fdr_bh", "primary_p_value"], na_position="last").head(15)
        for _, row in top.iterrows():
            metric = row["metric"]
            p = row.get("primary_p_value", np.nan)
            q = row.get("primary_p_value_fdr_bh", np.nan)
            diff = row.get("mean_diff_high_minus_low", np.nan)
            eff = row.get("effect_size", np.nan)
            test = row.get("primary_test", "")
            lines.append(f"- {metric}: mean_diff_high-low={diff:.4g}, effect={eff:.4g}, {test}, p={p:.4g}, q={q:.4g}")
    else:
        lines.append("- No stats generated.")
    lines.append("")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a pathology-report information-density audit from completed outer_split_* artifacts."
    )
    parser.add_argument("--split-root", type=str, default=None, help="Directory containing completed outer_split* subdirectories.")
    parser.add_argument("--split-dirs", nargs="*", default=None, help="Explicit list of completed split directories.")
    parser.add_argument("--csv-path", type=str, required=True, help="Original raw CSV containing pathology report text.")
    parser.add_argument("--path-text-col", type=str, default=None, help="Pathology text column in raw CSV. If omitted, the script auto-detects it.")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory. Defaults to a new pathology_underperformance_audit folder in the parent of --split-root.")
    parser.add_argument(
        "--dedupe-policy",
        choices=["drop_duplicate_test_sets", "drop_duplicate_source_ids", "keep_all"],
        default="drop_duplicate_test_sets",
        help="How to deduplicate repeated completed split directories before aggregation.",
    )
    parser.add_argument(
        "--plot-unit",
        choices=["case", "split_row"],
        default="case",
        help="Use case-level collapsed rows or raw split-level rows for high/low stats and plots.",
    )
    parser.add_argument(
        "--include-mri-compatible-aliases",
        action="store_true",
        help="Also write mri_report_chars/mri_report_words/mri_extraction_density aliases for drop-in compatibility with MRI plotting code.",
    )
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.out_dir is None:
        args.out_dir = default_out_dir(args.split_root, args.split_dirs)
    args.out_dir = os.path.abspath(args.out_dir)
    plots_dir = os.path.join(args.out_dir, "plots")
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    print(f"[LOAD] Reading raw CSV: {args.csv_path}")
    raw_df = pd.read_csv(args.csv_path)
    path_text_col = detect_path_text_col(raw_df, args.path_text_col)
    print(f"[LOAD] Using pathology text column: {path_text_col}")

    split_dirs = discover_split_dirs(args.split_root, args.split_dirs)
    print(f"[DISCOVER] Found {len(split_dirs)} candidate split directories.")

    read_splits: List[SplitArtifacts] = []
    read_errors: List[Dict[str, Any]] = []
    for split_dir in split_dirs:
        try:
            read_splits.append(read_completed_split(split_dir))
        except Exception as exc:
            print(f"[WARN] Skipping split_dir={split_dir}: {exc}")
            read_errors.append({"split_dir": split_dir, "error": str(exc)})
    if not read_splits:
        raise RuntimeError("No usable split directories were read successfully.")

    chosen_splits, dedupe_df = choose_splits_after_dedupe(read_splits, args.dedupe_policy)
    print(f"[DEDUPE] Retained {len(chosen_splits)} split directories after policy={args.dedupe_policy}.")

    inventory_df = pd.DataFrame([s.inventory for s in chosen_splits])
    if len(inventory_df):
        inventory_df.to_csv(os.path.join(args.out_dir, "pathology_audit_retained_split_inventory.csv"), index=False)
    if len(dedupe_df):
        dedupe_df.to_csv(os.path.join(args.out_dir, "pathology_audit_deduped_splits.csv"), index=False)
    if read_errors:
        pd.DataFrame(read_errors).to_csv(os.path.join(args.out_dir, "pathology_audit_split_read_errors.csv"), index=False)

    audit_all, summary_all, audit_log = collect_pathology_audits(
        splits=chosen_splits,
        raw_df=raw_df,
        path_text_col=path_text_col,
        include_mri_compatible_aliases=bool(args.include_mri_compatible_aliases),
    )
    audit_log.to_csv(os.path.join(args.out_dir, "pathology_audit_generation_log.csv"), index=False)
    if len(audit_all) == 0:
        raise RuntimeError("No pathology audit rows were generated. Check pathology recode provenance and raw CSV path/text column.")

    audit_all = coerce_numeric_targets(audit_all)
    audit_all_path = os.path.join(args.out_dir, "pathology_audit_case_table_all_resumed_splits.csv")
    audit_all.to_csv(audit_all_path, index=False)
    print(f"[SAVE] {audit_all_path}")

    summary_all_path = os.path.join(args.out_dir, "pathology_audit_density_summary_all_resumed_splits.csv")
    summary_all.to_csv(summary_all_path, index=False)
    print(f"[SAVE] {summary_all_path}")

    audit_case = collapse_to_case_level(audit_all)
    audit_case_path = os.path.join(args.out_dir, "pathology_audit_case_table_case_level.csv")
    audit_case.to_csv(audit_case_path, index=False)
    print(f"[SAVE] {audit_case_path}")

    plot_df = audit_case if args.plot_unit == "case" else audit_all
    metrics_to_test = [m for m in PATH_AUDIT_METRICS if m in plot_df.columns]
    stats_df = compare_high_low(plot_df, metrics_to_test)
    stats_path = os.path.join(args.out_dir, "pathology_audit_high_low_metric_stats.csv")
    stats_df.to_csv(stats_path, index=False)
    print(f"[SAVE] {stats_path}")

    make_metric_plots(plot_df, stats_df, plots_dir)
    print(f"[SAVE] Wrote metric plots to {plots_dir}")

    summary_txt = os.path.join(args.out_dir, "pathology_audit_summary.txt")
    write_summary_text(summary_txt, args, path_text_col, audit_all, audit_case, stats_df, audit_log)
    print(f"[SAVE] {summary_txt}")
    print("[DONE] Pathology audit complete.")


if __name__ == "__main__":
    main()
