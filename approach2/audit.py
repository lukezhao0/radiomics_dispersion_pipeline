"""MRI audit table construction."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import html
import json
import math
import os
import re
import sys
import time
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, HuberRegressor, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    mutual_info_score,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold, StratifiedShuffleSplit, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR, SVC

from approach2.config import (
    AMBIGUITY_GROUPS,
    CANONICAL_GROUP_PATTERNS,
    COEF_ZERO_TOL,
    DEFAULT_BOOTSTRAP_N,
    DISPERSION_TRUE_HIGH_THRESHOLD,
    DISTRIBUTION_GROUPS,
    EPS,
    INNER_CV_MAX_SPLITS,
    META_COLS,
    NEGATION_PATTERNS,
    RANDOM_SEED,
    SHARED_CONCEPT_ONTOLOGY,
    SPATIAL_MORPH_RESPONSE_GROUPS,
    TARGET_NAME_DISPERSION_HIGH_LOW,
    TARGET_NAME_DISPERSION_SCORE,
    TARGET_NAME_RELAPSE_STATUS,
    UNCERTAINTY_PATTERNS,
)
from approach2.extraction import (
    MAX_TOKENS,
    Tee,
    _is_missing_text,
    _selected_report_text,
    _true_dispersion_high_low,
    build_html_report,
    build_user_prompt,
    confirm_cost_estimate_or_exit,
    configure_global_api_concurrency,
    df_to_html_table,
    estimate_prompt_tokens_from_messages,
    extract_subset_records,
    html_paragraph,
    html_plot_block,
    html_section,
    load_cases,
    make_case_from_row,
    preflight_check,
    print_apriori_cost_estimate_report,
    print_cumulative_report,
    summarize_apriori_cost_estimate,
    write_cost_tracker_json,
    write_extractions,
)
from approach2.io_atomic import atomic_write_df as _atomic_write_df
from approach2.io_atomic import safe_read_csv_if_exists as _safe_read_csv_if_exists
from approach2.metrics import calibration_intercept_slope, rmse, safe_pearson, safe_spearman
from approach2.models import LowInfoFeatureFilter, ModelSpec
from approach2.text_utils import (
    clean_phrase_for_display,
    detect_negation,
    detect_uncertainty,
    make_slug,
    normalize_text,
    parse_jsonish,
    resolve_default_api_workers,
    resolve_default_ml_n_jobs,
    resolve_default_parallel_modality_workers,
)

# -----------------------------
# MRI audit outputs
# -----------------------------

def _report_section_flags(text: str) -> Dict[str, int]:
    t = normalize_text(text)
    section_patterns = {
        "has_findings_section": r"\bfindings\b",
        "has_impression_section": r"\bimpression\b",
        "has_comparison_section": r"\bcomparison\b|\bcompared with\b|\bprior\b",
        "has_response_language": r"\bresponse\b|\bdecreased\b|\bresolved\b|\bpersistent\b|\bresidual\b",
        "has_measurement_language": r"\b\d+(?:\.\d+)?\s*(?:cm|mm)\b",
    }
    return {name: int(bool(re.search(pat, t))) for name, pat in section_patterns.items()}


def compute_mri_audit_table(
    raw_df: pd.DataFrame,
    outer_case_df: pd.DataFrame,
    phrase_prov_df: pd.DataFrame,
    group_prov_df: pd.DataFrame,
    split_id: str,
) -> pd.DataFrame:
    raw_df = ensure_case_id(raw_df).copy()
    raw_df["row_index"] = list(range(len(raw_df)))
    outer_rows = raw_df[raw_df["row_index"].isin(set(outer_case_df["row_index"].astype(int)))].copy()

    phrase_counts = pd.DataFrame()
    if len(phrase_prov_df):
        phrase_counts = phrase_prov_df.groupby(["case_id", "row_index"]).agg(
            n_phrase_features=("feature_slug", "nunique"),
            n_phrase_present=("present", "sum"),
            n_phrase_uncertain=("uncertain_count", "sum"),
            n_phrase_negated=("negated_count", "sum"),
        ).reset_index()

    group_counts = pd.DataFrame()
    if len(group_prov_df):
        tmp = group_prov_df.copy()
        tmp["is_spatial_morph_response"] = tmp["feature_slug"].isin(SPATIAL_MORPH_RESPONSE_GROUPS).astype(int)
        tmp["is_distribution"] = tmp["feature_slug"].isin(DISTRIBUTION_GROUPS).astype(int)
        tmp["is_ambiguity"] = tmp["feature_slug"].isin(AMBIGUITY_GROUPS).astype(int)
        tmp["spatial_count"] = tmp["count"] * tmp["is_spatial_morph_response"]
        tmp["distribution_count"] = tmp["count"] * tmp["is_distribution"]
        tmp["ambiguity_count"] = tmp["count"] * tmp["is_ambiguity"]
        group_counts = tmp.groupby(["case_id", "row_index"]).agg(
            n_group_features=("feature_slug", "nunique"),
            n_group_present=("present", "sum"),
            n_spatial_morph_response_phrases=("spatial_count", "sum"),
            n_distribution_phrases=("distribution_count", "sum"),
            n_ambiguity_phrases=("ambiguity_count", "sum"),
            n_uncertain_group_mentions=("uncertain_count", "sum"),
            n_negated_group_mentions=("negated_count", "sum"),
        ).reset_index()

    rows = []
    for _, row in outer_rows.iterrows():
        report_text = str(row.get("preop_MRI_text", "") or "")
        word_count = len([w for w in re.split(r"\s+", report_text.strip()) if w])
        char_count = len(report_text)
        case_id = str(row["case_id"])
        row_index = int(row["row_index"])
        base = {
            "split_id": split_id,
            "case_id": case_id,
            "row_index": row_index,
            "dispersion_true": pd.to_numeric(row["dispersion_invasive_DCIS_geographic"], errors="coerce"),
            "dispersion_true_high_low": _true_dispersion_high_low(row["dispersion_invasive_DCIS_geographic"]),
            "relapse_true": pd.to_numeric(row.get("relapse", np.nan), errors="coerce"),
            "mri_report_missing": int(_is_missing_text(report_text)),
            "mri_report_chars": char_count,
            "mri_report_words": word_count,
        }
        base.update(_report_section_flags(report_text))
        rows.append(base)

    audit = pd.DataFrame(rows)
    for extra in [phrase_counts, group_counts]:
        if len(extra):
            audit = audit.merge(extra, on=["case_id", "row_index"], how="left")
    count_cols = [c for c in audit.columns if c.startswith("n_") or c.startswith("has_")]
    audit[count_cols] = audit[count_cols].fillna(0.0)
    denom = audit["mri_report_words"].replace(0, np.nan)
    audit["mri_extraction_density"] = audit.get("n_spatial_morph_response_phrases", 0.0) / denom
    audit["distribution_density"] = audit.get("n_distribution_phrases", 0.0) / denom
    audit["uncertainty_density"] = audit.get("n_uncertain_group_mentions", 0.0) / denom
    audit["negation_density"] = audit.get("n_negated_group_mentions", 0.0) / denom
    audit = audit.fillna({
        "mri_extraction_density": 0.0,
        "distribution_density": 0.0,
        "uncertainty_density": 0.0,
        "negation_density": 0.0,
    })
    return audit


def summarize_audit_by_groups(audit_df: pd.DataFrame) -> pd.DataFrame:
    if len(audit_df) == 0:
        return pd.DataFrame()
    metrics = [
        "mri_report_words",
        "n_phrase_present",
        "n_group_present",
        "n_spatial_morph_response_phrases",
        "n_distribution_phrases",
        "n_uncertain_group_mentions",
        "n_negated_group_mentions",
        "mri_extraction_density",
        "distribution_density",
        "uncertainty_density",
    ]
    rows = []
    for label_col in ["dispersion_true_high_low", "relapse_true"]:
        if label_col not in audit_df.columns:
            continue
        sub_df = audit_df[audit_df[label_col].notna()].copy()
        if len(sub_df) == 0 or sub_df[label_col].nunique() < 2:
            continue
        for label_value, sub in sub_df.groupby(label_col):
            rec = {"comparison": label_col, "group_value": label_value, "n_cases": len(sub)}
            for m in metrics:
                if m in sub.columns:
                    rec[f"{m}_mean"] = float(pd.to_numeric(sub[m], errors="coerce").mean())
                    rec[f"{m}_median"] = float(pd.to_numeric(sub[m], errors="coerce").median())
            rows.append(rec)
    return pd.DataFrame(rows)
