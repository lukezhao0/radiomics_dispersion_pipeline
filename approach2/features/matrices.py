"""Group/phrase feature matrices and early fusion."""

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
# Feature matrices
# -----------------------------

def _base_case_table(outer_case_df: pd.DataFrame, split_id: str, report_mode: str) -> pd.DataFrame:
    out = outer_case_df[["case_id", "row_index", "dispersion_true", "dispersion_true_high_low", "relapse_true"]].copy()
    out["split_id"] = split_id
    out["report_mode"] = report_mode
    return out.drop_duplicates().reset_index(drop=True)


def build_group_feature_matrix(group_prov_df: pd.DataFrame, outer_case_df: pd.DataFrame, split_id: str, report_mode: str) -> pd.DataFrame:
    base = _base_case_table(outer_case_df, split_id, report_mode)

    if len(group_prov_df) == 0:
        return base

    wide_parts = []
    for value_col, suffix in [
        ("present", "present"),
        ("count", "count"),
        ("negated_count", "negated_count"),
        ("uncertain_count", "uncertain_count"),
    ]:
        piv = (
            group_prov_df.pivot_table(
                index=["case_id", "row_index"],
                columns="feature_slug",
                values=value_col,
                aggfunc="sum",
                fill_value=0.0,
            )
            .rename(columns=lambda c: f"grp__{c}__{suffix}")
            .reset_index()
        )
        wide_parts.append(piv)

    out = base.copy()
    for part in wide_parts:
        out = out.merge(part, on=["case_id", "row_index"], how="left")

    out = out.fillna(0.0)
    return out


def build_phrase_feature_matrix(phrase_prov_df: pd.DataFrame, outer_case_df: pd.DataFrame, split_id: str, report_mode: str) -> pd.DataFrame:
    base = _base_case_table(outer_case_df, split_id, report_mode)

    if len(phrase_prov_df) == 0:
        return base

    piv = (
        phrase_prov_df.pivot_table(
            index=["case_id", "row_index"],
            columns="feature_slug",
            values="present",
            aggfunc="max",
            fill_value=0.0,
        )
        .rename(columns=lambda c: f"phr__{c}__present")
        .reset_index()
    )
    out = base.merge(piv, on=["case_id", "row_index"], how="left")
    out = out.fillna(0.0)
    return out


def get_representation_matrix(df: pd.DataFrame, representation: str) -> Tuple[pd.DataFrame, List[str]]:
    df = df.copy()
    if representation == "group_binary":
        cols = [c for c in df.columns if c.startswith("grp__") and c.endswith("__present")]
    elif representation == "group_count":
        cols = [c for c in df.columns if c.startswith("grp__") and c.endswith("__count") and "__negated_" not in c and "__uncertain_" not in c]
    elif representation == "group_status":
        cols = [c for c in df.columns if c.startswith("grp__") and (
            c.endswith("__present") or c.endswith("__negated_count") or c.endswith("__uncertain_count")
        )]
    elif representation == "phrase_binary":
        cols = [c for c in df.columns if c.startswith("phr__") and c.endswith("__present")]
    elif representation in {"weighted_concept_score", "weighted_plus_group_status"}:
        cols = [c for c in df.columns if c.startswith("wgrp__")]
        if representation == "weighted_plus_group_status":
            cols += [c for c in df.columns if c.startswith("grp__") and (
                c.endswith("__present") or c.endswith("__negated_count") or c.endswith("__uncertain_count")
            )]
    else:
        raise ValueError(f"Unsupported representation: {representation}")

    cols = sorted(set(cols))
    X = df[cols].copy() if cols else pd.DataFrame(index=df.index)
    return X, cols


def merge_modalities_early_fusion(
    mri_df: pd.DataFrame,
    path_df: pd.DataFrame,
    representation: str,
) -> Tuple[pd.DataFrame, List[str]]:
    mri_X, mri_cols = get_representation_matrix(mri_df, representation)
    path_X, path_cols = get_representation_matrix(path_df, representation)

    mri_meta = mri_df[["case_id", "row_index", "dispersion_true", "dispersion_true_high_low", "relapse_true"]].copy()
    path_meta = path_df[["case_id", "row_index"]].copy()

    out = mri_meta.merge(path_meta, on=["case_id", "row_index"], how="outer")

    if len(mri_cols):
        tmp = pd.concat([mri_df[["case_id", "row_index"]], mri_X], axis=1)
        tmp = tmp.rename(columns={c: f"mri__{c}" for c in mri_cols})
        out = out.merge(tmp, on=["case_id", "row_index"], how="left")

    if len(path_cols):
        tmp = pd.concat([path_df[["case_id", "row_index"]], path_X], axis=1)
        tmp = tmp.rename(columns={c: f"path__{c}" for c in path_cols})
        out = out.merge(tmp, on=["case_id", "row_index"], how="left")

    out = out.fillna(0.0)
    feature_cols = [c for c in out.columns if c not in META_COLS]
    return out, feature_cols
