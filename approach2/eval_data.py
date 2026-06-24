"""Target frame construction and MRI-missing filters."""

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


def ensure_case_id(df: pd.DataFrame) -> pd.DataFrame:
    if "case_id" not in df.columns:
        df = df.copy()
        df["case_id"] = [f"row_{i}" for i in range(len(df))]
    return df


def get_target_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = ensure_case_id(df).copy()
    out["row_index"] = list(range(len(out)))
    out["dispersion_true"] = pd.to_numeric(
        out["dispersion_invasive_DCIS_geographic"], errors="coerce"
    )
    out["dispersion_true_high_low"] = out["dispersion_true"].apply(_true_dispersion_high_low)
    out["relapse_true"] = pd.to_numeric(out["relapse"], errors="coerce")
    out = out[out["dispersion_true"].notna() & out["dispersion_true_high_low"].notna()].copy()
    out["dispersion_true_high_low"] = out["dispersion_true_high_low"].astype(int)
    out["row_index"] = out["row_index"].astype(int)
    return out




def _raw_df_with_row_index(raw_df: pd.DataFrame) -> pd.DataFrame:
    out = ensure_case_id(raw_df).copy()
    if "row_index" not in out.columns:
        out["row_index"] = list(range(len(out)))
    return out


def _mri_missing_row_indices(raw_df: pd.DataFrame) -> set:
    raw = _raw_df_with_row_index(raw_df)
    if "preop_MRI_text" not in raw.columns:
        return set()
    return set(raw.loc[raw["preop_MRI_text"].apply(_is_missing_text), "row_index"].astype(int).tolist())


def dataset_requires_mri_report(dataset_key: str) -> bool:
    """Return True for datasets whose features cannot be interpreted without MRI text."""
    dataset_key = str(dataset_key)
    return (
        dataset_key == "mri"
        or dataset_key == "combined"
        or dataset_key.startswith("mri_pathcal")
        or dataset_key.startswith("mri_teacher_student")
    )


def filter_missing_mri_for_dataset(
    dataset_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    dataset_key: str,
    split_id: str,
) -> pd.DataFrame:
    """Drop MRI-missing cases from MRI-derived evaluations.

    Pathology-only evaluations intentionally keep these cases because every case is
    expected to have a pathology report. MRI-only, combined MRI+pathology,
    pathology-calibrated MRI, and teacher-student MRI evaluations are not valid
    without the MRI report, so those rows are removed from both train and test
    portions before model fitting.
    """
    if not dataset_requires_mri_report(dataset_key) or len(dataset_df) == 0:
        return dataset_df

    missing_rows = _mri_missing_row_indices(raw_df)
    if not missing_rows or "row_index" not in dataset_df.columns:
        return dataset_df

    before = len(dataset_df)
    out = dataset_df[~dataset_df["row_index"].astype(int).isin(missing_rows)].copy().reset_index(drop=True)
    skipped = before - len(out)
    if skipped > 0:
        print(
            f"[MISSING_MRI] split={split_id} dataset={dataset_key}: "
            f"skipped {skipped} case rows with missing preop_MRI_text."
        )
    return out
