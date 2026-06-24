"""Outer and rediscovery split generation."""

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
# Split generation
# -----------------------------

def case_id_list_hash(values: Sequence[Any]) -> str:
    """Stable short hash for ordered case ID lists (split provenance)."""
    import hashlib
    payload = "\n".join(map(str, values))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def validate_outer_splits(
    splits: Sequence[Tuple[np.ndarray, np.ndarray]],
    scheme: str,
    n_cases: int,
) -> None:
    """Fail loudly on train/test overlap or invalid k-fold coverage."""
    if not splits:
        raise ValueError("No outer splits were generated.")

    test_counts: Counter = Counter()
    for split_idx, (train_idx, test_idx) in enumerate(splits, 1):
        train_set = set(int(x) for x in train_idx)
        test_set = set(int(x) for x in test_idx)
        overlap = train_set & test_set
        if overlap:
            raise ValueError(
                f"Outer split {split_idx}: train/test overlap detected for indices {sorted(overlap)[:10]}"
            )
        if train_set | test_set != set(range(n_cases)):
            missing = set(range(n_cases)) - (train_set | test_set)
            extra = (train_set | test_set) - set(range(n_cases))
            raise ValueError(
                f"Outer split {split_idx}: split does not partition all cases "
                f"(missing={len(missing)} extra={len(extra)})."
            )
        for idx in test_set:
            test_counts[idx] += 1

    if scheme == "stratified_kfold":
        expected_folds = len(splits)
        bad = {idx: c for idx, c in test_counts.items() if c != 1}
        if bad:
            preview = dict(list(bad.items())[:5])
            raise ValueError(
                f"stratified_kfold requires each case in held-out test exactly once across "
                f"{expected_folds} folds; violations (index->count): {preview}"
            )
        print(
            f"[OUTER] Validated stratified_kfold: {expected_folds} folds, "
            f"each of {n_cases} cases held out exactly once."
        )
    elif scheme == "repeated_mc":
        print(
            f"[OUTER] Validated repeated_mc: {len(splits)} independent 80/20 draws; "
            f"cases may appear in multiple outer-test sets (dedup uses mean aggregation)."
        )
    else:
        raise ValueError(f"Unsupported outer scheme for validation: {scheme}")


def log_outer_split_summary(
    splits: Sequence[Tuple[np.ndarray, np.ndarray]],
    y_binary: np.ndarray,
    scheme: str,
    n_repeats: int,
    test_frac: float,
    n_folds: int,
) -> None:
    """Log split mode and per-fold label distributions."""
    print(
        f"[OUTER] scheme={scheme} n_splits={len(splits)} "
        f"repeats={n_repeats} test_frac={test_frac} folds={n_folds}"
    )
    for split_idx, (train_idx, test_idx) in enumerate(splits, 1):
        y_train = y_binary[np.asarray(train_idx, dtype=int)]
        y_test = y_binary[np.asarray(test_idx, dtype=int)]
        print(
            f"[OUTER] split={split_idx:03d} n_train={len(train_idx)} n_test={len(test_idx)} "
            f"train_pos={int(y_train.sum())}/{len(y_train)} "
            f"test_pos={int(y_test.sum())}/{len(y_test)}"
        )


def build_outer_splits(
    y_binary: np.ndarray,
    scheme: str,
    random_seed: int,
    n_repeats: int,
    test_frac: float,
    n_folds: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    indices = np.arange(len(y_binary))
    splits: List[Tuple[np.ndarray, np.ndarray]] = []

    if scheme == "repeated_mc":
        for rep in range(n_repeats):
            rs = random_seed + rep
            splitter = StratifiedShuffleSplit(
                n_splits=1,
                test_size=test_frac,
                random_state=rs,
            )
            for train_idx, test_idx in splitter.split(indices, y_binary):
                splits.append((train_idx, test_idx))
    elif scheme == "stratified_kfold":
        splitter = StratifiedKFold(
            n_splits=n_folds,
            shuffle=True,
            random_state=random_seed,
        )
        for train_idx, test_idx in splitter.split(indices, y_binary):
            splits.append((train_idx, test_idx))
    else:
        raise ValueError(f"Unsupported outer scheme: {scheme}")

    return splits


def build_rediscovery_subsplits(
    case_ids: np.ndarray,
    y_binary: np.ndarray,
    scheme: str,
    random_seed: int,
    n_repeats: int,
    test_frac: float,
    n_folds: int,
) -> List[np.ndarray]:
    indices = np.arange(len(case_ids))
    train_subsets: List[np.ndarray] = []

    if len(indices) < 4 or len(np.unique(y_binary)) < 2:
        return [indices]

    if scheme == "repeated_mc":
        n_repeats = max(1, n_repeats)
        for rep in range(n_repeats):
            rs = random_seed + rep
            splitter = StratifiedShuffleSplit(
                n_splits=1,
                test_size=test_frac,
                random_state=rs,
            )
            for train_idx, _ in splitter.split(indices, y_binary):
                train_subsets.append(train_idx)
    elif scheme == "stratified_kfold":
        n_folds = min(n_folds, len(indices))
        n_folds = max(2, n_folds)
        splitter = StratifiedKFold(
            n_splits=n_folds,
            shuffle=True,
            random_state=random_seed,
        )
        for train_idx, _ in splitter.split(indices, y_binary):
            train_subsets.append(train_idx)
    else:
        raise ValueError(f"Unsupported rediscovery scheme: {scheme}")

    if not train_subsets:
        train_subsets = [indices]
    return train_subsets
