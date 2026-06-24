"""Frozen lexicon recoding and ontology groups."""

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
from approach2.eval_data import ensure_case_id
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
# Re-coding with frozen lexicon and ontology groups
# -----------------------------

def _phrase_context_status(report_norm: str, phrase_norm: str) -> Tuple[int, int, int, str]:
    idx = report_norm.find(phrase_norm)
    if idx < 0:
        return 0, 0, 0, ""
    start = max(0, idx - 80)
    end = min(len(report_norm), idx + len(phrase_norm) + 80)
    context = report_norm[start:end]
    if detect_negation(context):
        return 0, 1, 0, context
    if detect_uncertainty(context):
        return 0, 0, 1, context
    return 1, 0, 0, context


def _pattern_group_counts(report_norm: str, patterns: Sequence[str]) -> Tuple[int, int, int, str]:
    present = 0
    negated = 0
    uncertain = 0
    support = ""

    for pat in patterns:
        for match in re.finditer(pat, report_norm):
            start = max(0, match.start() - 80)
            end = min(len(report_norm), match.end() + 80)
            context = report_norm[start:end]
            support = context
            if detect_negation(context):
                negated += 1
            elif detect_uncertainty(context):
                uncertain += 1
            else:
                present += 1

    return present, negated, uncertain, support


def recode_cases_with_frozen_lexicon(
    raw_df: pd.DataFrame,
    outer_case_df: pd.DataFrame,
    report_mode: str,
    stable_phrase_df: pd.DataFrame,
    stable_group_df: pd.DataFrame,
    split_id: str,
    train_case_ids: Sequence[str],
    extra_group_names: Optional[Sequence[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    raw_df = ensure_case_id(raw_df).copy()
    raw_df["row_index"] = list(range(len(raw_df)))

    train_case_id_set = set(str(x) for x in train_case_ids)
    outer_row_indices = set(int(x) for x in outer_case_df["row_index"].tolist())

    phrase_rows: List[Dict[str, Any]] = []
    group_rows: List[Dict[str, Any]] = []

    stable_phrase_records = stable_phrase_df.to_dict("records") if len(stable_phrase_df) else []
    stable_groups = stable_group_df["canonical_group"].astype(str).tolist() if len(stable_group_df) else []
    if extra_group_names:
        stable_groups = sorted(set(stable_groups).union(set(map(str, extra_group_names))))

    outer_raw = raw_df[raw_df["row_index"].isin(outer_row_indices)].copy()

    for _, row in outer_raw.iterrows():
        case_id = str(row["case_id"])
        row_index = int(row["row_index"])
        report_text = _selected_report_text(
            type("TmpCase", (), {
                "preop_mri": row.get("preop_MRI_text", ""),
                "path_report": row.get("path_report_text", ""),
            })(),
            report_mode,
        )
        report_norm = normalize_text(report_text)
        missing = int(_is_missing_text(report_text))
        split_role = "train" if case_id in train_case_id_set else "test"
        y_score = pd.to_numeric(row["dispersion_invasive_DCIS_geographic"], errors="coerce")
        y_highlow = _true_dispersion_high_low(y_score)
        relapse_true = pd.to_numeric(row["relapse"], errors="coerce")

        for phrase_rec in stable_phrase_records:
            phrase_slug = str(phrase_rec["phrase_slug"])
            phrase_norm = str(phrase_rec["quote_norm"])
            canonical_group = str(phrase_rec["canonical_group"])

            if missing:
                present = negated = uncertain = 0
                support_text = ""
            else:
                present, negated, uncertain, support_text = _phrase_context_status(report_norm, phrase_norm)

            phrase_rows.append({
                "case_id": case_id,
                "row_index": row_index,
                "report_mode": report_mode,
                "split_id": split_id,
                "split_role": split_role,
                "dispersion_true": y_score,
                "dispersion_true_high_low": y_highlow,
                "relapse_true": relapse_true,
                "feature_slug": phrase_slug,
                "feature_type": "phrase",
                "feature_group": canonical_group,
                "present": float(present),
                "count": float(present),
                "negated_count": float(negated),
                "uncertain_count": float(uncertain),
                "support_text": support_text,
                "selected_report_missing": missing,
            })

        for group in stable_groups:
            if missing:
                present = negated = uncertain = 0
                support_text = ""
            else:
                present, negated, uncertain, support_text = _pattern_group_counts(
                    report_norm, CANONICAL_GROUP_PATTERNS.get(group, [])
                )

            group_rows.append({
                "case_id": case_id,
                "row_index": row_index,
                "report_mode": report_mode,
                "split_id": split_id,
                "split_role": split_role,
                "dispersion_true": y_score,
                "dispersion_true_high_low": y_highlow,
                "relapse_true": relapse_true,
                "feature_slug": group,
                "feature_type": "group",
                "feature_group": group,
                "present": float(int(present > 0)),
                "count": float(present),
                "negated_count": float(negated),
                "uncertain_count": float(uncertain),
                "support_text": support_text,
                "selected_report_missing": missing,
            })

    return pd.DataFrame(phrase_rows), pd.DataFrame(group_rows)
