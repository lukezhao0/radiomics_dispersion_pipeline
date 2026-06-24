"""Stable lexicon discovery from training extractions."""

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

from approach2.splits import build_rediscovery_subsplits

# -----------------------------
# Rediscovery and frozen lexicon
# -----------------------------


def cap_stable_phrase_lexicon(
    stable_phrase_df: pd.DataFrame,
    target_count: int,
    report_mode: str,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Select top-ranked stable phrases up to target_count (train-only ranking)."""
    meta: Dict[str, Any] = {
        "report_mode": report_mode,
        "target_stable_features": int(target_count),
        "n_stable_before_cap": len(stable_phrase_df),
        "n_stable_after_cap": len(stable_phrase_df),
        "cap_applied": False,
    }
    if target_count <= 0 or len(stable_phrase_df) <= target_count:
        if target_count > 0 and len(stable_phrase_df) < target_count:
            meta["note"] = "fewer_stable_features_than_target_using_all_available"
        return stable_phrase_df.copy(), meta

    ranked = stable_phrase_df.sort_values(
        ["selection_frequency", "mean_support_cases", "n_rows"],
        ascending=[False, False, False],
    ).head(int(target_count)).copy().reset_index(drop=True)
    meta.update({
        "n_stable_after_cap": len(ranked),
        "cap_applied": True,
    })
    print(
        f"[REDISCOVERY] Capped stable phrases for {report_mode}: "
        f"target={target_count} before={meta['n_stable_before_cap']} after={len(ranked)}"
    )
    return ranked, meta


def build_stable_lexicon_from_training_extractions(
    train_extractions_df: pd.DataFrame,
    train_phrase_df: pd.DataFrame,
    rediscovery_scheme: str,
    rediscovery_repeats: int,
    rediscovery_test_frac: float,
    rediscovery_folds: int,
    stability_threshold: float,
    min_phrase_cases: int,
    min_group_cases: int,
    random_seed: int,
    target_stable_features_per_modality: int = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    case_meta = (
        train_extractions_df[["case_id", "dispersion_true_high_low"]]
        .drop_duplicates()
        .copy()
    )
    case_ids = case_meta["case_id"].astype(str).values
    y_binary = case_meta["dispersion_true_high_low"].astype(int).values

    rediscovery_subsets = build_rediscovery_subsplits(
        case_ids=case_ids,
        y_binary=y_binary,
        scheme=rediscovery_scheme,
        random_seed=random_seed,
        n_repeats=rediscovery_repeats,
        test_frac=rediscovery_test_frac,
        n_folds=rediscovery_folds,
    )
    n_subsplits = len(rediscovery_subsets)
    print(f"[REDISCOVERY] Built {n_subsplits} inner rediscovery subsets.")

    if len(train_phrase_df) == 0:
        empty_phrase = pd.DataFrame(columns=["phrase_slug", "quote_norm", "canonical_group", "n_rows", "selected_count", "mean_support_cases", "selection_frequency", "stable"])
        empty_group = pd.DataFrame(columns=["canonical_group", "n_rows", "selected_count", "mean_support_cases", "selection_frequency", "stable"])
        return empty_phrase, empty_group, empty_phrase.copy(), empty_group.copy(), {
            "stability_threshold": stability_threshold,
            "n_stable_phrases": 0,
            "n_stable_groups": 0,
        }

    phrase_counts = Counter()
    group_counts = Counter()
    phrase_support_total = Counter()
    group_support_total = Counter()

    phrase_lookup = (
        train_phrase_df.groupby(["phrase_slug", "quote_norm", "canonical_group"])
        .size()
        .reset_index(name="n_rows")
    )
    group_lookup = (
        train_phrase_df.groupby(["canonical_group"])
        .size()
        .reset_index(name="n_rows")
    )

    case_id_arr = np.asarray(case_ids)

    for sub_idx, train_subset_idx in enumerate(rediscovery_subsets, 1):
        subset_case_ids = set(case_id_arr[train_subset_idx].tolist())
        subset_phrases = train_phrase_df[train_phrase_df["case_id"].isin(subset_case_ids)].copy()

        if len(subset_phrases) == 0:
            continue

        phrase_support = (
            subset_phrases.groupby("phrase_slug")["case_id"]
            .nunique()
            .to_dict()
        )
        group_support = (
            subset_phrases.groupby("canonical_group")["case_id"]
            .nunique()
            .to_dict()
        )

        for phrase_slug, support in phrase_support.items():
            phrase_support_total[phrase_slug] += int(support)
            if int(support) >= min_phrase_cases:
                phrase_counts[phrase_slug] += 1

        for group, support in group_support.items():
            group_support_total[group] += int(support)
            if int(support) >= min_group_cases:
                group_counts[group] += 1

        print(
            f"[REDISCOVERY] subset={sub_idx}/{n_subsplits} "
            f"n_cases={len(subset_case_ids)} "
            f"n_phrase_candidates={len(phrase_support)} "
            f"n_group_candidates={len(group_support)}"
        )

    phrase_freq_df = phrase_lookup.copy()
    phrase_freq_df["selected_count"] = phrase_freq_df["phrase_slug"].map(lambda x: phrase_counts.get(x, 0))
    phrase_freq_df["mean_support_cases"] = phrase_freq_df["phrase_slug"].map(
        lambda x: phrase_support_total.get(x, 0) / max(1, n_subsplits)
    )
    phrase_freq_df["selection_frequency"] = phrase_freq_df["selected_count"] / max(1, n_subsplits)
    phrase_freq_df["stable"] = (phrase_freq_df["selection_frequency"] >= stability_threshold).astype(int)
    phrase_freq_df = phrase_freq_df.sort_values(
        ["stable", "selection_frequency", "mean_support_cases", "n_rows"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)

    group_freq_df = group_lookup.copy()
    group_freq_df["selected_count"] = group_freq_df["canonical_group"].map(lambda x: group_counts.get(x, 0))
    group_freq_df["mean_support_cases"] = group_freq_df["canonical_group"].map(
        lambda x: group_support_total.get(x, 0) / max(1, n_subsplits)
    )
    group_freq_df["selection_frequency"] = group_freq_df["selected_count"] / max(1, n_subsplits)
    group_freq_df["stable"] = (group_freq_df["selection_frequency"] >= stability_threshold).astype(int)
    group_freq_df = group_freq_df.sort_values(
        ["stable", "selection_frequency", "mean_support_cases", "n_rows"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)

    stable_phrase_df = phrase_freq_df[phrase_freq_df["stable"] == 1].copy()
    stable_group_df = group_freq_df[group_freq_df["stable"] == 1].copy()

    cap_meta: Dict[str, Any] = {}
    if target_stable_features_per_modality > 0:
        stable_phrase_df, cap_meta = cap_stable_phrase_lexicon(
            stable_phrase_df,
            target_count=target_stable_features_per_modality,
            report_mode=str(train_extractions_df["report_mode"].iloc[0]) if "report_mode" in train_extractions_df.columns and len(train_extractions_df) else "unknown",
        )

    lexicon_meta = {
        "stability_threshold": stability_threshold,
        "target_stable_features_per_modality": int(target_stable_features_per_modality),
        "n_stable_phrases": len(stable_phrase_df),
        "n_stable_groups": len(stable_group_df),
        "n_rediscovery_subsplits": n_subsplits,
        **cap_meta,
    }

    print(
        f"[REDISCOVERY] Stable lexicon summary: "
        f"n_stable_phrases={len(stable_phrase_df)} "
        f"n_stable_groups={len(stable_group_df)} "
        f"threshold={stability_threshold:.3f}"
    )

    return phrase_freq_df, group_freq_df, stable_phrase_df, stable_group_df, lexicon_meta
