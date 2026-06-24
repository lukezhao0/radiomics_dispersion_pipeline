"""Cross-modal reliability and weighted MRI concept scores."""

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
# Cross-modal calibration and weighted concept features
# -----------------------------

def _group_presence_cols(df: pd.DataFrame) -> Dict[str, str]:
    out = {}
    for c in df.columns:
        if c.startswith("grp__") and c.endswith("__present"):
            concept = c[len("grp__"):-len("__present")]
            out[concept] = c
    return out


def _group_status_value(df: pd.DataFrame, concept: str, uncertain_value: float, negated_value: float) -> pd.Series:
    present = df.get(f"grp__{concept}__present", pd.Series(0.0, index=df.index)).astype(float)
    uncertain = df.get(f"grp__{concept}__uncertain_count", pd.Series(0.0, index=df.index)).astype(float)
    negated = df.get(f"grp__{concept}__negated_count", pd.Series(0.0, index=df.index)).astype(float)
    return present + uncertain_value * (uncertain > 0).astype(float) + negated_value * (negated > 0).astype(float)


def compute_cross_modal_reliability(
    mri_group_matrix_df: pd.DataFrame,
    path_group_matrix_df: pd.DataFrame,
    train_case_ids: Sequence[str],
    split_id: str,
    smoothing: float = 0.5,
) -> pd.DataFrame:
    train_case_ids = set(map(str, train_case_ids))
    mri = mri_group_matrix_df[mri_group_matrix_df["case_id"].astype(str).isin(train_case_ids)].copy()
    path = path_group_matrix_df[path_group_matrix_df["case_id"].astype(str).isin(train_case_ids)].copy()
    merged = mri[["case_id", "row_index"] + list(_group_presence_cols(mri).values())].merge(
        path[["case_id", "row_index"] + list(_group_presence_cols(path).values())],
        on=["case_id", "row_index"],
        suffixes=("__mri", "__path"),
        how="inner",
    )
    rows: List[Dict[str, Any]] = []
    if len(merged) == 0:
        return pd.DataFrame(rows)

    mri_cols = [c for c in merged.columns if c.startswith("grp__") and c.endswith("__present__mri")]
    path_cols = [c for c in merged.columns if c.startswith("grp__") and c.endswith("__present__path")]

    for m_col in mri_cols:
        m = m_col[len("grp__"):-len("__present__mri")]
        m_vec = (pd.to_numeric(merged[m_col], errors="coerce").fillna(0.0).values > 0).astype(int)
        for g_col in path_cols:
            g = g_col[len("grp__"):-len("__present__path")]
            g_vec = (pd.to_numeric(merged[g_col], errors="coerce").fillna(0.0).values > 0).astype(int)
            n = len(m_vec)
            n_m = int(m_vec.sum())
            n_not_m = int((1 - m_vec).sum())
            n_g = int(g_vec.sum())
            n_mg = int(((m_vec == 1) & (g_vec == 1)).sum())
            n_notm_g = int(((m_vec == 0) & (g_vec == 1)).sum())
            n_m_notg = int(((m_vec == 1) & (g_vec == 0)).sum())
            n_notm_notg = int(((m_vec == 0) & (g_vec == 0)).sum())

            p_g = (n_g + smoothing) / (n + 2 * smoothing)
            p_g_given_m = (n_mg + smoothing) / (n_m + 2 * smoothing) if n_m > 0 else np.nan
            p_m_given_g = (n_mg + smoothing) / (n_g + 2 * smoothing) if n_g > 0 else np.nan
            p_g_given_not_m = (n_notm_g + smoothing) / (n_not_m + 2 * smoothing) if n_not_m > 0 else np.nan
            delta = p_g_given_m - p_g_given_not_m if np.isfinite(p_g_given_m) and np.isfinite(p_g_given_not_m) else np.nan
            lift = p_g_given_m / max(p_g, EPS) if np.isfinite(p_g_given_m) else np.nan
            odds_ratio = ((n_mg + smoothing) * (n_notm_notg + smoothing)) / max((n_m_notg + smoothing) * (n_notm_g + smoothing), EPS)
            mi = mutual_info_score(m_vec, g_vec) if len(np.unique(m_vec)) > 1 and len(np.unique(g_vec)) > 1 else 0.0

            rows.append({
                "split_id": split_id,
                "mri_concept": m,
                "path_concept": g,
                "n_train_pairs": n,
                "n_mri_present": n_m,
                "n_path_present": n_g,
                "n_both_present": n_mg,
                "p_path": p_g,
                "p_path_given_mri": p_g_given_m,
                "p_mri_given_path": p_m_given_g,
                "p_path_given_not_mri": p_g_given_not_m,
                "delta_p_path_given_mri": delta,
                "lift": lift,
                "odds_ratio": odds_ratio,
                "mutual_information": float(mi),
            })
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(["delta_p_path_given_mri", "lift", "n_both_present"], ascending=[False, False, False]).reset_index(drop=True)
    return out


def compute_weighted_mri_lexicon(
    reliability_df: pd.DataFrame,
    mri_group_freq_df: pd.DataFrame,
    mri_group_matrix_df: pd.DataFrame,
    train_case_ids: Sequence[str],
    y_train_continuous: pd.Series,
    split_id: str,
    min_selection_frequency: float,
    reliability_power: float,
    stability_power: float,
    association_power: float,
) -> pd.DataFrame:
    train_case_ids = set(map(str, train_case_ids))
    mri_train = mri_group_matrix_df[mri_group_matrix_df["case_id"].astype(str).isin(train_case_ids)].copy()
    y_map = pd.Series(y_train_continuous.values, index=list(map(str, y_train_continuous.index)))

    stability = defaultdict(float)
    if len(mri_group_freq_df):
        for _, r in mri_group_freq_df.iterrows():
            stability[str(r["canonical_group"])] = float(r.get("selection_frequency", 0.0))

    presence_cols = _group_presence_cols(mri_train)
    rows = []
    for concept, col in presence_cols.items():
        if concept == "other_candidate_feature":
            continue
        x = pd.to_numeric(mri_train[col], errors="coerce").fillna(0.0).values
        prevalence = float(np.mean(x > 0)) if len(x) else 0.0
        if prevalence <= 0:
            continue
        stab = max(stability.get(concept, 0.0), min_selection_frequency)
        if stab < min_selection_frequency:
            continue

        sub_rel = reliability_df[reliability_df["mri_concept"] == concept].copy() if len(reliability_df) else pd.DataFrame()
        if len(sub_rel):
            best = sub_rel.sort_values(["delta_p_path_given_mri", "lift", "n_both_present"], ascending=[False, False, False]).iloc[0]
            best_path_concept = str(best["path_concept"])
            rel_score = float(max(0.0, best.get("delta_p_path_given_mri", 0.0)))
            lift_score = float(max(0.0, min(best.get("lift", 1.0), 5.0) / 5.0))
            mi_score = float(max(0.0, best.get("mutual_information", 0.0)))
            concordance_score = 0.70 * rel_score + 0.20 * lift_score + 0.10 * mi_score
        else:
            best_path_concept = "none"
            rel_score = 0.0
            lift_score = 0.0
            mi_score = 0.0
            concordance_score = 0.0

        # Training-only univariate association with continuous dispersion.
        try:
            y = pd.to_numeric(mri_train["dispersion_true"], errors="coerce").values
            rho, _ = safe_spearman(x.astype(float), y.astype(float))
            assoc = abs(rho) if np.isfinite(rho) else 0.0
        except Exception:
            assoc = 0.0

        ambiguity_penalty = 0.55 if concept in AMBIGUITY_GROUPS else 1.0
        prevalence_penalty = min(1.0, prevalence / 0.05) if prevalence < 0.05 else 1.0
        raw_weight = (
            (max(stab, EPS) ** stability_power)
            * (max(concordance_score, EPS) ** reliability_power)
            * ((1.0 + assoc) ** association_power)
            * ambiguity_penalty
            * prevalence_penalty
        )
        rows.append({
            "split_id": split_id,
            "mri_concept": concept,
            "best_path_concept": best_path_concept,
            "weight": float(raw_weight),
            "selection_frequency": float(stab),
            "mri_prevalence_train": prevalence,
            "path_concordance_score": float(concordance_score),
            "delta_component": rel_score,
            "lift_component": lift_score,
            "mutual_information_component": mi_score,
            "abs_spearman_with_dispersion": float(assoc),
            "ambiguity_penalty": ambiguity_penalty,
            "weight_formula": "stability^stability_power * concordance^reliability_power * (1+association)^association_power * ambiguity_penalty * prevalence_penalty",
        })
    out = pd.DataFrame(rows)
    if len(out):
        max_w = float(out["weight"].max())
        if max_w > 0:
            out["weight_normalized"] = out["weight"] / max_w
        else:
            out["weight_normalized"] = 0.0
        out = out.sort_values(["weight_normalized", "path_concordance_score", "mri_prevalence_train"], ascending=[False, False, False]).reset_index(drop=True)
    return out


def build_weighted_mri_concept_score_matrix(
    mri_group_matrix_df: pd.DataFrame,
    weighted_lexicon_df: pd.DataFrame,
    uncertain_value: float,
    negated_value: float,
    split_id: str,
) -> pd.DataFrame:
    base = mri_group_matrix_df[["case_id", "row_index", "dispersion_true", "dispersion_true_high_low", "relapse_true", "split_id", "report_mode"]].copy()
    out = base.copy()
    if len(weighted_lexicon_df) == 0:
        return out
    for _, r in weighted_lexicon_df.iterrows():
        concept = str(r["mri_concept"])
        w = float(r.get("weight_normalized", r.get("weight", 0.0)))
        status = _group_status_value(mri_group_matrix_df, concept, uncertain_value, negated_value)
        out[f"wgrp__{concept}__pathcal_score"] = status.values * w
        out[f"wgrp__{concept}__status_unweighted"] = status.values
    out = out.fillna(0.0)
    return out


def randomized_or_mismatched_path_matrix(path_group_matrix_df: pd.DataFrame, train_case_ids: Sequence[str], mode: str, random_seed: int) -> pd.DataFrame:
    rng = np.random.RandomState(random_seed)
    out = path_group_matrix_df.copy()
    train_mask = out["case_id"].astype(str).isin(set(map(str, train_case_ids)))
    feature_cols = [c for c in out.columns if c.startswith("grp__")]
    if not feature_cols or train_mask.sum() <= 1:
        return out
    if mode == "randomized_labels":
        for c in feature_cols:
            vals = out.loc[train_mask, c].values.copy()
            rng.shuffle(vals)
            out.loc[train_mask, c] = vals
    elif mode == "mismatched_pairing":
        vals_df = out.loc[train_mask, feature_cols].sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
        out.loc[train_mask, feature_cols] = vals_df.values
    else:
        raise ValueError(mode)
    return out
