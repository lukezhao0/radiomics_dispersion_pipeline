"""Metric comparison and diagnostic plots."""

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
# Ranking, stability, plotting
# -----------------------------

def rank_features_across_models(stability_tables: List[pd.DataFrame], task_type: str) -> pd.DataFrame:
    tables = [df for df in stability_tables if len(df) and df["task_type"].iloc[0] == task_type]
    if not tables:
        return pd.DataFrame(columns=["target_name", "feature", "n_nonzero", "mean_abs_coef", "models"])

    all_df = pd.concat(tables, ignore_index=True)
    if "target_name" not in all_df.columns:
        all_df["target_name"] = "unknown"
    group_cols = ["target_name", "feature"]
    agg = (
        all_df.groupby(group_cols)
        .agg(
            n_nonzero=("abs_coef", lambda s: int((pd.Series(s) > COEF_ZERO_TOL).sum())),
            mean_abs_coef=("abs_coef", "mean"),
            median_abs_coef=("abs_coef", "median"),
            models=("model_key", lambda s: ";".join(sorted(set(map(str, s))))),
        )
        .reset_index()
        .sort_values(["target_name", "n_nonzero", "mean_abs_coef"], ascending=[True, False, False])
        .reset_index(drop=True)
    )
    return agg


def summarize_coefficient_sign_stability(coef_all: pd.DataFrame) -> pd.DataFrame:
    if len(coef_all) == 0:
        return pd.DataFrame()
    df = coef_all.copy()
    df["sign"] = np.sign(df["coef"].astype(float))
    df["nonzero"] = (df["abs_coef"].astype(float) > COEF_ZERO_TOL).astype(int)
    if "target_name" not in df.columns:
        df["target_name"] = "unknown"
    agg = df.groupby(["dataset_key", "representation", "model_key", "task_type", "target_name", "feature"]).agg(
        n_rows=("coef", "size"),
        n_outer_splits=("split_id", "nunique"),
        n_nonzero=("nonzero", "sum"),
        mean_coef=("coef", "mean"),
        median_coef=("coef", "median"),
        mean_abs_coef=("abs_coef", "mean"),
        n_positive=("sign", lambda s: int((pd.Series(s) > 0).sum())),
        n_negative=("sign", lambda s: int((pd.Series(s) < 0).sum())),
    ).reset_index()
    agg["dominant_sign"] = np.where(agg["n_positive"] >= agg["n_negative"], "positive", "negative")
    agg["sign_consistency"] = agg[["n_positive", "n_negative"]].max(axis=1) / agg["n_rows"].replace(0, np.nan)
    agg = agg.sort_values(["n_nonzero", "sign_consistency", "mean_abs_coef"], ascending=[False, False, False]).reset_index(drop=True)
    return agg


def _metric_plot_label(df: pd.DataFrame) -> pd.Series:
    target = df["target_name"].astype(str) if "target_name" in df.columns else pd.Series("target", index=df.index)
    return target + " | " + df["dataset_key"].astype(str) + " | " + df["model_key"].astype(str) + " | " + df["representation"].astype(str)


def _metric_bar_figure_size(labels: Sequence[str], horizontal: bool = False) -> Tuple[float, float]:
    n = max(len(labels), 1)
    max_len = max((len(str(x)) for x in labels), default=10)
    if horizontal:
        return (max(10.0, min(0.45 * max_len + 4.0, 18.0)), max(4.5, min(0.38 * n + 1.5, 24.0)))
    return (max(12.0, min(0.55 * n + 2.0, 28.0)), max(6.0, min(0.18 * max_len + 4.5, 16.0)))


def _save_figure(path: str, fig=None) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if fig is None:
        fig = plt.gcf()
    with contextlib.suppress(Exception):
        fig.tight_layout(pad=1.2)
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.35)
    plt.close(fig)


def _plot_metric_bars(labels: Sequence[str], values: Sequence[float], ylabel: str, title: str, out_png: str, ascending: bool = False) -> None:
    if len(labels) == 0:
        return
    order = np.argsort(values)
    if not ascending:
        order = order[::-1]
    labels = [str(labels[i]) for i in order]
    values = [float(values[i]) for i in order]
    fig_w, fig_h = _metric_bar_figure_size(labels, horizontal=True)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ypos = np.arange(len(labels))
    ax.barh(ypos, values)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel(ylabel)
    ax.set_title(title)
    _save_figure(out_png, fig)


def _has_task_type_column(metrics_df: pd.DataFrame) -> bool:
    return len(metrics_df) > 0 and "task_type" in metrics_df.columns


def plot_classification_comparison(metrics_df: pd.DataFrame, out_png: str) -> None:
    if not _has_task_type_column(metrics_df):
        return
    df = metrics_df[metrics_df["task_type"] == "classification"].copy()
    if len(df) == 0 or "auroc" not in df.columns:
        return
    df["label"] = _metric_plot_label(df)
    _plot_metric_bars(
        df["label"].tolist(),
        df["auroc"].astype(float).tolist(),
        ylabel="AUROC",
        title="Nested held-out classification AUROC",
        out_png=out_png,
    )


def plot_relapse_metric_comparison(metrics_df: pd.DataFrame, out_png: str, metric: str) -> None:
    if not _has_task_type_column(metrics_df) or "target_name" not in metrics_df.columns:
        return
    df = metrics_df[(metrics_df["task_type"] == "classification") & (metrics_df["target_name"] == TARGET_NAME_RELAPSE_STATUS)].copy()
    if len(df) == 0 or metric not in df.columns:
        return
    df = df.sort_values(metric, ascending=(metric == "brier")).copy()
    df["label"] = _metric_plot_label(df)
    _plot_metric_bars(
        df["label"].tolist(),
        df[metric].astype(float).tolist(),
        ylabel=metric.upper(),
        title=f"Nested held-out relapse prediction {metric.upper()}",
        out_png=out_png,
        ascending=(metric == "brier"),
    )


def plot_relapse_curves(pred_all: pd.DataFrame, metrics_df: pd.DataFrame, out_dir: str, top_n: int = 8) -> None:
    if len(pred_all) == 0 or "target_name" not in pred_all.columns or "task_type" not in pred_all.columns:
        return
    rel_pred = pred_all[(pred_all["task_type"] == "classification") & (pred_all["target_name"] == TARGET_NAME_RELAPSE_STATUS)].copy()
    rel_metrics = metrics_df[(metrics_df["task_type"] == "classification") & (metrics_df["target_name"] == TARGET_NAME_RELAPSE_STATUS)].copy() if _has_task_type_column(metrics_df) and "target_name" in metrics_df.columns else pd.DataFrame()
    if len(rel_pred) == 0 or len(rel_metrics) == 0:
        return
    rel_metrics = rel_metrics.sort_values(["auroc", "auprc"], ascending=[False, False]).head(top_n)

    curve_specs = [
        ("roc", "nested_relapse_roc_curves_top_models.png", "False positive rate", "True positive rate"),
        ("pr", "nested_relapse_pr_curves_top_models.png", "Recall", "Precision"),
    ]
    for curve_type, filename, xlabel, ylabel in curve_specs:
        fig, ax = plt.subplots(figsize=(9, 7))
        any_curve = False
        for _, r in rel_metrics.iterrows():
            mask = (
                (rel_pred["dataset_key"].astype(str) == str(r["dataset_key"]))
                & (rel_pred["representation"].astype(str) == str(r["representation"]))
                & (rel_pred["model_key"].astype(str) == str(r["model_key"]))
            )
            sub = rel_pred.loc[mask].copy()
            if len(sub) < 2 or sub["y_true"].nunique() < 2:
                continue
            y_true = sub["y_true"].astype(int).values
            prob = sub["y_prob"].astype(float).values
            label = f"{r['dataset_key']} | {r['representation']} | {r['model_key']}"
            if curve_type == "roc":
                x, y, _ = roc_curve(y_true, prob)
            else:
                y, x, _ = precision_recall_curve(y_true, prob)
            ax.plot(x, y, label=label[:90])
            any_curve = True
        if any_curve:
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title("Held-out relapse prediction curves")
            ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.02, 0.5))
            _save_figure(os.path.join(out_dir, filename), fig)
        else:
            plt.close(fig)


def plot_regression_error_comparison(metrics_df: pd.DataFrame, out_png: str) -> None:
    if not _has_task_type_column(metrics_df):
        return
    df = metrics_df[metrics_df["task_type"] == "regression"].copy()
    if len(df) == 0 or "mae" not in df.columns:
        return
    df = df.sort_values("mae", ascending=True)
    df["label"] = _metric_plot_label(df)
    _plot_metric_bars(
        df["label"].tolist(),
        df["mae"].astype(float).tolist(),
        ylabel="MAE",
        title="Nested held-out regression MAE",
        out_png=out_png,
        ascending=True,
    )


def plot_regression_correlation_comparison(metrics_df: pd.DataFrame, out_png: str) -> None:
    if not _has_task_type_column(metrics_df):
        return
    df = metrics_df[metrics_df["task_type"] == "regression"].copy()
    if len(df) == 0 or "spearman_rho" not in df.columns:
        return
    df = df.sort_values("spearman_rho", ascending=False)
    df["label"] = _metric_plot_label(df)
    _plot_metric_bars(
        df["label"].tolist(),
        df["spearman_rho"].astype(float).tolist(),
        ylabel="Spearman rho",
        title="Nested held-out regression rank correlation",
        out_png=out_png,
    )
