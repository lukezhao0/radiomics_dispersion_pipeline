"""Automated reports, diagnostics, and plot regeneration."""

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

from approach2.eval_data import _mri_missing_row_indices, _raw_df_with_row_index
from common.cost_comparison import (
    APPROACH2_APRIORI,
    build_cost_comparison_summary_df,
    extract_actual_cumulative,
    generate_cost_comparison_plots,
    normalize_apriori_estimate,
)
from approach2.evaluation.plots import (
    _metric_bar_figure_size,
    _metric_plot_label,
    _plot_metric_bars,
    _save_figure,
    plot_classification_comparison,
    plot_relapse_curves,
    plot_relapse_metric_comparison,
    plot_regression_correlation_comparison,
    plot_regression_error_comparison,
)
from approach2.models_ml import classification_metrics, regression_metrics
from approach2.orchestration import (
    _best_row,
    compute_metrics_from_predictions,
    deduplicate_outer_predictions,
)

def summarize_metrics(metrics_df: pd.DataFrame) -> str:
    lines: List[str] = []
    lines.append("=== Nested Resampling Predictive Evaluation Summary ===")
    if len(metrics_df) == 0:
        lines.append("No metrics were generated.")
        return "\n".join(lines)

    cls = metrics_df[metrics_df["task_type"] == "classification"].copy()
    reg = metrics_df[metrics_df["task_type"] == "regression"].copy()

    if len(cls):
        if "target_name" not in cls.columns:
            cls["target_name"] = TARGET_NAME_DISPERSION_HIGH_LOW
        for target_name, target_df in cls.groupby("target_name"):
            target_df = target_df.copy()
            sort_cols = [c for c in ["auroc", "auprc", "f1"] if c in target_df.columns]
            if not sort_cols:
                continue
            best_cls = target_df.sort_values(sort_cols, ascending=[False] * len(sort_cols)).iloc[0]
            lines.append(
                f"Best classification setting for {target_name}: "
                f"{best_cls['dataset_key']} | {best_cls['representation']} | {best_cls['model_key']} "
                f"(AUROC={best_cls.get('auroc', np.nan):.3f}, AUPRC={best_cls.get('auprc', np.nan):.3f}, "
                f"F1={best_cls.get('f1', np.nan):.3f}, Brier={best_cls.get('brier', np.nan):.3f})"
            )

    if len(reg):
        best_reg = reg.sort_values(["mae", "spearman_rho"], ascending=[True, False]).iloc[0]
        lines.append(
            "Best regression setting: "
            f"{best_reg['dataset_key']} | {best_reg['representation']} | {best_reg['model_key']} "
            f"(MAE={best_reg['mae']:.3f}, RMSE={best_reg['rmse']:.3f}, "
            f"R2={best_reg['r2']:.3f}, Spearman={best_reg['spearman_rho']:.3f})"
        )

    lines.append("")
    lines.append("All metrics below are aggregated over outer held-out predictions only.")
    return "\n".join(lines)


def write_methodology_markdown(out_dir: str, args: argparse.Namespace) -> str:
    path = os.path.join(out_dir, "PATHOLOGY_INFORMED_MRI_REFINEMENT_SUMMARY.md")
    text = f"""# Pathology-Informed MRI Lexical Refinement: Implementation Summary

## What changed

This run extends the original nested NLP + ML pipeline with pathology-informed MRI refinement while preserving the original leakage-aware evaluation design.

### Added outputs

- `shared_biological_concept_ontology.csv`: concept definitions, examples, and mapping regexes.
- per-split `*_mri_audit_case_table.csv`: MRI report length, extraction/concept densities, uncertainty/negation densities, and section flags.
- per-split `*_mri_audit_density_summary.csv`: high/low and relapse-stratified audit summaries when labels are available.
- per-split `*_mri_pathology_reliability_matrix.csv`: training-only MRI-concept to pathology-concept concordance.
- per-split `*_weighted_mri_lexicon.csv`: pathology-informed MRI concept weights learned from outer-training cases only.
- per-split `*_weighted_mri_concept_score_matrix.csv`: MRI-only weighted concept scores applied to train/test cases using frozen weights.
- optional ablation weighted matrices for randomized pathology labels and mismatched MRI-pathology pairing.
- relapse-status predictions from the same dispersion-vector feature matrices used for high/low dispersion classification.
- process-wide fold-level concurrency when `--parallel-fold-workers > 1`; each outer fold writes to an isolated split directory with its own frozen lexicons, calibration files, logs/checkpoints, and split provenance manifest.
- process-wide API concurrency control through a global semaphore, so `--max-api-workers` caps active API calls across all folds/modalities/cases rather than multiplying silently.
- automated performance reports: `automated_results_report.md` and `automated_results_report.html`, plus interleaved plots and deduplicated one-prediction-per-case metrics.
- interpretability reports: `interpretability_report.md` and `interpretability_report.html`, coefficient annotations, feature-density summaries, reliability heatmaps, and calibration-weight plots.
- missed-case/error reports: `missed_case_error_analysis.csv` and `missed_case_error_analysis.md`.
- relapse-specific AUROC, AUPRC, no-skill AUPRC baseline, F1, Brier, precision/PPV, recall/sensitivity, specificity, NPV, balanced accuracy, confusion-matrix counts, calibration diagnostics, bootstrap confidence intervals, class-balance diagnostics, and permutation tests.
- MRI-missing cases are explicitly skipped for MRI-only, MRI+pathology combined, pathology-calibrated MRI, and teacher-student MRI evaluations. Pathology-only evaluations retain these cases.
- `nested_feature_sign_stability.csv`: sign-consistency and coefficient-stability summary across outer splits.
- `logs/run_log_feature_discovery_nested_eval.txt`: persisted stdout/stderr log for resume/debugging.

## Leakage protocol

For the primary scientific claim, the following operations are restricted to each outer-training split:

1. LLM extraction.
2. rediscovery and stable lexicon definition.
3. MRI-pathology reliability estimation.
4. weighted MRI lexicon derivation.
5. pathology-teacher score construction.
6. teacher-student training.

Outer-test pathology is not used to calibrate, weight, or train MRI features. Outer-test prediction uses MRI-derived features only.

## Main new dataset/model keys

- `mri`: original MRI-only lexical baseline.
- `path`: original pathology-only lexical baseline.
- `combined`: original MRI + pathology early-fusion baseline.
- `mri_pathcal_weighted`: MRI-only weighted concept-score model using training-only pathology calibration.
- `mri_pathcal_weighted_random_pathology`: optional negative-control ablation.
- `mri_pathcal_weighted_mismatched_pairing`: optional pairing-control ablation.
- `mri_teacher_student`: optional MRI-only multi-task ridge student trained with pathology-derived training targets.

## Ablation meaning in this code

An ablation is a deliberately altered control run that removes or corrupts one component of the proposed method while keeping the rest of the pipeline similar. Here, randomized-pathology and mismatched-pairing ablations test whether pathology-calibrated MRI weights help because of real MRI-pathology biological concordance, rather than because any extra weighting or regularization improves performance nonspecifically.

## Important critique and limitations

1. The ontology is rule-based. It improves interpretability and robustness but can miss synonyms, local report conventions, and context-dependent meanings.
2. Regex recoding is not a substitute for full clinical text understanding. It is intended as a leakage-safe, frozen representation layer after LLM discovery.
3. Pathology is a teacher, not truth. Concordance with pathology may downweight MRI features that are radiographically meaningful but not sampled or described pathologically.
4. Weighted lexicon formulas are heuristic and should be ablated. Improved performance under randomized or mismatched pathology ablations would indicate overfitting or non-specific regularization rather than true biological supervision.
5. The teacher-student model is intentionally simple. With ~100 cases, complex neural multi-task learning would likely overfit unless externally validated.
6. Internal nested performance is still small-sample and high-variance. External validation remains necessary before clinical claims.

## Run configuration

```text
pathology_calibration_enabled = {args.enable_pathology_calibration}
teacher_student_enabled = {args.enable_teacher_student}
ontology_groups_mode = {args.ontology_groups_mode}
weighted_uncertain_value = {args.weighted_uncertain_value}
weighted_negated_value = {args.weighted_negated_value}
calibration_smoothing = {args.calibration_smoothing}
run_calibration_ablations = {args.run_calibration_ablations}
parallel_fold_workers = {getattr(args, "parallel_fold_workers", 1)}
parallel_modality_workers = {getattr(args, "parallel_modality_workers", 1)}
max_api_workers_global = {getattr(args, "max_api_workers", 1)}
ml_n_jobs = {getattr(args, "ml_n_jobs", 1)}
bootstrap_n = {getattr(args, "bootstrap_n", 0)}
relapse_permutation_n = {getattr(args, "relapse_permutation_n", 0)}
```
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path
def _df_to_markdown(df: pd.DataFrame, max_rows: int = 25, float_digits: int = 3) -> str:
    if df is None or len(df) == 0:
        return "_No rows available._"
    tmp = df.head(max_rows).copy()
    for col in tmp.columns:
        if pd.api.types.is_float_dtype(tmp[col]):
            tmp[col] = tmp[col].map(lambda x: "" if pd.isna(x) else f"{x:.{float_digits}f}")
    try:
        return tmp.to_markdown(index=False)
    except Exception:
        return tmp.to_csv(index=False)


def _safe_savefig(path: str) -> None:
    _save_figure(path)


def _read_json_if_exists(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _load_outer_split_summaries(out_dir: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    splits_root = os.path.join(out_dir, "outer_splits")
    if not os.path.isdir(splits_root):
        return pd.DataFrame()
    for split_name in sorted(os.listdir(splits_root)):
        split_dir = os.path.join(splits_root, split_name)
        if not os.path.isdir(split_dir):
            continue
        manifest_path = os.path.join(split_dir, f"{split_name}_split_manifest.json")
        manifest = _read_json_if_exists(manifest_path)
        prov_path = os.path.join(split_dir, f"{split_name}_split_provenance.csv")
        prov_n = 0
        if os.path.isfile(prov_path):
            with contextlib.suppress(Exception):
                prov_n = len(pd.read_csv(prov_path))
        rows.append({
            "split_id": split_name,
            "n_train": manifest.get("n_train", manifest.get("train_n", np.nan)),
            "n_test": manifest.get("n_test", manifest.get("test_n", np.nan)),
            "provenance_rows": prov_n,
            "checkpoint_status": (
                "completed" if os.path.isfile(os.path.join(split_dir, "_split_resume_checkpoint", "COMPLETED.json"))
                else ("failed" if os.path.isfile(os.path.join(split_dir, "_split_resume_checkpoint", "FAILED.json")) else "unknown")
            ),
        })
    return pd.DataFrame(rows)


def _load_run_metadata(out_dir: str) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}
    for key, fname in [
        ("cost_apriori", "llm_cost_estimate_apriori.json"),
        ("cost_post_run", "llm_token_cost_report.json"),
        ("cohort_availability", "cohort_report_availability_summary.json"),
    ]:
        meta[key] = _read_json_if_exists(os.path.join(out_dir, fname))
    mri_missing_path = os.path.join(out_dir, "mri_missing_case_summary.csv")
    meta["mri_missing_df"] = _safe_read_csv_if_exists(mri_missing_path)
    summary_path = os.path.join(out_dir, "nested_resampling_summary.txt")
    if os.path.isfile(summary_path):
        with open(summary_path, "r", encoding="utf-8") as f:
            meta["nested_summary_text"] = f.read().strip()
    else:
        meta["nested_summary_text"] = ""
    meta["split_errors_df"] = _safe_read_csv_if_exists(os.path.join(out_dir, "nested_outer_split_errors.csv"))
    meta["split_summary_df"] = _load_outer_split_summaries(out_dir)
    return meta


def _build_cost_comparison_report_parts(
    out_dir: str,
    cost_apriori: Optional[Dict[str, Any]],
    cost_post_run: Optional[Dict[str, Any]],
) -> Tuple[List[str], List[str]]:
    """Return (markdown_lines, html_parts) for estimate-vs-actual cost comparison."""
    apriori_norm = normalize_apriori_estimate(cost_apriori, flavor=APPROACH2_APRIORI)
    actual_norm = extract_actual_cumulative(cost_post_run)
    if not apriori_norm:
        return [], []
    if not actual_norm:
        return [], []

    comparison_df = build_cost_comparison_summary_df(apriori_norm, actual_norm)
    plot_paths = generate_cost_comparison_plots(out_dir, apriori_norm, actual_norm)

    md_lines = ["## Cost estimate vs actual\n"]
    md_lines.append(
        "A-priori estimates use rendered extraction prompts and the configured completion-token cap. "
        "Post-run actuals come from cumulative API usage metadata.\n"
    )
    md_lines.append(_df_to_markdown(comparison_df, max_rows=20))
    md_lines.append("\n")
    for plot_path in plot_paths:
        md_lines.append(f"- `{os.path.relpath(plot_path, out_dir)}`")
    md_lines.append("")

    html_parts: List[str] = [
        html_paragraph(
            "Before LLM extraction calls, the pipeline estimates token usage and USD cost from the exact "
            "prompts that will be sent plus the configured max completion tokens per call. After the run, "
            "cumulative usage is taken from API billing metadata. Completion tokens in the estimate are an "
            "upper bound; actual completion is typically lower, so actual cost often falls below the "
            "cache-aware estimate."
        ),
        df_to_html_table(comparison_df, max_rows=20, float_digits=4),
    ]
    for plot_path in plot_paths:
        html_parts.append(
            html_plot_block(
                plot_path,
                os.path.relpath(plot_path, out_dir),
                title=os.path.basename(plot_path).replace("_", " ").replace(".png", ""),
            )
        )
    return md_lines, html_parts


def _summarize_label_distributions(pred_case_df: pd.DataFrame) -> pd.DataFrame:
    if pred_case_df is None or len(pred_case_df) == 0:
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    dedup = pred_case_df.drop_duplicates(subset=["case_id", "target_name", "dataset_key", "representation", "model_key"], keep="first")
    for target_name, sub in dedup.groupby("target_name"):
        task = str(sub["task_type"].iloc[0]) if "task_type" in sub.columns else "unknown"
        y = sub["y_true"]
        if task == "regression":
            yy = pd.to_numeric(y, errors="coerce").dropna()
            rows.append({
                "target_name": target_name,
                "task_type": task,
                "n_cases": int(len(yy)),
                "mean": float(yy.mean()) if len(yy) else np.nan,
                "std": float(yy.std(ddof=0)) if len(yy) else np.nan,
                "min": float(yy.min()) if len(yy) else np.nan,
                "max": float(yy.max()) if len(yy) else np.nan,
            })
        else:
            yy = pd.to_numeric(y, errors="coerce").dropna()
            n_pos = int((yy == 1).sum())
            n_neg = int((yy == 0).sum())
            rows.append({
                "target_name": target_name,
                "task_type": task,
                "n_cases": int(len(yy)),
                "positive": n_pos,
                "negative": n_neg,
                "prevalence": float(n_pos / len(yy)) if len(yy) else np.nan,
            })
    return pd.DataFrame(rows)


def _plot_pathway_modality_comparison(metrics_df: pd.DataFrame, out_png: str) -> bool:
    if metrics_df is None or len(metrics_df) == 0:
        return False
    panels: List[Tuple[str, pd.DataFrame, str, bool]] = []
    for task_type, target_name, metric, ascending in [
        ("regression", TARGET_NAME_DISPERSION_SCORE, "mae", True),
        ("classification", TARGET_NAME_DISPERSION_HIGH_LOW, "auroc", False),
        ("classification", TARGET_NAME_RELAPSE_STATUS, "auroc", False),
    ]:
        sub = metrics_df[(metrics_df["task_type"] == task_type) & (metrics_df["target_name"] == target_name)].copy()
        if len(sub) == 0 or metric not in sub.columns:
            continue
        sub = sub.sort_values(metric, ascending=ascending).head(20)
        sub["label"] = sub["dataset_key"].astype(str) + " | " + sub["model_key"].astype(str)
        panels.append((f"{target_name} ({metric})", sub, metric, ascending))
    if not panels:
        return False
    fig_h = max(4.0 * len(panels), 6.0)
    fig, axes = plt.subplots(len(panels), 1, figsize=(11, fig_h))
    if len(panels) == 1:
        axes = [axes]
    for ax, (title, sub, metric, ascending) in zip(axes, panels):
        ypos = np.arange(len(sub))
        ax.barh(ypos, sub[metric].astype(float), color="#4C78A8")
        ax.set_yticks(ypos)
        ax.set_yticklabels(sub["label"], fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel(metric.upper())
        ax.set_title(title)
    fig.suptitle("Pathway and modality comparison", y=1.01, fontsize=12)
    fig.tight_layout()
    _save_figure(out_png, fig)
    return True


def _plot_per_fold_regression_mae(fold_results_all: pd.DataFrame, metrics_df: pd.DataFrame, out_png: str) -> bool:
    if fold_results_all is None or len(fold_results_all) == 0:
        return False
    best = _best_row(metrics_df, "regression", TARGET_NAME_DISPERSION_SCORE)
    if best is None:
        return False
    sub = fold_results_all[
        (fold_results_all["task_type"] == "regression")
        & (fold_results_all["target_name"] == TARGET_NAME_DISPERSION_SCORE)
        & (fold_results_all["dataset_key"].astype(str) == str(best["dataset_key"]))
        & (fold_results_all["representation"].astype(str) == str(best["representation"]))
        & (fold_results_all["model_key"].astype(str) == str(best["model_key"]))
    ].copy()
    if len(sub) == 0 or "mae" not in sub.columns or "split_id" not in sub.columns:
        return False
    sub = sub.sort_values("split_id")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(sub["split_id"].astype(str), sub["mae"].astype(float))
    ax.set_ylabel("MAE")
    ax.set_xlabel("Outer split")
    ax.set_title(f"Per-fold MAE: {best['dataset_key']} | {best['model_key']}")
    plt.xticks(rotation=45, ha="right")
    _save_figure(out_png, fig)
    return True


def generate_performance_plots(pred_case_df: pd.DataFrame, metrics_df: pd.DataFrame, out_dir: str) -> List[str]:
    plot_dir = os.path.join(out_dir, "report_plots")
    os.makedirs(plot_dir, exist_ok=True)
    paths: List[str] = []
    if pred_case_df is None or len(pred_case_df) == 0 or metrics_df is None or len(metrics_df) == 0:
        return paths

    best_reg = _best_row(metrics_df, "regression", TARGET_NAME_DISPERSION_SCORE)
    if best_reg is not None:
        mask = (
            (pred_case_df["dataset_key"].astype(str) == str(best_reg["dataset_key"]))
            & (pred_case_df["representation"].astype(str) == str(best_reg["representation"]))
            & (pred_case_df["model_key"].astype(str) == str(best_reg["model_key"]))
            & (pred_case_df["task_type"].astype(str) == "regression")
        )
        sub = pred_case_df.loc[mask].copy()
        if len(sub):
            y_true = pd.to_numeric(sub["y_true"], errors="coerce")
            y_pred = pd.to_numeric(sub["y_pred_value"], errors="coerce")
            plt.figure(figsize=(6, 6))
            plt.scatter(y_true, y_pred)
            lo = float(np.nanmin([y_true.min(), y_pred.min()]))
            hi = float(np.nanmax([y_true.max(), y_pred.max()]))
            plt.plot([lo, hi], [lo, hi], linestyle="--")
            plt.xlabel("True dispersion score")
            plt.ylabel("Predicted dispersion score")
            plt.title("Top regression model: predicted vs true")
            path = os.path.join(plot_dir, "top_regression_predicted_vs_true.png")
            _safe_savefig(path); paths.append(path)

            residual = y_pred - y_true
            plt.figure(figsize=(7, 5))
            plt.scatter(y_pred, residual)
            plt.axhline(0, linestyle="--")
            plt.xlabel("Predicted dispersion score")
            plt.ylabel("Residual: predicted - true")
            plt.title("Top regression model residuals")
            path = os.path.join(plot_dir, "top_regression_residuals.png")
            _safe_savefig(path); paths.append(path)

            ranks_true = y_true.rank(method="average")
            ranks_pred = y_pred.rank(method="average")
            rho, _ = safe_spearman(y_true.values, y_pred.values)
            plt.figure(figsize=(6, 6))
            plt.scatter(ranks_true, ranks_pred, alpha=0.75)
            lo, hi = 0.5, float(max(ranks_true.max(), ranks_pred.max(), 1.0)) + 0.5
            plt.plot([lo, hi], [lo, hi], linestyle="--", color="gray")
            plt.xlabel("Rank of true dispersion score")
            plt.ylabel("Rank of predicted dispersion score")
            plt.title(f"Top regression model: rank plot (Spearman rho={rho:.3f})")
            path = os.path.join(plot_dir, "top_regression_spearman_rank.png")
            _safe_savefig(path); paths.append(path)

    for target_name in [TARGET_NAME_DISPERSION_HIGH_LOW, TARGET_NAME_RELAPSE_STATUS]:
        best_cls = _best_row(metrics_df, "classification", target_name)
        if best_cls is None:
            continue
        mask = (
            (pred_case_df["dataset_key"].astype(str) == str(best_cls["dataset_key"]))
            & (pred_case_df["representation"].astype(str) == str(best_cls["representation"]))
            & (pred_case_df["model_key"].astype(str) == str(best_cls["model_key"]))
            & (pred_case_df["target_name"].astype(str) == str(target_name))
        )
        sub = pred_case_df.loc[mask].copy()
        if len(sub) == 0:
            continue
        y_true = sub["y_true"].astype(int).values
        prob = sub["y_prob"].astype(float).values
        y_pred = sub["y_pred"].astype(int).values
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        plt.figure(figsize=(5, 4))
        plt.imshow(cm)
        plt.xticks([0, 1], ["Pred 0", "Pred 1"])
        plt.yticks([0, 1], ["True 0", "True 1"])
        for i in range(2):
            for j in range(2):
                plt.text(j, i, str(cm[i, j]), ha="center", va="center")
        plt.title(f"Confusion matrix: {target_name}")
        path = os.path.join(plot_dir, f"top_{target_name}_confusion_matrix.png")
        _safe_savefig(path); paths.append(path)

        if len(np.unique(y_true)) >= 2:
            fpr, tpr, _ = roc_curve(y_true, prob)
            plt.figure(figsize=(6, 5))
            plt.plot(fpr, tpr)
            plt.plot([0, 1], [0, 1], linestyle="--")
            plt.xlabel("False positive rate")
            plt.ylabel("True positive rate")
            plt.title(f"ROC curve: {target_name}")
            path = os.path.join(plot_dir, f"top_{target_name}_roc.png")
            _safe_savefig(path); paths.append(path)

            precision, recall, _ = precision_recall_curve(y_true, prob)
            plt.figure(figsize=(6, 5))
            plt.plot(recall, precision)
            plt.axhline(float(np.mean(y_true)), linestyle="--")
            plt.xlabel("Recall / sensitivity")
            plt.ylabel("Precision / PPV")
            plt.title(f"Precision-recall curve: {target_name}")
            path = os.path.join(plot_dir, f"top_{target_name}_pr.png")
            _safe_savefig(path); paths.append(path)

            bins = np.linspace(0, 1, 6)
            bin_id = np.digitize(prob, bins, right=True)
            cal_rows = []
            for b in sorted(set(bin_id)):
                idx = bin_id == b
                if idx.sum() >= 1:
                    cal_rows.append((float(np.mean(prob[idx])), float(np.mean(y_true[idx])), int(idx.sum())))
            if cal_rows:
                cal = pd.DataFrame(cal_rows, columns=["mean_predicted_risk", "observed_rate", "n"])
                plt.figure(figsize=(6, 5))
                plt.scatter(cal["mean_predicted_risk"], cal["observed_rate"])
                plt.plot([0, 1], [0, 1], linestyle="--")
                for _, r in cal.iterrows():
                    plt.text(r["mean_predicted_risk"], r["observed_rate"], str(int(r["n"])))
                plt.xlabel("Mean predicted risk")
                plt.ylabel("Observed event rate")
                plt.title(f"Calibration plot: {target_name}")
                path = os.path.join(plot_dir, f"top_{target_name}_calibration.png")
                _safe_savefig(path); paths.append(path)

    if len(metrics_df):
        rank_df = metrics_df.copy()
        rank_df["label"] = _metric_plot_label(rank_df)
        for metric, title in [("mae", "Regression MAE"), ("auprc", "Classification AUPRC"), ("auroc", "Classification AUROC")]:
            sub = rank_df[rank_df[metric].notna()].copy() if metric in rank_df.columns else pd.DataFrame()
            if len(sub) == 0:
                continue
            sub = sub.sort_values(metric, ascending=(metric == "mae")).head(25)
            path = os.path.join(plot_dir, f"ranked_model_{metric}.png")
            _plot_metric_bars(
                sub["label"].tolist(),
                sub[metric].astype(float).tolist(),
                ylabel=metric,
                title=title,
                out_png=path,
                ascending=(metric == "mae"),
            )
            paths.append(path)

        ci_rows = []
        for _, r in metrics_df.iterrows():
            primary = "mae" if r.get("task_type") == "regression" else ("auprc" if r.get("target_name") == TARGET_NAME_RELAPSE_STATUS else "auroc")
            if primary in r and f"{primary}_ci_low" in r and pd.notna(r.get(primary)):
                ci_rows.append({
                    "label": f"{r.get('target_name')}|{r.get('dataset_key')}|{r.get('model_key')}"[:80],
                    "metric": primary,
                    "value": r.get(primary),
                    "ci_low": r.get(f"{primary}_ci_low"),
                    "ci_high": r.get(f"{primary}_ci_high"),
                })
        if ci_rows:
            ci = pd.DataFrame(ci_rows).head(30)
            fig_w, fig_h = _metric_bar_figure_size(ci["label"].astype(str).tolist(), horizontal=False)
            fig_h = max(fig_h, 7.0)
            fig, ax = plt.subplots(figsize=(fig_w, fig_h))
            xpos = np.arange(len(ci))
            yerr_low = np.maximum(0.0, ci["value"].astype(float) - ci["ci_low"].astype(float))
            yerr_high = np.maximum(0.0, ci["ci_high"].astype(float) - ci["value"].astype(float))
            ax.errorbar(xpos, ci["value"], yerr=[yerr_low, yerr_high], fmt="o")
            ax.set_xticks(xpos)
            ax.set_xticklabels(ci["label"], rotation=75, ha="right", fontsize=7)
            ax.set_ylabel("Primary metric with 95% bootstrap CI")
            ax.set_title("Bootstrap confidence intervals")
            path = os.path.join(plot_dir, "bootstrap_ci_primary_metrics.png")
            _save_figure(path, fig)
            paths.append(path)

    pathway_png = os.path.join(plot_dir, "pathway_modality_comparison.png")
    if _plot_pathway_modality_comparison(metrics_df, pathway_png):
        paths.append(pathway_png)

    return paths


def relapse_split_diagnostics(target_df: pd.DataFrame, outer_splits: Sequence[Tuple[np.ndarray, np.ndarray]], out_dir: str) -> pd.DataFrame:
    rows = []
    y_all = pd.to_numeric(target_df.get("relapse_true"), errors="coerce")
    rows.append({"split_id": "overall", "partition": "all", "n": int(y_all.notna().sum()), "relapse_positive": int((y_all == 1).sum()), "relapse_negative": int((y_all == 0).sum()), "prevalence": float((y_all == 1).mean()) if y_all.notna().any() else np.nan, "warning": ""})
    for split_num, (train_pos, test_pos) in enumerate(outer_splits, 1):
        split_id = f"outer_split_{split_num:03d}"
        for partition, pos in [("train", train_pos), ("test", test_pos)]:
            yy = pd.to_numeric(target_df.iloc[pos].get("relapse_true"), errors="coerce")
            n_pos = int((yy == 1).sum())
            n_neg = int((yy == 0).sum())
            warnings = []
            if n_pos == 0 or n_neg == 0:
                warnings.append("single_class_partition")
            if n_pos < 2:
                warnings.append("too_few_relapse_positive_for_stable_auc_or_calibration")
            rows.append({"split_id": split_id, "partition": partition, "n": int(yy.notna().sum()), "relapse_positive": n_pos, "relapse_negative": n_neg, "prevalence": float((yy == 1).mean()) if yy.notna().any() else np.nan, "warning": ";".join(warnings)})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(out_dir, "relapse_class_balance_by_split.csv"), index=False)
    return out


def permutation_test_relapse_metrics(pred_case_df: pd.DataFrame, metrics_df: pd.DataFrame, args: argparse.Namespace, out_dir: str) -> pd.DataFrame:
    n_perm = int(getattr(args, "relapse_permutation_n", 0) or 0)
    if n_perm <= 0 or pred_case_df is None or len(pred_case_df) == 0:
        return pd.DataFrame()
    rows = []
    rng = np.random.default_rng(int(args.random_seed) + 991)
    rel_metrics = metrics_df[(metrics_df["task_type"] == "classification") & (metrics_df["target_name"] == TARGET_NAME_RELAPSE_STATUS)].copy() if len(metrics_df) else pd.DataFrame()
    if len(rel_metrics) == 0:
        return pd.DataFrame()
    for _, m in rel_metrics.iterrows():
        mask = (
            (pred_case_df["dataset_key"].astype(str) == str(m["dataset_key"]))
            & (pred_case_df["representation"].astype(str) == str(m["representation"]))
            & (pred_case_df["model_key"].astype(str) == str(m["model_key"]))
            & (pred_case_df["target_name"].astype(str) == TARGET_NAME_RELAPSE_STATUS)
        )
        sub = pred_case_df.loc[mask].copy()
        if len(sub) < 4 or sub["y_true"].nunique() < 2:
            continue
        y = sub["y_true"].astype(int).values
        prob = sub["y_prob"].astype(float).values
        obs_auroc = roc_auc_score(y, prob)
        obs_auprc = average_precision_score(y, prob)
        null_auroc = []
        null_auprc = []
        for _ in range(n_perm):
            yp = rng.permutation(y)
            if len(np.unique(yp)) < 2:
                continue
            null_auroc.append(float(roc_auc_score(yp, prob)))
            null_auprc.append(float(average_precision_score(yp, prob)))
        rows.append({
            "dataset_key": m["dataset_key"],
            "representation": m["representation"],
            "model_key": m["model_key"],
            "n": len(sub),
            "n_permutations": len(null_auroc),
            "observed_auroc": obs_auroc,
            "observed_auprc": obs_auprc,
            "auroc_empirical_p": float((1 + np.sum(np.asarray(null_auroc) >= obs_auroc)) / (1 + len(null_auroc))) if null_auroc else np.nan,
            "auprc_empirical_p": float((1 + np.sum(np.asarray(null_auprc) >= obs_auprc)) / (1 + len(null_auprc))) if null_auprc else np.nan,
            "null_auroc_mean": float(np.mean(null_auroc)) if null_auroc else np.nan,
            "null_auprc_mean": float(np.mean(null_auprc)) if null_auprc else np.nan,
        })
    out = pd.DataFrame(rows)
    if len(out):
        out.to_csv(os.path.join(out_dir, "relapse_permutation_tests.csv"), index=False)
    return out


def generate_error_analysis(pred_case_df: pd.DataFrame, metrics_df: pd.DataFrame, raw_df: pd.DataFrame, out_dir: str) -> Tuple[pd.DataFrame, str]:
    rows: List[Dict[str, Any]] = []
    if pred_case_df is None or len(pred_case_df) == 0 or metrics_df is None or len(metrics_df) == 0:
        return pd.DataFrame(), ""
    raw_flags = pd.DataFrame()
    if raw_df is not None and len(raw_df):
        raw = _raw_df_with_row_index(raw_df)
        needed = {"case_id", "row_index", "preop_MRI_text", "path_report_text"}
        if needed.issubset(raw.columns):
            raw_flags = raw[["case_id", "row_index", "preop_MRI_text", "path_report_text"]].copy()
            raw_flags["mri_report_missing"] = raw_flags["preop_MRI_text"].apply(_is_missing_text).astype(int)
            raw_flags["path_report_missing"] = raw_flags["path_report_text"].apply(_is_missing_text).astype(int)
            raw_flags["mri_report_chars"] = raw_flags["preop_MRI_text"].fillna("").astype(str).str.len()
            raw_flags = raw_flags.drop(columns=["preop_MRI_text", "path_report_text"])

    best_reg = _best_row(metrics_df, "regression", TARGET_NAME_DISPERSION_SCORE)
    if best_reg is not None:
        sub = pred_case_df[
            (pred_case_df["dataset_key"].astype(str) == str(best_reg["dataset_key"]))
            & (pred_case_df["representation"].astype(str) == str(best_reg["representation"]))
            & (pred_case_df["model_key"].astype(str) == str(best_reg["model_key"]))
            & (pred_case_df["task_type"].astype(str) == "regression")
        ].copy()
        if len(sub):
            sub["residual"] = pd.to_numeric(sub["y_pred_value"], errors="coerce") - pd.to_numeric(sub["y_true"], errors="coerce")
            sd = float(sub["residual"].std(ddof=0) or 1.0)
            sub["abs_residual"] = sub["residual"].abs()
            sub["standardized_residual"] = sub["residual"] / max(sd, EPS)
            for _, r in sub.sort_values("abs_residual", ascending=False).head(20).iterrows():
                rows.append({**r.to_dict(), "error_task": "dispersion_regression", "error_type": "strong_overprediction" if r["residual"] > 0 else "strong_underprediction", "model_summary": f"{best_reg['dataset_key']}|{best_reg['representation']}|{best_reg['model_key']}"})

    for target_name in [TARGET_NAME_DISPERSION_HIGH_LOW, TARGET_NAME_RELAPSE_STATUS]:
        best_cls = _best_row(metrics_df, "classification", target_name)
        if best_cls is None:
            continue
        sub = pred_case_df[
            (pred_case_df["dataset_key"].astype(str) == str(best_cls["dataset_key"]))
            & (pred_case_df["representation"].astype(str) == str(best_cls["representation"]))
            & (pred_case_df["model_key"].astype(str) == str(best_cls["model_key"]))
            & (pred_case_df["target_name"].astype(str) == str(target_name))
        ].copy()
        if len(sub) == 0:
            continue
        sub["confidence"] = (pd.to_numeric(sub["y_prob"], errors="coerce") - 0.5).abs() * 2.0
        sub["correct"] = sub["y_true"].astype(int) == sub["y_pred"].astype(int)
        for _, r in sub[(sub["y_true"] == 0) & (sub["y_pred"] == 1)].sort_values("confidence", ascending=False).head(10).iterrows():
            rows.append({**r.to_dict(), "error_task": target_name, "error_type": "false_positive", "model_summary": f"{best_cls['dataset_key']}|{best_cls['representation']}|{best_cls['model_key']}"})
        for _, r in sub[(sub["y_true"] == 1) & (sub["y_pred"] == 0)].sort_values("confidence", ascending=False).head(10).iterrows():
            rows.append({**r.to_dict(), "error_task": target_name, "error_type": "false_negative", "model_summary": f"{best_cls['dataset_key']}|{best_cls['representation']}|{best_cls['model_key']}"})
        for _, r in sub[sub["correct"]].sort_values("confidence", ascending=True).head(10).iterrows():
            rows.append({**r.to_dict(), "error_task": target_name, "error_type": "low_confidence_correct", "model_summary": f"{best_cls['dataset_key']}|{best_cls['representation']}|{best_cls['model_key']}"})
        for _, r in sub[~sub["correct"]].sort_values("confidence", ascending=False).head(10).iterrows():
            rows.append({**r.to_dict(), "error_task": target_name, "error_type": "high_confidence_incorrect", "model_summary": f"{best_cls['dataset_key']}|{best_cls['representation']}|{best_cls['model_key']}"})

    out = pd.DataFrame(rows)
    if len(out):
        if len(raw_flags):
            out = out.merge(raw_flags, on=["case_id", "row_index"], how="left")
        out["likely_failure_modes"] = out.apply(lambda r: ";".join([
            "missing_mri_report" if int(r.get("mri_report_missing", 0) or 0) == 1 else "",
            "sparse_mri_language" if int(r.get("mri_report_chars", 9999) or 9999) < 800 and str(r.get("dataset_key", "")).startswith("mri") else "",
            "near_dispersion_threshold" if abs(float(r.get("y_true", np.nan)) - DISPERSION_TRUE_HIGH_THRESHOLD) <= 10 and r.get("error_task") == "dispersion_regression" else "",
            "low_confidence_boundary_case" if str(r.get("error_type", "")).startswith("low_confidence") else "",
        ]).strip(";"), axis=1)
        out["likely_failure_modes"] = out["likely_failure_modes"].replace("", "requires_manual_review")
        out_path = os.path.join(out_dir, "missed_case_error_analysis.csv")
        out.to_csv(out_path, index=False)
    else:
        out_path = ""
    md_path = os.path.join(out_dir, "missed_case_error_analysis.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Missed-case and Error Analysis\n\n")
        f.write("Cases are highlighted from the top aggregate held-out models after one-prediction-per-case deduplication.\n\n")
        f.write(_df_to_markdown(out.head(50) if len(out) else out, max_rows=50))
        f.write("\n")
    return out, md_path


def generate_missed_case_html_report(
    error_df: pd.DataFrame,
    out_dir: str,
    metrics_df: Optional[pd.DataFrame] = None,
) -> str:
    """Build a readable HTML review page for poorly predicted cases."""
    html_path = os.path.join(out_dir, "missed_case_review.html")
    if error_df is None or len(error_df) == 0:
        body = html_paragraph(
            "No missed-case rows were available. Run the full pipeline or provide "
            "`nested_outer_predictions_case_deduplicated.csv` plus `--csv-path` for MRI availability flags."
        )
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(build_html_report(
                "Missed-Case Review",
                "Case-level error analysis for the top aggregate models on deduplicated held-out predictions.",
                [html_section("Status", [body])],
            ))
        return html_path

    display_cols = [
        c for c in [
            "error_task", "error_type", "case_id", "row_index", "split_id",
            "dataset_key", "representation", "model_key", "model_summary",
            "y_true", "y_pred", "y_pred_value", "y_prob", "residual", "abs_residual",
            "confidence", "mri_report_missing", "path_report_missing", "mri_report_chars",
            "likely_failure_modes",
        ]
        if c in error_df.columns
    ]
    preview = error_df[display_cols].copy() if display_cols else error_df.copy()

    sections: List[str] = []
    sections.append(html_section(
        "Overview",
        [html_paragraph(
            "This page lists cases with the largest regression residuals, classification false positives/negatives, "
            "and high-confidence errors from the best-performing aggregate models. Evidence columns reflect report "
            "availability only; feature-level evidence requires per-split extraction artifacts."
        )],
    ))

    if metrics_df is not None and len(metrics_df):
        pathway_rows = []
        for label, row in [
            ("Best MRI regression", _best_row(metrics_df, "regression", TARGET_NAME_DISPERSION_SCORE, "mri")),
            ("Best pathology regression", _best_row(metrics_df, "regression", TARGET_NAME_DISPERSION_SCORE, "path")),
            ("Best combined regression", _best_row(metrics_df, "regression", TARGET_NAME_DISPERSION_SCORE, "combined")),
            ("Best high/low classifier", _best_row(metrics_df, "classification", TARGET_NAME_DISPERSION_HIGH_LOW)),
            ("Best relapse classifier", _best_row(metrics_df, "classification", TARGET_NAME_RELAPSE_STATUS)),
        ]:
            if row is not None:
                pathway_rows.append({
                    "pathway": label,
                    "dataset_key": row.get("dataset_key"),
                    "model_key": row.get("model_key"),
                    "representation": row.get("representation"),
                })
        if pathway_rows:
            sections.append(html_section(
                "Reference models for error extraction",
                [df_to_html_table(pd.DataFrame(pathway_rows), max_rows=10)],
            ))

    for error_type, title in [
        ("strong_overprediction", "Largest over-predictions (regression)"),
        ("strong_underprediction", "Largest under-predictions (regression)"),
        ("false_positive", "False positives (predicted positive, true negative)"),
        ("false_negative", "False negatives (predicted negative, true positive)"),
        ("high_confidence_incorrect", "High-confidence incorrect classifications"),
    ]:
        sub = preview[preview["error_type"] == error_type] if "error_type" in preview.columns else pd.DataFrame()
        if len(sub):
            sections.append(html_section(title, [df_to_html_table(sub.head(15), max_rows=15)]))

    remaining = preview[~preview["error_type"].isin([
        "strong_overprediction", "strong_underprediction", "false_positive",
        "false_negative", "high_confidence_incorrect",
    ])] if "error_type" in preview.columns else pd.DataFrame()
    if len(remaining):
        sections.append(html_section("Other flagged cases", [df_to_html_table(remaining.head(20), max_rows=20)]))

    if "likely_failure_modes" in preview.columns:
        mode_counts = (
            preview["likely_failure_modes"].astype(str).str.split(";").explode().replace("", np.nan).dropna().value_counts().reset_index()
        )
        mode_counts.columns = ["failure_mode", "count"]
        if len(mode_counts):
            sections.append(html_section(
                "Common failure-mode tags",
                [
                    df_to_html_table(mode_counts, max_rows=20),
                    html_paragraph(
                        "Tags are heuristic summaries from available artifacts (missing MRI, sparse language, "
                        "threshold proximity). They are not causal explanations."
                    ),
                ],
            ))

    sections.append(html_section(
        "Diagnostics note",
        [html_paragraph(
            "Improvement suggestions should be grounded in split-level lexicon coverage, class balance, and "
            "modality availability. Compare MRI-only versus pathology-only errors before attributing failures "
            "to a single representation."
        )],
    ))

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(build_html_report(
            "Missed-Case Review",
            "Case-level error analysis for poorly predicted held-out cases.",
            sections,
        ))
    return html_path


def generate_interpretability_report(
    out_dir: str,
    coef_all: pd.DataFrame,
    sign_stability_df: pd.DataFrame,
    phrase_freq_all: pd.DataFrame,
    group_freq_all: pd.DataFrame,
    stable_phrase_summary: pd.DataFrame,
    stable_group_summary: pd.DataFrame,
    reliability_all: pd.DataFrame,
    weighted_lexicon_all: pd.DataFrame,
    hyperparams_df: Optional[pd.DataFrame] = None,
    ontology_df: Optional[pd.DataFrame] = None,
) -> Tuple[str, str]:
    md_path = os.path.join(out_dir, "interpretability_report.md")
    html_path = os.path.join(out_dir, "interpretability_report.html")
    report_dir = os.path.join(out_dir, "interpretability_plots")
    os.makedirs(report_dir, exist_ok=True)

    density_rows = []
    if len(phrase_freq_all):
        for mode, sub in phrase_freq_all.groupby("report_mode"):
            density_rows.append({"report_mode": mode, "n_candidate_phrases": sub["phrase_slug"].nunique() if "phrase_slug" in sub.columns else len(sub), "n_stable_phrases": int((sub.get("stable", pd.Series(dtype=int)) == 1).sum()) if "stable" in sub.columns else np.nan})
    if len(group_freq_all):
        for mode, sub in group_freq_all.groupby("report_mode"):
            match = next((r for r in density_rows if r["report_mode"] == mode), None)
            if match is None:
                match = {"report_mode": mode}; density_rows.append(match)
            match["n_candidate_groups"] = sub["canonical_group"].nunique() if "canonical_group" in sub.columns else len(sub)
            match["n_stable_groups"] = int((sub.get("stable", pd.Series(dtype=int)) == 1).sum()) if "stable" in sub.columns else np.nan
    density_df = pd.DataFrame(density_rows)
    if len(density_df):
        density_df.to_csv(os.path.join(out_dir, "feature_density_summary_by_modality.csv"), index=False)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        x = np.arange(len(density_df))
        width = 0.35
        if "n_candidate_phrases" in density_df.columns:
            ax.bar(x - width / 2, density_df["n_candidate_phrases"].fillna(0), width, label="candidate phrases")
        if "n_stable_phrases" in density_df.columns:
            ax.bar(x + width / 2, density_df["n_stable_phrases"].fillna(0), width, label="stable phrases")
        ax.set_xticks(x)
        ax.set_xticklabels(density_df["report_mode"].astype(str))
        ax.set_ylabel("Feature count")
        ax.set_title("Feature counts by modality")
        ax.legend(fontsize=8)
        _save_figure(os.path.join(report_dir, "feature_count_by_modality.png"), fig)

    if len(phrase_freq_all) and "selection_frequency" in phrase_freq_all.columns:
        prev = phrase_freq_all.copy()
        if "report_mode" in prev.columns:
            prev = prev.groupby("report_mode")["selection_frequency"].mean().reset_index()
            prev.columns = ["report_mode", "mean_selection_frequency"]
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.bar(prev["report_mode"].astype(str), prev["mean_selection_frequency"].astype(float))
            ax.set_ylabel("Mean rediscovery frequency")
            ax.set_title("Feature stability prevalence by modality")
            _save_figure(os.path.join(report_dir, "feature_prevalence_by_modality.png"), fig)

    top_coef = pd.DataFrame()
    if len(coef_all):
        top_coef = coef_all.copy()
        top_coef = top_coef.sort_values("abs_coef", ascending=False).head(200)
        top_coef.to_csv(os.path.join(out_dir, "top_model_coefficients_interpretability.csv"), index=False)
        top_reg = top_coef[top_coef["task_type"].astype(str) == "regression"].head(25) if "task_type" in top_coef.columns else pd.DataFrame()
        if len(top_reg) and "feature" in top_reg.columns:
            fig_h = max(5.0, min(0.35 * len(top_reg) + 2.0, 16.0))
            fig, ax = plt.subplots(figsize=(10, fig_h))
            colors = np.where(top_reg["coef"].astype(float) >= 0, "#4C78A8", "#E45756")
            ax.barh(top_reg["feature"].astype(str), top_reg["coef"].astype(float), color=colors)
            ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
            ax.invert_yaxis()
            ax.set_xlabel("Coefficient")
            ax.set_title("Top regression coefficients")
            _save_figure(os.path.join(report_dir, "top_regression_coefficients.png"), fig)

    if len(sign_stability_df):
        top_stab = sign_stability_df.sort_values(["n_nonzero", "sign_consistency", "mean_abs_coef"], ascending=[False, False, False]).head(40)
        fig_h = max(5.0, min(0.35 * len(top_stab) + 2.0, 18.0))
        fig, ax = plt.subplots(figsize=(10, fig_h))
        ax.barh(top_stab["feature"].astype(str), top_stab["sign_consistency"].astype(float))
        ax.set_xlabel("Fold-level sign consistency")
        ax.set_ylabel("Feature")
        ax.set_title("Coefficient sign stability across folds")
        ax.invert_yaxis()
        _save_figure(os.path.join(report_dir, "coefficient_sign_stability.png"), fig)

    if len(weighted_lexicon_all):
        weight_col = "final_weight" if "final_weight" in weighted_lexicon_all.columns else (
            "weight_normalized" if "weight_normalized" in weighted_lexicon_all.columns else (
                "weight" if "weight" in weighted_lexicon_all.columns else None
            )
        )
        concept_col = "mri_concept" if "mri_concept" in weighted_lexicon_all.columns else weighted_lexicon_all.columns[0]
        if weight_col is not None:
            top_w = weighted_lexicon_all.groupby(concept_col, as_index=False)[weight_col].mean()
            top_w = top_w.sort_values(weight_col, ascending=False).head(30)
            fig_w, fig_h = _metric_bar_figure_size(top_w[concept_col].astype(str).tolist(), horizontal=True)
            fig, ax = plt.subplots(figsize=(fig_w, fig_h))
            ypos = np.arange(len(top_w))
            ax.barh(ypos, top_w[weight_col].astype(float))
            ax.set_yticks(ypos)
            ax.set_yticklabels(top_w[concept_col].astype(str), fontsize=8)
            ax.invert_yaxis()
            ax.set_xlabel("Calibration-derived MRI concept weight")
            ax.set_title("Top pathology-calibrated MRI concept weights")
            _save_figure(os.path.join(report_dir, "weighted_mri_concepts.png"), fig)

    if len(reliability_all):
        val_col = "delta_p_path_given_mri" if "delta_p_path_given_mri" in reliability_all.columns else ("lift" if "lift" in reliability_all.columns else None)
        if val_col and {"mri_concept", "path_concept"}.issubset(reliability_all.columns):
            piv = reliability_all.pivot_table(index="mri_concept", columns="path_concept", values=val_col, aggfunc="mean")
            if len(piv):
                fig_w = max(10.0, min(0.45 * len(piv.columns) + 4.0, 20.0))
                fig_h = max(8.0, min(0.45 * len(piv.index) + 3.0, 16.0))
                fig, ax = plt.subplots(figsize=(fig_w, fig_h))
                im = ax.imshow(piv.fillna(0).values, aspect="auto")
                ax.set_xticks(range(len(piv.columns)))
                ax.set_xticklabels(piv.columns, rotation=75, ha="right", fontsize=8)
                ax.set_yticks(range(len(piv.index)))
                ax.set_yticklabels(piv.index, fontsize=8)
                ax.set_title("MRI-pathology concept reliability matrix")
                fig.colorbar(im, ax=ax, label=val_col, fraction=0.046, pad=0.04)
                _save_figure(os.path.join(report_dir, "mri_pathology_reliability_heatmap.png"), fig)

    if len(coef_all):
        comp = coef_all.copy()
        comp = comp.groupby(["feature", "target_name"]).agg(mean_abs_coef=("abs_coef", "mean")).reset_index()
        piv = comp.pivot_table(index="feature", columns="target_name", values="mean_abs_coef", aggfunc="mean").fillna(0)
        if TARGET_NAME_RELAPSE_STATUS in piv.columns and TARGET_NAME_DISPERSION_SCORE in piv.columns:
            piv["relapse_minus_dispersion_abscoef"] = piv[TARGET_NAME_RELAPSE_STATUS] - piv[TARGET_NAME_DISPERSION_SCORE]
            piv.sort_values("relapse_minus_dispersion_abscoef", ascending=False).to_csv(os.path.join(out_dir, "feature_association_dispersion_vs_relapse.csv"))

    md = []
    md.append("# Interpretability Report\n")
    md.append("This report summarizes extracted lexical feature density, stable lexicons, fitted model coefficients, coefficient stability, pathology-MRI reliability, and pathology-calibrated MRI weights. MRI and pathology are not forced to have equal feature counts; any matched-budget comparison should be interpreted as a sensitivity analysis, not the primary extraction.\n")
    md.append("## Feature-density summary\n")
    md.append(_df_to_markdown(density_df, max_rows=50))
    md.append("\n## Stable phrase summary\n")
    md.append(_df_to_markdown(stable_phrase_summary, max_rows=25))
    md.append("\n## Stable group summary\n")
    md.append(_df_to_markdown(stable_group_summary, max_rows=25))
    md.append("\n## Top coefficients\n")
    cols = [c for c in ["dataset_key", "representation", "model_key", "target_name", "feature", "coef", "abs_coef", "coef_sign", "feature_prevalence_train", "feature_modality", "ontology_concept"] if c in top_coef.columns]
    md.append(_df_to_markdown(top_coef[cols] if len(top_coef) and cols else top_coef, max_rows=50))
    md.append("\n## Sign stability\n")
    md.append(_df_to_markdown(sign_stability_df.head(50) if len(sign_stability_df) else sign_stability_df, max_rows=50))
    md.append("\n## Pathology-calibrated MRI weights\n")
    md.append(_df_to_markdown(weighted_lexicon_all.head(50) if len(weighted_lexicon_all) else weighted_lexicon_all, max_rows=50))
    if hyperparams_df is not None and len(hyperparams_df):
        md.append("\n## Selected hyperparameters\n")
        md.append(_df_to_markdown(hyperparams_df.head(50), max_rows=50))
    if ontology_df is not None and len(ontology_df):
        md.append("\n## Shared ontology concepts\n")
        md.append(_df_to_markdown(ontology_df.head(30), max_rows=30))
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    interp_plot_map = {
        "coefficient_sign_stability.png": os.path.join(report_dir, "coefficient_sign_stability.png"),
        "mri_pathology_reliability_heatmap.png": os.path.join(report_dir, "mri_pathology_reliability_heatmap.png"),
        "weighted_mri_concepts.png": os.path.join(report_dir, "weighted_mri_concepts.png"),
        "feature_count_by_modality.png": os.path.join(report_dir, "feature_count_by_modality.png"),
        "feature_prevalence_by_modality.png": os.path.join(report_dir, "feature_prevalence_by_modality.png"),
        "top_regression_coefficients.png": os.path.join(report_dir, "top_regression_coefficients.png"),
    }

    def _interp_plot_html(filename: str, title: str) -> str:
        plot_path = interp_plot_map.get(filename, "")
        if plot_path and os.path.exists(plot_path):
            return html_plot_block(plot_path, f"interpretability_plots/{filename}", title=title)
        return ""

    html_sections = [
        html_section("Feature-density summary", [df_to_html_table(density_df, max_rows=50)]),
        html_section("Stable phrase summary", [df_to_html_table(stable_phrase_summary, max_rows=25)]),
        html_section("Stable group summary", [df_to_html_table(stable_group_summary, max_rows=25)]),
        html_section(
            "Top coefficients",
            [df_to_html_table(top_coef[cols] if len(top_coef) and cols else top_coef, max_rows=50)],
        ),
        html_section(
            "Sign stability",
            [
                df_to_html_table(sign_stability_df.head(50) if len(sign_stability_df) else sign_stability_df, max_rows=50),
                _interp_plot_html("coefficient_sign_stability.png", "Coefficient sign stability"),
            ],
        ),
        html_section(
            "MRI-pathology reliability",
            [
                _interp_plot_html("mri_pathology_reliability_heatmap.png", "MRI-pathology concept reliability")
                or html_paragraph("No reliability heatmap was generated for this run."),
            ],
        ),
        html_section(
            "Pathology-calibrated MRI weights",
            [
                df_to_html_table(weighted_lexicon_all.head(50) if len(weighted_lexicon_all) else weighted_lexicon_all, max_rows=50),
                _interp_plot_html("weighted_mri_concepts.png", "Top pathology-calibrated MRI concept weights"),
            ],
        ),
        html_section(
            "Feature engineering summary",
            [
                _interp_plot_html("feature_count_by_modality.png", "Feature counts by modality"),
                _interp_plot_html("feature_prevalence_by_modality.png", "Feature stability prevalence"),
                _interp_plot_html("top_regression_coefficients.png", "Top regression coefficients"),
            ],
        ),
    ]
    if hyperparams_df is not None and len(hyperparams_df):
        html_sections.append(html_section("Selected hyperparameters", [df_to_html_table(hyperparams_df.head(50), max_rows=50)]))
    if ontology_df is not None and len(ontology_df):
        html_sections.append(html_section("Shared ontology concepts", [df_to_html_table(ontology_df.head(30), max_rows=30)]))
    intro = (
        "This report summarizes extracted lexical feature density, stable lexicons, fitted model coefficients, "
        "coefficient stability, pathology-MRI reliability, and pathology-calibrated MRI weights. MRI and pathology "
        "are not forced to have equal feature counts; any matched-budget comparison should be interpreted as a "
        "sensitivity analysis, not the primary extraction."
    )
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(build_html_report("Interpretability Report", intro, html_sections))
    return md_path, html_path


def generate_results_report(
    out_dir: str,
    metrics_df: pd.DataFrame,
    pred_case_df: pd.DataFrame,
    fold_results_all: pd.DataFrame,
    relapse_balance_df: pd.DataFrame,
    permutation_df: pd.DataFrame,
    plot_paths: Sequence[str],
    path_mri_subset_metrics_df: pd.DataFrame,
    run_metadata: Optional[Dict[str, Any]] = None,
) -> Tuple[str, str]:
    md_path = os.path.join(out_dir, "automated_results_report.md")
    html_path = os.path.join(out_dir, "automated_results_report.html")
    run_metadata = run_metadata or _load_run_metadata(out_dir)
    label_dist_df = _summarize_label_distributions(pred_case_df)
    split_summary_df = run_metadata.get("split_summary_df", pd.DataFrame())
    mri_missing_df = run_metadata.get("mri_missing_df", pd.DataFrame())
    lines = []
    lines.append("# Automated Model Performance Report\n")
    lines.append("All aggregate metrics in this report are computed from held-out outer-test predictions after case-level deduplication, so each case contributes at most one prediction per dataset / representation / model / target. Raw per-split predictions are saved separately.\n")

    if run_metadata.get("nested_summary_text"):
        lines.append("## Run overview\n")
        lines.append("```text\n" + run_metadata["nested_summary_text"] + "\n```\n")
    cost_post = run_metadata.get("cost_post_run") or {}
    cost_apriori = run_metadata.get("cost_apriori") or {}
    cost_md_lines, cost_html_parts = _build_cost_comparison_report_parts(out_dir, cost_apriori, cost_post)
    if cost_md_lines:
        lines.extend(cost_md_lines)
    elif cost_post:
        actual = extract_actual_cumulative(cost_post)
        lines.append("## Token cost summary\n")
        lines.append(_df_to_markdown(pd.DataFrame([{
            "api_calls": actual.get("calls"),
            "prompt_tokens": actual.get("prompt_tokens"),
            "completion_tokens": actual.get("completion_tokens"),
            "total_tokens": actual.get("total_tokens"),
            "estimated_cost_usd": actual.get("estimated_cost_usd"),
            "cost_type": cost_post.get("cost_type"),
        }]), max_rows=5))

    if len(split_summary_df):
        lines.append("\n## Outer-fold split summary\n")
        lines.append(_df_to_markdown(split_summary_df, max_rows=20))
    if len(mri_missing_df):
        lines.append("\n## Missing MRI handling\n")
        lines.append(_df_to_markdown(mri_missing_df, max_rows=20))
    if len(label_dist_df):
        lines.append("\n## Label distributions (deduplicated held-out predictions)\n")
        lines.append(_df_to_markdown(label_dist_df, max_rows=20))
    lines.append("## Top model summary\n")
    best_specs = [
        ("Top MRI-only regression", _best_row(metrics_df, "regression", TARGET_NAME_DISPERSION_SCORE, "mri")),
        ("Top pathology-only regression", _best_row(metrics_df, "regression", TARGET_NAME_DISPERSION_SCORE, "path")),
        ("Top combined regression", _best_row(metrics_df, "regression", TARGET_NAME_DISPERSION_SCORE, "combined")),
        ("Top overall regression", _best_row(metrics_df, "regression", TARGET_NAME_DISPERSION_SCORE)),
        ("Top high/low classification", _best_row(metrics_df, "classification", TARGET_NAME_DISPERSION_HIGH_LOW)),
        ("Top relapse classification", _best_row(metrics_df, "classification", TARGET_NAME_RELAPSE_STATUS)),
    ]
    summary_rows = []
    for label, row in best_specs:
        if row is None:
            continue
        summary_rows.append({"selection": label, "dataset_key": row.get("dataset_key"), "representation": row.get("representation"), "model_key": row.get("model_key"), "target_name": row.get("target_name"), "mae": row.get("mae", np.nan), "spearman_rho": row.get("spearman_rho", np.nan), "auroc": row.get("auroc", np.nan), "auprc": row.get("auprc", np.nan), "f1": row.get("f1", np.nan), "brier": row.get("brier", np.nan)})
    lines.append(_df_to_markdown(pd.DataFrame(summary_rows), max_rows=20))

    lines.append("\n## Aggregate held-out metrics\n")
    preferred_cols = [c for c in ["target_name", "dataset_key", "representation", "model_key", "n", "mae", "mae_ci_low", "mae_ci_high", "rmse", "pearson_r", "spearman_rho", "r2", "accuracy", "balanced_accuracy", "f1", "auroc", "auprc", "auprc_no_skill_baseline", "brier", "precision_ppv", "recall_sensitivity", "specificity", "npv", "tn", "fp", "fn", "tp", "prevalence"] if c in metrics_df.columns]
    lines.append(_df_to_markdown(metrics_df[preferred_cols].sort_values(["target_name", "dataset_key"]) if len(metrics_df) and preferred_cols else metrics_df, max_rows=100))

    lines.append("\n## Relapse imbalance diagnostics\n")
    lines.append(_df_to_markdown(relapse_balance_df, max_rows=50))
    lines.append("\nAUPRC no-skill baselines are included in the aggregate metrics table and equal the event prevalence for the evaluated prediction set.\n")

    if len(permutation_df):
        lines.append("\n## Relapse permutation tests\n")
        lines.append(_df_to_markdown(permutation_df.sort_values("auprc_empirical_p"), max_rows=50))

    if len(path_mri_subset_metrics_df):
        lines.append("\n## Pathology-only full-cohort vs MRI-complete sensitivity\n")
        lines.append("Pathology-only models can use all target-eligible cases, while MRI-derived and combined models exclude MRI-missing cases. The table below recomputes pathology-only aggregate metrics on MRI-complete cases for comparison.\n")
        lines.append(_df_to_markdown(path_mri_subset_metrics_df, max_rows=50))

    if len(fold_results_all):
        fold_cols = [c for c in ["split_id", "target_name", "dataset_key", "representation", "model_key", "mae", "spearman_rho", "auroc", "auprc", "f1", "brier", "n"] if c in fold_results_all.columns]
        lines.append("\n## Per-fold performance (sample)\n")
        lines.append(_df_to_markdown(fold_results_all[fold_cols].head(80) if fold_cols else fold_results_all.head(80), max_rows=80))

    lines.append("\n## Generated plots\n")
    for path in plot_paths:
        rel = os.path.relpath(path, out_dir)
        lines.append(f"- `{rel}`")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    summary_df = pd.DataFrame(summary_rows)
    metrics_table_df = metrics_df[preferred_cols].sort_values(["target_name", "dataset_key"]) if len(metrics_df) and preferred_cols else metrics_df

    plot_groups: List[Tuple[str, List[str]]] = [
        ("Regression performance", [
            "top_regression_predicted_vs_true.png",
            "top_regression_residuals.png",
            "top_regression_spearman_rank.png",
            "nested_regression_error_comparison.png",
            "nested_regression_correlation_comparison.png",
            "ranked_model_mae.png",
            "per_fold_regression_mae.png",
        ]),
        ("Dispersion high/low classification", [
            "top_dispersion_high_low_confusion_matrix.png",
            "top_dispersion_high_low_roc.png",
            "top_dispersion_high_low_pr.png",
            "top_dispersion_high_low_calibration.png",
            "nested_classification_comparison.png",
            "ranked_model_auroc.png",
        ]),
        ("Relapse classification", [
            "top_relapse_status_confusion_matrix.png",
            "top_relapse_status_roc.png",
            "top_relapse_status_pr.png",
            "top_relapse_status_calibration.png",
            "nested_relapse_auroc_comparison.png",
            "nested_relapse_auprc_comparison.png",
            "nested_relapse_f1_comparison.png",
            "nested_relapse_brier_comparison.png",
            "nested_relapse_roc_curves_top_models.png",
            "nested_relapse_pr_curves_top_models.png",
            "ranked_model_auprc.png",
        ]),
        ("Uncertainty and model ranking", [
            "bootstrap_ci_primary_metrics.png",
            "pathway_modality_comparison.png",
        ]),
    ]
    plot_path_by_name = {os.path.basename(p): p for p in plot_paths if os.path.exists(p)}
    plotted = set()
    plot_sections: List[str] = []
    for group_title, filenames in plot_groups:
        blocks = []
        for filename in filenames:
            plot_path = plot_path_by_name.get(filename)
            if plot_path is None:
                continue
            plotted.add(filename)
            rel = os.path.relpath(plot_path, out_dir)
            blocks.append(html_plot_block(plot_path, rel))
        if blocks:
            plot_sections.append(html_section(group_title, blocks))
    remaining = [
        html_plot_block(path, os.path.relpath(path, out_dir))
        for name, path in sorted(plot_path_by_name.items())
        if name not in plotted
    ]
    if remaining:
        plot_sections.append(html_section("Additional figures", remaining))

    html_sections = []
    if run_metadata.get("nested_summary_text"):
        html_sections.append(html_section("Run overview", [html_paragraph(run_metadata["nested_summary_text"])]))
    if cost_html_parts:
        html_sections.append(html_section("Cost estimate vs actual", cost_html_parts))
    elif cost_post:
        actual = extract_actual_cumulative(cost_post)
        html_sections.append(html_section("Token cost summary", [
            df_to_html_table(pd.DataFrame([{
                "api_calls": actual.get("calls"),
                "prompt_tokens": actual.get("prompt_tokens"),
                "cached_tokens": actual.get("cached_tokens"),
                "completion_tokens": actual.get("completion_tokens"),
                "total_tokens": actual.get("total_tokens"),
                "estimated_cost_usd": actual.get("estimated_cost_usd"),
                "cost_type": cost_post.get("cost_type"),
            }]), max_rows=5),
        ]))
    if len(split_summary_df):
        html_sections.append(html_section("Outer-fold split summary", [df_to_html_table(split_summary_df, max_rows=20)]))
    if len(mri_missing_df):
        html_sections.append(html_section("Missing MRI handling", [
            df_to_html_table(mri_missing_df, max_rows=20),
            html_paragraph("MRI-missing cases are excluded from MRI-only, combined, calibrated-MRI, and teacher-student pathways."),
        ]))
    if len(label_dist_df):
        html_sections.append(html_section("Label distributions", [df_to_html_table(label_dist_df, max_rows=20)]))

    html_sections.extend([
        html_section("Top model summary", [df_to_html_table(summary_df, max_rows=20)]),
        html_section("Aggregate held-out metrics", [df_to_html_table(metrics_table_df, max_rows=100)]),
        html_section("Relapse imbalance diagnostics", [
            df_to_html_table(relapse_balance_df, max_rows=50),
            html_paragraph(
                "AUPRC no-skill baselines are included in the aggregate metrics table and equal the event prevalence for the evaluated prediction set."
            ),
        ]),
    ])
    if len(permutation_df):
        html_sections.append(
            html_section(
                "Relapse permutation tests",
                [df_to_html_table(permutation_df.sort_values("auprc_empirical_p"), max_rows=50)],
            )
        )
    if len(path_mri_subset_metrics_df):
        html_sections.append(
            html_section(
                "Pathology-only full-cohort vs MRI-complete sensitivity",
                [
                    html_paragraph(
                        "Pathology-only models can use all target-eligible cases, while MRI-derived and combined models exclude MRI-missing cases. The table below recomputes pathology-only aggregate metrics on MRI-complete cases for comparison."
                    ),
                    df_to_html_table(path_mri_subset_metrics_df, max_rows=50),
                ],
            )
        )
    if len(fold_results_all):
        fold_cols = [c for c in ["split_id", "target_name", "dataset_key", "representation", "model_key", "mae", "spearman_rho", "auroc", "auprc", "f1", "brier", "n"] if c in fold_results_all.columns]
        html_sections.append(html_section(
            "Per-fold performance",
            [df_to_html_table(fold_results_all[fold_cols].head(80) if fold_cols else fold_results_all.head(80), max_rows=80)],
        ))
    html_sections.extend(plot_sections)
    intro = (
        "All aggregate metrics in this report are computed from held-out outer-test predictions after case-level "
        "deduplication, so each case contributes at most one prediction per dataset / representation / model / target. "
        "Raw per-split predictions are saved separately."
    )
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(build_html_report("Automated Model Performance Report", intro, html_sections))
    return md_path, html_path


def generate_all_reports(
    out_dir: str,
    *,
    csv_path: Optional[str] = None,
    force: bool = False,
) -> Dict[str, str]:
    """Generate plots and all three HTML review reports from saved run artifacts."""
    metrics_path = os.path.join(out_dir, "nested_outer_metrics_summary.csv")
    pred_path = os.path.join(out_dir, "nested_outer_predictions_case_deduplicated.csv")
    if not os.path.exists(metrics_path) or not os.path.exists(pred_path):
        raise FileNotFoundError(
            f"Expected {metrics_path} and {pred_path} for report generation."
        )

    results_html = os.path.join(out_dir, "automated_results_report.html")
    interp_html = os.path.join(out_dir, "interpretability_report.html")
    missed_html = os.path.join(out_dir, "missed_case_review.html")
    if not force and all(os.path.exists(p) for p in [results_html, interp_html, missed_html]):
        print("[SKIP] HTML reports already exist; pass force=True to regenerate.")

    metrics_df = pd.read_csv(metrics_path)
    pred_case_all = pd.read_csv(pred_path)
    coef_all = pd.read_csv(os.path.join(out_dir, "nested_outer_feature_coefficients_all.csv")) if os.path.exists(os.path.join(out_dir, "nested_outer_feature_coefficients_all.csv")) else pd.DataFrame()
    sign_stability_df = pd.read_csv(os.path.join(out_dir, "nested_feature_sign_stability.csv")) if os.path.exists(os.path.join(out_dir, "nested_feature_sign_stability.csv")) else pd.DataFrame()
    phrase_freq_all = pd.read_csv(os.path.join(out_dir, "all_outer_phrase_rediscovery_frequencies.csv")) if os.path.exists(os.path.join(out_dir, "all_outer_phrase_rediscovery_frequencies.csv")) else pd.DataFrame()
    group_freq_all = pd.read_csv(os.path.join(out_dir, "all_outer_group_rediscovery_frequencies.csv")) if os.path.exists(os.path.join(out_dir, "all_outer_group_rediscovery_frequencies.csv")) else pd.DataFrame()
    stable_phrase_summary = pd.read_csv(os.path.join(out_dir, "stable_phrase_lexicon_outer_summary.csv")) if os.path.exists(os.path.join(out_dir, "stable_phrase_lexicon_outer_summary.csv")) else pd.DataFrame()
    stable_group_summary = pd.read_csv(os.path.join(out_dir, "stable_group_lexicon_outer_summary.csv")) if os.path.exists(os.path.join(out_dir, "stable_group_lexicon_outer_summary.csv")) else pd.DataFrame()
    reliability_all = pd.read_csv(os.path.join(out_dir, "all_outer_mri_pathology_reliability_matrices.csv")) if os.path.exists(os.path.join(out_dir, "all_outer_mri_pathology_reliability_matrices.csv")) else pd.DataFrame()
    weighted_lexicon_all = pd.read_csv(os.path.join(out_dir, "all_outer_weighted_mri_lexicons.csv")) if os.path.exists(os.path.join(out_dir, "all_outer_weighted_mri_lexicons.csv")) else pd.DataFrame()
    hyperparams_df = pd.read_csv(os.path.join(out_dir, "nested_outer_hyperparameters_all.csv")) if os.path.exists(os.path.join(out_dir, "nested_outer_hyperparameters_all.csv")) else pd.DataFrame()
    ontology_df = pd.read_csv(os.path.join(out_dir, "shared_biological_concept_ontology.csv")) if os.path.exists(os.path.join(out_dir, "shared_biological_concept_ontology.csv")) else pd.DataFrame()
    relapse_balance_df = pd.read_csv(os.path.join(out_dir, "relapse_class_balance_by_split.csv")) if os.path.exists(os.path.join(out_dir, "relapse_class_balance_by_split.csv")) else pd.DataFrame()
    permutation_df = pd.read_csv(os.path.join(out_dir, "relapse_permutation_tests.csv")) if os.path.exists(os.path.join(out_dir, "relapse_permutation_tests.csv")) else pd.DataFrame()
    path_mri_subset_metrics_df = pd.read_csv(os.path.join(out_dir, "pathology_only_mri_complete_subset_metrics.csv")) if os.path.exists(os.path.join(out_dir, "pathology_only_mri_complete_subset_metrics.csv")) else pd.DataFrame()
    fold_results_all = pd.read_csv(os.path.join(out_dir, "nested_outer_fold_metrics_all.csv")) if os.path.exists(os.path.join(out_dir, "nested_outer_fold_metrics_all.csv")) else pd.DataFrame()
    run_metadata = _load_run_metadata(out_dir)

    interp_md, interp_html_path = generate_interpretability_report(
        out_dir=out_dir,
        coef_all=coef_all,
        sign_stability_df=sign_stability_df,
        phrase_freq_all=phrase_freq_all,
        group_freq_all=group_freq_all,
        stable_phrase_summary=stable_phrase_summary,
        stable_group_summary=stable_group_summary,
        reliability_all=reliability_all,
        weighted_lexicon_all=weighted_lexicon_all,
        hyperparams_df=hyperparams_df,
        ontology_df=ontology_df,
    )
    print(f"[SAVE] Wrote interpretability reports: {interp_md}, {interp_html_path}")

    cls_plot_png = os.path.join(out_dir, "nested_classification_comparison.png")
    relapse_auroc_plot_png = os.path.join(out_dir, "nested_relapse_auroc_comparison.png")
    relapse_auprc_plot_png = os.path.join(out_dir, "nested_relapse_auprc_comparison.png")
    relapse_f1_plot_png = os.path.join(out_dir, "nested_relapse_f1_comparison.png")
    relapse_brier_plot_png = os.path.join(out_dir, "nested_relapse_brier_comparison.png")
    reg_err_plot_png = os.path.join(out_dir, "nested_regression_error_comparison.png")
    reg_corr_plot_png = os.path.join(out_dir, "nested_regression_correlation_comparison.png")
    plot_classification_comparison(metrics_df, cls_plot_png)
    plot_relapse_metric_comparison(metrics_df, relapse_auroc_plot_png, "auroc")
    plot_relapse_metric_comparison(metrics_df, relapse_auprc_plot_png, "auprc")
    plot_relapse_metric_comparison(metrics_df, relapse_f1_plot_png, "f1")
    plot_relapse_metric_comparison(metrics_df, relapse_brier_plot_png, "brier")
    plot_relapse_curves(pred_case_all, metrics_df, out_dir)
    plot_regression_error_comparison(metrics_df, reg_err_plot_png)
    plot_regression_correlation_comparison(metrics_df, reg_corr_plot_png)
    report_plot_paths = generate_performance_plots(pred_case_all, metrics_df, out_dir)
    per_fold_png = os.path.join(out_dir, "report_plots", "per_fold_regression_mae.png")
    if _plot_per_fold_regression_mae(fold_results_all, metrics_df, per_fold_png):
        report_plot_paths.append(per_fold_png)
    for pth in [
        cls_plot_png,
        relapse_auroc_plot_png,
        relapse_auprc_plot_png,
        relapse_f1_plot_png,
        relapse_brier_plot_png,
        reg_err_plot_png,
        reg_corr_plot_png,
        os.path.join(out_dir, "nested_relapse_roc_curves_top_models.png"),
        os.path.join(out_dir, "nested_relapse_pr_curves_top_models.png"),
    ]:
        if os.path.exists(pth):
            report_plot_paths.append(pth)

    results_md, results_html_path = generate_results_report(
        out_dir=out_dir,
        metrics_df=metrics_df,
        pred_case_df=pred_case_all,
        fold_results_all=fold_results_all,
        relapse_balance_df=relapse_balance_df,
        permutation_df=permutation_df,
        plot_paths=report_plot_paths,
        path_mri_subset_metrics_df=path_mri_subset_metrics_df,
        run_metadata=run_metadata,
    )
    print(f"[SAVE] Wrote automated results reports: {results_md}, {results_html_path}")

    raw_df = pd.DataFrame()
    if csv_path and os.path.isfile(csv_path):
        raw_df = load_cases(csv_path)
    error_df, error_md = generate_error_analysis(pred_case_all, metrics_df, raw_df, out_dir)
    if error_md:
        print(f"[SAVE] Wrote missed-case markdown/csv: {error_md}")
    missed_html_path = generate_missed_case_html_report(error_df, out_dir, metrics_df)
    print(f"[SAVE] Wrote missed-case HTML review: {missed_html_path}")

    return {
        "results_html": results_html_path,
        "interpretability_html": interp_html_path,
        "missed_case_html": missed_html_path,
        "results_md": results_md,
        "interpretability_md": interp_md,
        "missed_case_md": error_md,
    }


def regenerate_reports_and_plots(out_dir: str, args: argparse.Namespace) -> None:
    """Rebuild plots and HTML reports from saved nested-evaluation artifacts."""
    csv_path = getattr(args, "csv_path", None)
    generate_all_reports(out_dir, csv_path=csv_path, force=True)


def pathology_metrics_on_mri_complete(pred_case_df: pd.DataFrame, raw_df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if pred_case_df is None or len(pred_case_df) == 0:
        return pd.DataFrame()
    missing_rows = _mri_missing_row_indices(raw_df)
    df = pred_case_df[(pred_case_df["dataset_key"].astype(str) == "path") & (~pred_case_df["row_index"].astype(int).isin(missing_rows))].copy()
    if len(df) == 0:
        return pd.DataFrame()
    out = compute_metrics_from_predictions(df, args, seed_offset=404)
    if len(out):
        out["comparison_subset"] = "pathology_only_on_mri_complete_cases"
    return out
