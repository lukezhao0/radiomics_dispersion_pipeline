"""Per-split and fold-level nested evaluation orchestration."""

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

from approach2.audit import compute_mri_audit_table, summarize_audit_by_groups
from approach2.calibration import (
    build_weighted_mri_concept_score_matrix,
    compute_cross_modal_reliability,
    compute_weighted_mri_lexicon,
    randomized_or_mismatched_path_matrix,
)
from approach2.checkpoint import (
    build_split_resume_fingerprint,
    load_split_manifest,
    manifest_matches_split_membership,
    validate_completed_checkpoint_tables,
    validate_split_marker,
)
from approach2.eval_data import (
    _mri_missing_row_indices,
    filter_missing_mri_for_dataset,
    has_usable_mri_report,
)
from approach2.features.matrices import (
    build_group_feature_matrix,
    build_phrase_feature_matrix,
    get_representation_matrix,
    merge_modalities_early_fusion,
)
from approach2.features.normalize import explode_phrase_rows, load_extractions_csv, ontology_table
from approach2.lexicon import build_stable_lexicon_from_training_extractions
from approach2.evaluation.plots import rank_features_across_models, summarize_coefficient_sign_stability
from approach2.models_ml import (
    add_bootstrap_metric_cis,
    annotate_coefficient_table,
    classification_metrics,
    extract_fitted_feature_coefficients,
    fit_one_outer_model,
    fit_teacher_student_mri_model,
    get_model_specs,
    model_target_specs_for_model,
    prepare_task_frames,
    regression_metrics,
    should_skip_model_fit,
)
from approach2.recoding import recode_cases_with_frozen_lexicon
from approach2.splits import case_id_list_hash

# -----------------------------
# One modality / one outer split
# -----------------------------

def _split_resume_dir(split_dir: str) -> str:
    path = os.path.join(split_dir, "_split_resume_checkpoint")
    os.makedirs(path, exist_ok=True)
    return path


def _split_resume_marker(split_dir: str) -> str:
    return os.path.join(_split_resume_dir(split_dir), "COMPLETED.json")


def _split_resume_paths(split_dir: str) -> Dict[str, str]:
    root = _split_resume_dir(split_dir)
    return {
        "phrase_freq_df": os.path.join(root, "phrase_freq.csv"),
        "group_freq_df": os.path.join(root, "group_freq.csv"),
        "stable_phrase_df": os.path.join(root, "stable_phrase.csv"),
        "stable_group_df": os.path.join(root, "stable_group.csv"),
        "mri_audit_df": os.path.join(root, "mri_audit.csv"),
        "mri_audit_summary_df": os.path.join(root, "mri_audit_summary.csv"),
        "reliability_df": os.path.join(root, "reliability.csv"),
        "weighted_lexicon_df": os.path.join(root, "weighted_lexicon.csv"),
        "predictions_df": os.path.join(root, "predictions.csv"),
        "fold_metrics_df": os.path.join(root, "fold_metrics.csv"),
        "hyperparameters_df": os.path.join(root, "hyperparameters.csv"),
        "coefficients_df": os.path.join(root, "coefficients.csv"),
    }


def _concat_tables_for_split(tables: List[pd.DataFrame], split_id: str) -> pd.DataFrame:
    if not tables:
        return pd.DataFrame()
    usable = [t for t in tables if isinstance(t, pd.DataFrame) and len(t)]
    if not usable:
        return pd.DataFrame()
    df = pd.concat(usable, ignore_index=True)
    if "split_id" in df.columns:
        df = df[df["split_id"].astype(str) == str(split_id)].copy()
    return df.reset_index(drop=True)


def save_completed_split_checkpoint(
    split_dir: str,
    split_id: str,
    args: argparse.Namespace,
    all_phrase_freq_tables: List[pd.DataFrame],
    all_group_freq_tables: List[pd.DataFrame],
    all_stable_phrase_tables: List[pd.DataFrame],
    all_stable_group_tables: List[pd.DataFrame],
    all_mri_audit_tables: List[pd.DataFrame],
    all_mri_audit_summary_tables: List[pd.DataFrame],
    all_reliability_tables: List[pd.DataFrame],
    all_weighted_lexicon_tables: List[pd.DataFrame],
    prediction_tables: List[pd.DataFrame],
    fold_result_tables: List[pd.DataFrame],
    hyper_tables: List[pd.DataFrame],
    coef_tables: List[pd.DataFrame],
) -> None:
    """Write enough per-split state to skip this split on rerun."""
    paths = _split_resume_paths(split_dir)
    payloads = {
        "phrase_freq_df": _concat_tables_for_split(all_phrase_freq_tables, split_id),
        "group_freq_df": _concat_tables_for_split(all_group_freq_tables, split_id),
        "stable_phrase_df": _concat_tables_for_split(all_stable_phrase_tables, split_id),
        "stable_group_df": _concat_tables_for_split(all_stable_group_tables, split_id),
        "mri_audit_df": _concat_tables_for_split(all_mri_audit_tables, split_id),
        "mri_audit_summary_df": _concat_tables_for_split(all_mri_audit_summary_tables, split_id),
        "reliability_df": _concat_tables_for_split(all_reliability_tables, split_id),
        "weighted_lexicon_df": _concat_tables_for_split(all_weighted_lexicon_tables, split_id),
        "predictions_df": _concat_tables_for_split(prediction_tables, split_id),
        "fold_metrics_df": _concat_tables_for_split(fold_result_tables, split_id),
        "hyperparameters_df": _concat_tables_for_split(hyper_tables, split_id),
        "coefficients_df": _concat_tables_for_split(coef_tables, split_id),
    }
    for key, df in payloads.items():
        _atomic_write_df(df, paths[key])

    marker = {
        "split_id": split_id,
        "completed_at": datetime.now().isoformat(timespec="seconds"),
        "csv_path": getattr(args, "csv_path", ""),
        "outer_scheme": getattr(args, "outer_scheme", ""),
        "outer_repeats": getattr(args, "outer_repeats", None),
        "outer_test_frac": getattr(args, "outer_test_frac", None),
        "outer_folds": getattr(args, "outer_folds", None),
        "random_seed": getattr(args, "random_seed", None),
        "modalities": getattr(args, "modalities", []),
        "representations": getattr(args, "representations", []),
        "fingerprint": build_split_resume_fingerprint(args, split_id),
    }
    marker_path = _split_resume_marker(split_dir)
    tmp_marker = f"{marker_path}.tmp.{os.getpid()}"
    with open(tmp_marker, "w", encoding="utf-8") as f:
        json.dump(marker, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp_marker, marker_path)
    print(f"[RESUME] Wrote completed split checkpoint marker: {marker_path}")


def load_completed_split_checkpoint(
    split_dir: str,
    split_id: str,
    args: Optional[argparse.Namespace] = None,
    outer_train_case_df: Optional[pd.DataFrame] = None,
    outer_test_case_df: Optional[pd.DataFrame] = None,
) -> Optional[Dict[str, pd.DataFrame]]:
    marker_path = _split_resume_marker(split_dir)
    if not os.path.exists(marker_path):
        return None
    try:
        with open(marker_path, "r", encoding="utf-8") as f:
            marker = json.load(f)
        if str(marker.get("split_id")) != str(split_id):
            print(f"[RESUME] Marker split_id mismatch in {marker_path}; recomputing split.")
            return None
        if args is not None:
            ok, msg = validate_split_marker(marker, args, split_id)
            if not ok:
                print(f"[RESUME] Split checkpoint fingerprint mismatch for {split_id}: {msg}; recomputing.")
                return None
        manifest = load_split_manifest(split_dir, split_id)
        if manifest and outer_train_case_df is not None and outer_test_case_df is not None:
            ok, msg = manifest_matches_split_membership(
                manifest,
                outer_train_case_df["case_id"].astype(str).tolist(),
                outer_test_case_df["case_id"].astype(str).tolist(),
            )
            if not ok:
                print(f"[RESUME] Split manifest mismatch for {split_id}: {msg}; recomputing.")
                return None
    except Exception as e:
        print(f"[RESUME] Could not read split marker {marker_path}: {e}; recomputing split.")
        return None

    paths = _split_resume_paths(split_dir)
    loaded = {key: _safe_read_csv_if_exists(path) for key, path in paths.items()}
    ok, msg = validate_completed_checkpoint_tables(loaded, require_predictions=False)
    if not ok:
        print(f"[RESUME] Invalid completed checkpoint tables for {split_id}: {msg}; recomputing.")
        return None
    print(f"[RESUME] Loaded completed split checkpoint for {split_id}: {marker_path}")
    return loaded

def run_outer_split_for_modality(
    raw_df: pd.DataFrame,
    outer_train_case_df: pd.DataFrame,
    outer_test_case_df: pd.DataFrame,
    report_mode: str,
    split_id: str,
    split_dir: str,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    mode_dir = os.path.join(split_dir, report_mode)
    os.makedirs(mode_dir, exist_ok=True)

    print(
        f"[OUTER] split={split_id} report_mode={report_mode} "
        f"n_train={len(outer_train_case_df)} n_test={len(outer_test_case_df)}"
    )

    train_indices = outer_train_case_df["row_index"].astype(int).tolist()
    if report_mode == "mri":
        before = len(train_indices)
        train_indices = [i for i in train_indices if has_usable_mri_report(raw_df, int(i))]
        skipped = before - len(train_indices)
        if skipped > 0:
            print(
                f"[MISSING_MRI] split={split_id} report_mode=mri extraction: "
                f"excluding {skipped} outer-train cases without usable MRI (no API call)."
            )

    t0 = time.time()
    train_records = extract_subset_records(
        df=raw_df,
        row_indices=train_indices,
        report_mode=report_mode,
        split_id=split_id,
        split_role="train",
        sleep_between_calls_s=args.rate_limit_sleep_s,
        max_workers=args.max_api_workers,
        checkpoint_dir=mode_dir,
        resume=args.resume,
        force_reextract=args.force_reextract,
    )
    train_paths = write_extractions(
        extractions=train_records,
        out_dir=mode_dir,
        report_mode=report_mode,
        filename_prefix=f"{split_id}_train",
        split_id=split_id,
    )
    print(f"[OUTER] First-pass train-only extraction completed in {(time.time() - t0)/60:.2f} min.")

    train_extractions_df = load_extractions_csv(train_paths["csv"])
    train_phrase_df = explode_phrase_rows(train_extractions_df)
    train_phrase_csv = os.path.join(mode_dir, f"{split_id}_normalized_phrase_table_train_{report_mode}.csv")
    train_phrase_df.to_csv(train_phrase_csv, index=False)

    phrase_freq_df, group_freq_df, stable_phrase_df, stable_group_df, lexicon_meta = build_stable_lexicon_from_training_extractions(
        train_extractions_df=train_extractions_df,
        train_phrase_df=train_phrase_df,
        rediscovery_scheme=args.rediscovery_scheme,
        rediscovery_repeats=args.rediscovery_repeats,
        rediscovery_test_frac=args.rediscovery_test_frac,
        rediscovery_folds=args.rediscovery_folds,
        stability_threshold=args.stability_threshold,
        min_phrase_cases=args.min_phrase_cases,
        min_group_cases=args.min_group_cases,
        random_seed=args.random_seed + int(split_id.split("_")[-1]),
        target_stable_features_per_modality=getattr(args, "target_stable_features_per_modality", 0),
    )
    lexicon_meta_path = os.path.join(mode_dir, f"{split_id}_stable_lexicon_metadata_{report_mode}.json")
    with open(lexicon_meta_path, "w", encoding="utf-8") as f:
        json.dump(lexicon_meta, f, indent=2, sort_keys=True)
        f.write("\n")

    phrase_freq_csv = os.path.join(mode_dir, f"{split_id}_phrase_rediscovery_frequency_{report_mode}.csv")
    group_freq_csv = os.path.join(mode_dir, f"{split_id}_group_rediscovery_frequency_{report_mode}.csv")
    stable_phrase_csv = os.path.join(mode_dir, f"{split_id}_stable_phrase_lexicon_{report_mode}.csv")
    stable_group_csv = os.path.join(mode_dir, f"{split_id}_stable_group_lexicon_{report_mode}.csv")

    phrase_freq_df.to_csv(phrase_freq_csv, index=False)
    group_freq_df.to_csv(group_freq_csv, index=False)
    stable_phrase_df.to_csv(stable_phrase_csv, index=False)
    stable_group_df.to_csv(stable_group_csv, index=False)

    extra_groups: List[str] = []
    if args.ontology_groups_mode == "stable_plus_ontology":
        extra_groups = list(SHARED_CONCEPT_ONTOLOGY.keys())

    outer_case_df = pd.concat([outer_train_case_df, outer_test_case_df], ignore_index=True).copy()
    phrase_prov_df, group_prov_df = recode_cases_with_frozen_lexicon(
        raw_df=raw_df,
        outer_case_df=outer_case_df,
        report_mode=report_mode,
        stable_phrase_df=stable_phrase_df,
        stable_group_df=stable_group_df,
        split_id=split_id,
        train_case_ids=outer_train_case_df["case_id"].astype(str).tolist(),
        extra_group_names=extra_groups,
    )
    phrase_prov_csv = os.path.join(mode_dir, f"{split_id}_phrase_recode_provenance_{report_mode}.csv")
    group_prov_csv = os.path.join(mode_dir, f"{split_id}_group_recode_provenance_{report_mode}.csv")
    phrase_prov_df.to_csv(phrase_prov_csv, index=False)
    group_prov_df.to_csv(group_prov_csv, index=False)

    group_matrix_df = build_group_feature_matrix(group_prov_df, outer_case_df, split_id, report_mode)
    phrase_matrix_df = build_phrase_feature_matrix(phrase_prov_df, outer_case_df, split_id, report_mode)

    group_matrix_csv = os.path.join(mode_dir, f"{split_id}_group_feature_matrix_{report_mode}.csv")
    phrase_matrix_csv = os.path.join(mode_dir, f"{split_id}_phrase_feature_matrix_{report_mode}.csv")
    group_matrix_df.to_csv(group_matrix_csv, index=False)
    phrase_matrix_df.to_csv(phrase_matrix_csv, index=False)

    audit_df = pd.DataFrame()
    audit_summary_df = pd.DataFrame()
    if report_mode == "mri":
        audit_df = compute_mri_audit_table(raw_df, outer_case_df, phrase_prov_df, group_prov_df, split_id)
        audit_summary_df = summarize_audit_by_groups(audit_df)
        audit_csv = os.path.join(mode_dir, f"{split_id}_mri_audit_case_table.csv")
        audit_summary_csv = os.path.join(mode_dir, f"{split_id}_mri_audit_density_summary.csv")
        audit_df.to_csv(audit_csv, index=False)
        audit_summary_df.to_csv(audit_summary_csv, index=False)
        print(f"[AUDIT] Wrote MRI audit table: {audit_csv}")
        print(f"[AUDIT] Wrote MRI audit summary: {audit_summary_csv}")

    return {
        "train_extractions_df": train_extractions_df,
        "train_phrase_df": train_phrase_df,
        "phrase_freq_df": phrase_freq_df.assign(split_id=split_id, report_mode=report_mode),
        "group_freq_df": group_freq_df.assign(split_id=split_id, report_mode=report_mode),
        "stable_phrase_df": stable_phrase_df.assign(split_id=split_id, report_mode=report_mode),
        "stable_group_df": stable_group_df.assign(split_id=split_id, report_mode=report_mode),
        "phrase_prov_df": phrase_prov_df,
        "group_prov_df": group_prov_df,
        "group_matrix_df": group_matrix_df,
        "phrase_matrix_df": phrase_matrix_df,
        "mri_audit_df": audit_df,
        "mri_audit_summary_df": audit_summary_df,
    }



# -----------------------------
# Fold-level orchestration, reports, and diagnostics
# -----------------------------

def _empty_split_result(split_id: str) -> Dict[str, List[pd.DataFrame]]:
    return {
        "split_id": split_id,
        "all_phrase_freq_tables": [],
        "all_group_freq_tables": [],
        "all_stable_phrase_tables": [],
        "all_stable_group_tables": [],
        "all_mri_audit_tables": [],
        "all_mri_audit_summary_tables": [],
        "all_reliability_tables": [],
        "all_weighted_lexicon_tables": [],
        "prediction_tables": [],
        "fold_result_tables": [],
        "hyper_tables": [],
        "coef_tables": [],
        "error_tables": [],
        "mri_missing_summary_rows": [],
    }


def _result_from_loaded_checkpoint(split_id: str, loaded_split: Dict[str, pd.DataFrame]) -> Dict[str, List[pd.DataFrame]]:
    result = _empty_split_result(split_id)
    mapping = {
        "phrase_freq_df": "all_phrase_freq_tables",
        "group_freq_df": "all_group_freq_tables",
        "stable_phrase_df": "all_stable_phrase_tables",
        "stable_group_df": "all_stable_group_tables",
        "mri_audit_df": "all_mri_audit_tables",
        "mri_audit_summary_df": "all_mri_audit_summary_tables",
        "reliability_df": "all_reliability_tables",
        "weighted_lexicon_df": "all_weighted_lexicon_tables",
        "predictions_df": "prediction_tables",
        "fold_metrics_df": "fold_result_tables",
        "hyperparameters_df": "hyper_tables",
        "coefficients_df": "coef_tables",
    }
    for source_key, result_key in mapping.items():
        df = loaded_split.get(source_key, pd.DataFrame())
        if isinstance(df, pd.DataFrame) and len(df):
            result[result_key].append(df)
    return result


def _extend_aggregate_tables(result: Dict[str, Any], aggregate_lists: Dict[str, List[pd.DataFrame]]) -> None:
    for key, dest in aggregate_lists.items():
        for df in result.get(key, []):
            if isinstance(df, pd.DataFrame) and len(df):
                dest.append(df)


def _case_hash(values: Sequence[Any]) -> str:
    return case_id_list_hash(values)


def write_split_provenance(
    split_dir: str,
    split_id: str,
    outer_train_case_df: pd.DataFrame,
    outer_test_case_df: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    """Persist immutable train/test membership and hashes for audit/resume."""
    os.makedirs(split_dir, exist_ok=True)
    train_ids = outer_train_case_df["case_id"].astype(str).tolist()
    test_ids = outer_test_case_df["case_id"].astype(str).tolist()
    prov = pd.concat([
        outer_train_case_df[["case_id", "row_index", "dispersion_true", "dispersion_true_high_low", "relapse_true"]].assign(split_id=split_id, split_role="train"),
        outer_test_case_df[["case_id", "row_index", "dispersion_true", "dispersion_true_high_low", "relapse_true"]].assign(split_id=split_id, split_role="test"),
    ], ignore_index=True)
    prov.to_csv(os.path.join(split_dir, f"{split_id}_split_provenance.csv"), index=False)
    manifest = {
        "split_id": split_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "random_seed": int(getattr(args, "random_seed", RANDOM_SEED)),
        "outer_scheme": getattr(args, "outer_scheme", ""),
        "outer_repeats": getattr(args, "outer_repeats", None),
        "outer_test_frac": getattr(args, "outer_test_frac", None),
        "outer_folds": getattr(args, "outer_folds", None),
        "n_train": len(train_ids),
        "n_test": len(test_ids),
        "train_case_hash": _case_hash(train_ids),
        "test_case_hash": _case_hash(test_ids),
        "train_case_ids": train_ids,
        "test_case_ids": test_ids,
    }
    with open(os.path.join(split_dir, f"{split_id}_split_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")


def write_failed_split_marker(split_dir: str, split_id: str, exc: BaseException) -> pd.DataFrame:
    os.makedirs(split_dir, exist_ok=True)
    tb = traceback.format_exc()
    payload = {
        "split_id": split_id,
        "failed_at": datetime.now().isoformat(timespec="seconds"),
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "traceback": tb,
    }
    path = os.path.join(_split_resume_dir(split_dir), "FAILED.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"[ERROR] Fold {split_id} failed; wrote failure marker: {path}")
    return pd.DataFrame([payload])


def run_one_outer_split(
    split_num: int,
    train_pos: np.ndarray,
    test_pos: np.ndarray,
    raw_df: pd.DataFrame,
    target_df: pd.DataFrame,
    model_specs: Sequence[ModelSpec],
    standard_representations: Sequence[str],
    weighted_representations: Sequence[str],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """Run one outer split with only split-local mutable state.

    This function is the unit of fold-level parallelism. All artifacts are
    written under this split's own output directory and returned as local
    tables, avoiding concurrent mutation of aggregate lists in the main thread.
    """
    split_id = f"outer_split_{split_num:03d}"
    split_dir = os.path.join(args.out_dir, "outer_splits", split_id)
    os.makedirs(split_dir, exist_ok=True)
    result = _empty_split_result(split_id)

    try:
        manifest = load_split_manifest(split_dir, split_id)
        if manifest and args.resume:
            from approach2.checkpoint import indices_from_manifest
            m_train, m_test = indices_from_manifest(target_df, manifest)
            if m_train and m_test:
                train_pos = np.asarray(m_train, dtype=int)
                test_pos = np.asarray(m_test, dtype=int)
                print(f"[RESUME] Reconstructed {split_id} train/test membership from saved split manifest.")

        outer_train_case_df = target_df.iloc[train_pos].copy().reset_index(drop=True)
        outer_test_case_df = target_df.iloc[test_pos].copy().reset_index(drop=True)
        outer_train_case_df["split_role"] = "train"
        outer_test_case_df["split_role"] = "test"

        train_ids_set = set(outer_train_case_df["case_id"].astype(str))
        test_ids_set = set(outer_test_case_df["case_id"].astype(str))
        overlap_ids = train_ids_set & test_ids_set
        if overlap_ids:
            raise ValueError(
                f"{split_id}: train/test case_id overlap detected ({len(overlap_ids)} cases)."
            )

        if args.resume and args.skip_completed_splits:
            loaded_split = load_completed_split_checkpoint(
                split_dir,
                split_id,
                args=args,
                outer_train_case_df=outer_train_case_df,
                outer_test_case_df=outer_test_case_df,
            )
            if loaded_split is not None:
                print(f"[RESUME] Skipping completed {split_id}; loading per-split outputs into aggregate tables.")
                return _result_from_loaded_checkpoint(split_id, loaded_split)

        write_split_provenance(split_dir, split_id, outer_train_case_df, outer_test_case_df, args)

        print("=" * 100)
        print(
            f"[OUTER] {split_id}: "
            f"n_train={len(outer_train_case_df)} n_test={len(outer_test_case_df)} "
            f"train_pos={int(outer_train_case_df['dispersion_true_high_low'].sum())}/"
            f"{len(outer_train_case_df)} "
            f"test_pos={int(outer_test_case_df['dispersion_true_high_low'].sum())}/"
            f"{len(outer_test_case_df)}"
        )
        print("=" * 100)

        modality_results: Dict[str, Dict[str, Any]] = {}
        modes_to_run = [
            mode for mode in ["mri", "path"]
            if mode in args.modalities or "combined" in args.modalities or args.enable_pathology_calibration or args.enable_teacher_student
        ]

        if len(modes_to_run) > 1 and args.parallel_modality_workers > 1:
            mode_workers = min(args.parallel_modality_workers, len(modes_to_run))
            print(
                f"[OUTER] {split_id} running modalities in parallel with workers={mode_workers}: "
                f"{','.join(modes_to_run)}"
            )
            with ThreadPoolExecutor(max_workers=mode_workers) as executor:
                future_to_mode = {
                    executor.submit(
                        run_outer_split_for_modality,
                        raw_df=raw_df,
                        outer_train_case_df=outer_train_case_df,
                        outer_test_case_df=outer_test_case_df,
                        report_mode=mode,
                        split_id=split_id,
                        split_dir=split_dir,
                        args=args,
                    ): mode
                    for mode in modes_to_run
                }
                for future in as_completed(future_to_mode):
                    mode = future_to_mode[future]
                    modality_results[mode] = future.result()
        else:
            for mode in modes_to_run:
                modality_results[mode] = run_outer_split_for_modality(
                    raw_df=raw_df,
                    outer_train_case_df=outer_train_case_df,
                    outer_test_case_df=outer_test_case_df,
                    report_mode=mode,
                    split_id=split_id,
                    split_dir=split_dir,
                    args=args,
                )

        for mode in modes_to_run:
            result["all_phrase_freq_tables"].append(modality_results[mode]["phrase_freq_df"])
            result["all_group_freq_tables"].append(modality_results[mode]["group_freq_df"])
            result["all_stable_phrase_tables"].append(modality_results[mode]["stable_phrase_df"])
            result["all_stable_group_tables"].append(modality_results[mode]["stable_group_df"])
        if "mri" in modality_results and len(modality_results["mri"].get("mri_audit_df", pd.DataFrame())):
            result["all_mri_audit_tables"].append(modality_results["mri"]["mri_audit_df"])
            result["all_mri_audit_summary_tables"].append(modality_results["mri"]["mri_audit_summary_df"])

        dataset_matrices: Dict[str, Dict[str, pd.DataFrame]] = {}
        if "mri" in args.modalities:
            dataset_matrices["mri"] = {"group": modality_results["mri"]["group_matrix_df"], "phrase": modality_results["mri"]["phrase_matrix_df"]}
        if "path" in args.modalities:
            dataset_matrices["path"] = {"group": modality_results["path"]["group_matrix_df"], "phrase": modality_results["path"]["phrase_matrix_df"]}
        if "combined" in args.modalities:
            dataset_matrices["combined"] = {"group": None, "phrase": None}

        if args.enable_pathology_calibration:
            print(f"[CALIBRATION] {split_id}: computing MRI↔pathology reliability on outer-training cases only.")
            mri_group_matrix, mri_filter_stats = filter_missing_mri_for_dataset(
                modality_results["mri"]["group_matrix_df"], raw_df, "mri_pathcal_weighted", split_id, "all"
            )
            result["mri_missing_summary_rows"].append(mri_filter_stats)
            path_group_matrix = modality_results["path"]["group_matrix_df"]
            mri_available_train_df = outer_train_case_df[
                ~outer_train_case_df["row_index"].astype(int).isin(_mri_missing_row_indices(raw_df))
            ].copy()
            train_ids = mri_available_train_df["case_id"].astype(str).tolist()
            if len(train_ids) == 0:
                print(f"[CALIBRATION] {split_id}: no MRI-available training cases; skipping pathology-informed MRI calibration.")
                reliability_df = pd.DataFrame()
            else:
                reliability_df = compute_cross_modal_reliability(
                    mri_group_matrix_df=mri_group_matrix,
                    path_group_matrix_df=path_group_matrix,
                    train_case_ids=train_ids,
                    split_id=split_id,
                    smoothing=args.calibration_smoothing,
                )
            rel_csv = os.path.join(split_dir, f"{split_id}_mri_pathology_reliability_matrix.csv")
            reliability_df.to_csv(rel_csv, index=False)
            result["all_reliability_tables"].append(reliability_df)
            print(f"[CALIBRATION] Wrote reliability matrix: {rel_csv}")

            weighted_lexicon_df = compute_weighted_mri_lexicon(
                reliability_df=reliability_df,
                mri_group_freq_df=modality_results["mri"]["group_freq_df"],
                mri_group_matrix_df=mri_group_matrix,
                train_case_ids=train_ids,
                y_train_continuous=mri_available_train_df.set_index("case_id")["dispersion_true"],
                split_id=split_id,
                min_selection_frequency=args.weighted_lexicon_min_selection_frequency,
                reliability_power=args.weight_reliability_power,
                stability_power=args.weight_stability_power,
                association_power=args.weight_association_power,
            )
            weighted_lexicon_csv = os.path.join(split_dir, f"{split_id}_weighted_mri_lexicon.csv")
            weighted_lexicon_df.to_csv(weighted_lexicon_csv, index=False)
            result["all_weighted_lexicon_tables"].append(weighted_lexicon_df)
            print(f"[CALIBRATION] Wrote weighted MRI lexicon: {weighted_lexicon_csv}")

            weighted_matrix_df = build_weighted_mri_concept_score_matrix(
                mri_group_matrix_df=mri_group_matrix,
                weighted_lexicon_df=weighted_lexicon_df,
                uncertain_value=args.weighted_uncertain_value,
                negated_value=args.weighted_negated_value,
                split_id=split_id,
            )
            weighted_matrix_csv = os.path.join(split_dir, f"{split_id}_weighted_mri_concept_score_matrix.csv")
            weighted_matrix_df.to_csv(weighted_matrix_csv, index=False)
            dataset_matrices["mri_pathcal_weighted"] = {"weighted": weighted_matrix_df, "group": weighted_matrix_df, "phrase": weighted_matrix_df}
            print(f"[CALIBRATION] Wrote weighted MRI concept score matrix: {weighted_matrix_csv}")

            if args.run_calibration_ablations:
                for ablation_key, mode in [
                    ("mri_pathcal_weighted_random_pathology", "randomized_labels"),
                    ("mri_pathcal_weighted_mismatched_pairing", "mismatched_pairing"),
                ]:
                    path_ablate = randomized_or_mismatched_path_matrix(path_group_matrix, train_ids, mode, args.random_seed + split_num)
                    rel_ablate = compute_cross_modal_reliability(mri_group_matrix, path_ablate, train_ids, split_id, args.calibration_smoothing)
                    w_ablate = compute_weighted_mri_lexicon(rel_ablate, modality_results["mri"]["group_freq_df"], mri_group_matrix, train_ids, mri_available_train_df.set_index("case_id")["dispersion_true"], split_id, args.weighted_lexicon_min_selection_frequency, args.weight_reliability_power, args.weight_stability_power, args.weight_association_power)
                    mat_ablate = build_weighted_mri_concept_score_matrix(mri_group_matrix, w_ablate, args.weighted_uncertain_value, args.weighted_negated_value, split_id)
                    mat_ablate.to_csv(os.path.join(split_dir, f"{split_id}_{ablation_key}_matrix.csv"), index=False)
                    dataset_matrices[ablation_key] = {"weighted": mat_ablate, "group": mat_ablate, "phrase": mat_ablate}
                    print(f"[ABLATION] Built {ablation_key} dataset.")

        for dataset_key in list(dataset_matrices.keys()):
            reps = weighted_representations if "weighted" in dataset_key else standard_representations
            for representation in reps:
                if dataset_key == "combined":
                    if representation.startswith("group_"):
                        dataset_df, feature_cols = merge_modalities_early_fusion(
                            modality_results["mri"]["group_matrix_df"],
                            modality_results["path"]["group_matrix_df"],
                            representation=representation,
                        )
                    elif representation == "phrase_binary":
                        dataset_df, feature_cols = merge_modalities_early_fusion(
                            modality_results["mri"]["phrase_matrix_df"],
                            modality_results["path"]["phrase_matrix_df"],
                            representation=representation,
                        )
                    else:
                        continue
                elif "weighted" in dataset_key:
                    dataset_df = dataset_matrices[dataset_key]["weighted"].copy()
                    X_tmp, feature_cols = get_representation_matrix(dataset_df, representation)
                    dataset_df = pd.concat([
                        dataset_df[["case_id", "row_index", "dispersion_true", "dispersion_true_high_low", "relapse_true"]],
                        X_tmp,
                    ], axis=1)
                else:
                    source_key = "group" if representation.startswith("group_") else "phrase"
                    dataset_df = dataset_matrices[dataset_key][source_key].copy()
                    X_tmp, feature_cols = get_representation_matrix(dataset_df, representation)
                    dataset_df = pd.concat([
                        dataset_df[["case_id", "row_index", "dispersion_true", "dispersion_true_high_low", "relapse_true"]],
                        X_tmp,
                    ], axis=1)

                dataset_df = dataset_df.drop_duplicates(subset=["case_id", "row_index"]).reset_index(drop=True)
                train_pre = dataset_df[dataset_df["case_id"].isin(outer_train_case_df["case_id"])].copy()
                test_pre = dataset_df[dataset_df["case_id"].isin(outer_test_case_df["case_id"])].copy()
                dataset_df, filter_stats = filter_missing_mri_for_dataset(
                    dataset_df, raw_df, dataset_key, split_id, "all"
                )
                _, train_stats = filter_missing_mri_for_dataset(
                    train_pre, raw_df, dataset_key, split_id, "train"
                )
                _, test_stats = filter_missing_mri_for_dataset(
                    test_pre, raw_df, dataset_key, split_id, "test"
                )
                result["mri_missing_summary_rows"].extend([filter_stats, train_stats, test_stats])
                dataset_df["split_id"] = split_id

                if len(feature_cols) == 0:
                    print(f"[SKIP] split={split_id} dataset={dataset_key} representation={representation} has no features after recoding.")
                    continue

                train_mask = dataset_df["case_id"].isin(outer_train_case_df["case_id"])
                test_mask = dataset_df["case_id"].isin(outer_test_case_df["case_id"])
                train_df = dataset_df.loc[train_mask].copy().reset_index(drop=True)
                test_df = dataset_df.loc[test_mask].copy().reset_index(drop=True)

                for spec in model_specs:
                    for target_spec in model_target_specs_for_model(spec):
                        target_name = target_spec["target_name"]
                        target_col = target_spec["target_col"]
                        X_train, y_train, X_test, y_test = prepare_task_frames(
                            train_df=train_df,
                            test_df=test_df,
                            feature_cols=feature_cols,
                            target_col=target_col,
                            task_type=spec.task_type,
                        )
                        if should_skip_model_fit(
                            y_train=y_train,
                            y_test=y_test,
                            task_type=spec.task_type,
                            split_id=split_id,
                            dataset_key=dataset_key,
                            representation=representation,
                            model_key=spec.key,
                            target_name=target_name,
                        ):
                            continue

                        print(
                            f"[MODEL] split={split_id} dataset={dataset_key} "
                            f"representation={representation} model={spec.key} "
                            f"target={target_name} n_features={len(feature_cols)} "
                            f"n_train={len(y_train)} n_test={len(y_test)}"
                        )
                        pred_df, hyper, coef_df = fit_one_outer_model(
                            spec=spec,
                            X_train=X_train,
                            y_train=y_train,
                            X_test=X_test,
                            split_random_seed=args.random_seed + split_num,
                            ml_n_jobs=args.ml_n_jobs,
                        )
                        test_meta = test_df[test_df[target_col].notna()].copy().reset_index(drop=True)
                        meta_pred = test_meta[["case_id", "row_index"]].copy()
                        meta_pred["split_id"] = split_id
                        meta_pred["dataset_key"] = dataset_key
                        meta_pred["representation"] = representation
                        meta_pred["model_key"] = spec.key
                        meta_pred["task_type"] = spec.task_type
                        meta_pred["target_name"] = target_name
                        meta_pred["target_col"] = target_col
                        meta_pred["y_true"] = y_test.values
                        meta_pred = pd.concat([meta_pred.reset_index(drop=True), pred_df.reset_index(drop=True)], axis=1)
                        result["prediction_tables"].append(meta_pred)

                        metrics = classification_metrics(meta_pred) if spec.task_type == "classification" else regression_metrics(meta_pred)
                        metrics.update({
                            "split_id": split_id,
                            "dataset_key": dataset_key,
                            "representation": representation,
                            "model_key": spec.key,
                            "task_type": spec.task_type,
                            "target_name": target_name,
                            "target_col": target_col,
                            "family": spec.family,
                            "estimator_name": spec.estimator_name,
                            "notes": spec.notes,
                        })
                        result["fold_result_tables"].append(pd.DataFrame([metrics]))

                        hyper.update({
                            "split_id": split_id,
                            "dataset_key": dataset_key,
                            "representation": representation,
                            "model_key": spec.key,
                            "task_type": spec.task_type,
                            "target_name": target_name,
                            "target_col": target_col,
                        })
                        result["hyper_tables"].append(pd.DataFrame([hyper]))

                        if len(coef_df):
                            coef_df = annotate_coefficient_table(coef_df, X_train, X_test)
                            coef_df = coef_df.copy()
                            coef_df["split_id"] = split_id
                            coef_df["dataset_key"] = dataset_key
                            coef_df["representation"] = representation
                            coef_df["model_key"] = spec.key
                            coef_df["task_type"] = spec.task_type
                            coef_df["target_name"] = target_name
                            coef_df["target_col"] = target_col
                            result["coef_tables"].append(coef_df)

        if args.enable_teacher_student:
            try:
                representation = "group_status"
                mri_df = modality_results["mri"]["group_matrix_df"].drop_duplicates(subset=["case_id", "row_index"]).reset_index(drop=True)
                path_df = modality_results["path"]["group_matrix_df"].drop_duplicates(subset=["case_id", "row_index"]).reset_index(drop=True)
                X_mri_all, mri_cols = get_representation_matrix(mri_df, representation)
                X_path_all, path_cols = get_representation_matrix(path_df, representation)
                mri_model_df = pd.concat([mri_df[["case_id", "row_index", "dispersion_true", "dispersion_true_high_low", "relapse_true"]], X_mri_all], axis=1)
                mri_model_df, ts_filter_stats = filter_missing_mri_for_dataset(
                    mri_model_df, raw_df, "mri_teacher_student", split_id, "all"
                )
                result["mri_missing_summary_rows"].append(ts_filter_stats)
                path_model_df = pd.concat([path_df[["case_id", "row_index", "dispersion_true", "dispersion_true_high_low", "relapse_true"]], X_path_all], axis=1)
                train_mask = mri_model_df["case_id"].isin(outer_train_case_df["case_id"])
                test_mask = mri_model_df["case_id"].isin(outer_test_case_df["case_id"])

                if len(mri_cols) > 0 and len(path_cols) > 0:
                    train_mri = mri_model_df.loc[train_mask].reset_index(drop=True)
                    test_mri = mri_model_df.loc[test_mask].reset_index(drop=True)
                    train_pair = train_mri[["case_id", "row_index"] + mri_cols].merge(
                        path_model_df[["case_id", "row_index"] + path_cols],
                        on=["case_id", "row_index"],
                        how="inner",
                    )
                    train_mri = train_mri.merge(
                        train_pair[["case_id", "row_index"]],
                        on=["case_id", "row_index"],
                        how="inner",
                    ).reset_index(drop=True)
                    train_pair = train_mri[["case_id", "row_index"] + mri_cols].merge(
                        path_model_df[["case_id", "row_index"] + path_cols],
                        on=["case_id", "row_index"],
                        how="inner",
                    )
                    X_mri_train = train_pair[mri_cols]
                    X_mri_test = test_mri[mri_cols]
                    X_path_train = train_pair[path_cols]
                    path_concept_train = train_pair[["case_id", "row_index"] + path_cols].copy()
                    reg_pred, cls_pred, relapse_pred, ts_extra = fit_teacher_student_mri_model(
                        X_mri_train=X_mri_train,
                        X_mri_test=X_mri_test,
                        X_path_train=X_path_train,
                        y_train=train_mri["dispersion_true"].astype(float),
                        y_test_cont=test_mri["dispersion_true"].astype(float),
                        y_test_binary=test_mri["dispersion_true_high_low"].astype(int),
                        path_concept_train=path_concept_train,
                        split_random_seed=args.random_seed + split_num,
                        alpha=args.teacher_student_alpha,
                        lambda_dispersion=args.teacher_student_lambda_dispersion,
                        lambda_teacher_score=args.teacher_student_lambda_teacher_score,
                        lambda_path_concepts=args.teacher_student_lambda_path_concepts,
                        y_train_relapse=train_mri["relapse_true"],
                        y_test_relapse=test_mri["relapse_true"],
                    )
                    pred_specs = [
                        ("regression", reg_pred, TARGET_NAME_DISPERSION_SCORE, "dispersion_true"),
                        ("classification", cls_pred, TARGET_NAME_DISPERSION_HIGH_LOW, "dispersion_true_high_low"),
                    ]
                    if relapse_pred is not None:
                        pred_specs.append(
                            ("classification", relapse_pred, TARGET_NAME_RELAPSE_STATUS, "relapse_true")
                        )
                    for task_type, pred_df, target_name, target_col in pred_specs:
                        meta_pred = test_mri[["case_id", "row_index"]].copy()
                        meta_pred["split_id"] = split_id
                        meta_pred["dataset_key"] = "mri_teacher_student"
                        meta_pred["representation"] = representation
                        meta_pred["model_key"] = "multitask_ridge_student"
                        meta_pred["task_type"] = task_type
                        meta_pred["target_name"] = target_name
                        meta_pred["target_col"] = target_col
                        meta_pred = pd.concat([meta_pred.reset_index(drop=True), pred_df.reset_index(drop=True)], axis=1)
                        result["prediction_tables"].append(meta_pred)
                        metrics = regression_metrics(meta_pred) if task_type == "regression" else classification_metrics(meta_pred)
                        metrics.update({
                            "split_id": split_id,
                            "dataset_key": "mri_teacher_student",
                            "representation": representation,
                            "model_key": "multitask_ridge_student",
                            "task_type": task_type,
                            "target_name": target_name,
                            "target_col": target_col,
                            "family": "teacher_student_multitask_ridge",
                            "estimator_name": "Multi-output Ridge student",
                            "notes": "MRI-only inputs trained with true dispersion, pathology teacher score, and pathology concept outputs from outer-train only.",
                        })
                        result["fold_result_tables"].append(pd.DataFrame([metrics]))
                    ts_hyper = ts_extra["hyper"]
                    ts_hyper.update({
                        "split_id": split_id,
                        "dataset_key": "mri_teacher_student",
                        "representation": representation,
                        "model_key": "multitask_ridge_student",
                        "task_type": "regression_and_thresholded_classification",
                        "target_name": TARGET_NAME_DISPERSION_SCORE,
                        "target_col": "dispersion_true",
                    })
                    result["hyper_tables"].append(pd.DataFrame([ts_hyper]))
                    coef_df = ts_extra["coef_df"]
                    if len(coef_df):
                        coef_df = annotate_coefficient_table(coef_df, X_mri_train, X_mri_test)
                        coef_df["split_id"] = split_id
                        coef_df["dataset_key"] = "mri_teacher_student"
                        coef_df["representation"] = representation
                        coef_df["model_key"] = "multitask_ridge_student"
                        coef_df["task_type"] = "regression"
                        coef_df["target_name"] = TARGET_NAME_DISPERSION_SCORE
                        coef_df["target_col"] = "dispersion_true"
                        result["coef_tables"].append(coef_df)
                    print(f"[TEACHER_STUDENT] Completed {split_id} MRI-only multitask student.")
            except Exception as e:
                print(f"[WARN] Teacher-student model failed for {split_id}: {e}")

        if args.resume:
            save_completed_split_checkpoint(
                split_dir=split_dir,
                split_id=split_id,
                args=args,
                all_phrase_freq_tables=result["all_phrase_freq_tables"],
                all_group_freq_tables=result["all_group_freq_tables"],
                all_stable_phrase_tables=result["all_stable_phrase_tables"],
                all_stable_group_tables=result["all_stable_group_tables"],
                all_mri_audit_tables=result["all_mri_audit_tables"],
                all_mri_audit_summary_tables=result["all_mri_audit_summary_tables"],
                all_reliability_tables=result["all_reliability_tables"],
                all_weighted_lexicon_tables=result["all_weighted_lexicon_tables"],
                prediction_tables=result["prediction_tables"],
                fold_result_tables=result["fold_result_tables"],
                hyper_tables=result["hyper_tables"],
                coef_tables=result["coef_tables"],
            )
        return result
    except Exception as e:
        print(f"[ERROR] {split_id} failed with {type(e).__name__}: {e}")
        print(traceback.format_exc())
        result["error_tables"].append(write_failed_split_marker(split_dir, split_id, e))
        return result


def coordinate_parallelism(args: argparse.Namespace) -> argparse.Namespace:
    """Resolve worker controls and prevent obvious CPU/API oversubscription."""
    cpu_count = os.cpu_count() or 1
    args.parallel_fold_workers = max(1, int(getattr(args, "parallel_fold_workers", 1) or 1))
    args.max_api_workers = max(1, int(args.max_api_workers))
    args.parallel_modality_workers = max(1, int(args.parallel_modality_workers))
    args.ml_n_jobs = max(1, int(args.ml_n_jobs))

    # max_api_workers is treated as a process-wide cap enforced by a semaphore in
    # the auxiliary API client, not as a per-fold quota.
    configure_global_api_concurrency(args.max_api_workers)

    requested_cpu_slots = args.parallel_fold_workers * max(1, args.parallel_modality_workers) * max(1, args.ml_n_jobs)
    if requested_cpu_slots > max(1, cpu_count):
        new_ml_n_jobs = max(1, cpu_count // max(1, args.parallel_fold_workers * args.parallel_modality_workers))
        if new_ml_n_jobs < args.ml_n_jobs:
            print(
                f"[PARALLEL] Reducing ml_n_jobs from {args.ml_n_jobs} to {new_ml_n_jobs} "
                f"to avoid oversubscription: fold_workers={args.parallel_fold_workers}, "
                f"modality_workers={args.parallel_modality_workers}, cpus={cpu_count}."
            )
            args.ml_n_jobs = new_ml_n_jobs
    return args


def deduplicate_outer_predictions(pred_all: pd.DataFrame) -> pd.DataFrame:
    """Create one held-out prediction per case/model by averaging repeats.

    Stratified 5-fold CV already yields one held-out prediction per case. For
    repeated Monte Carlo splits, a case can appear in multiple outer-test sets;
    this aggregation keeps the final metric layer aligned with the one-case-one-
    prediction rule while preserving the raw per-split predictions separately.
    """
    if pred_all is None or len(pred_all) == 0:
        return pd.DataFrame()
    df = pred_all.copy()
    group_cols = ["dataset_key", "representation", "model_key", "task_type", "target_name", "target_col", "case_id", "row_index"]
    for col in group_cols:
        if col not in df.columns:
            df[col] = "unknown"
    rows: List[Dict[str, Any]] = []
    for keys, sub in df.groupby(group_cols, dropna=False):
        rec = dict(zip(group_cols, keys))
        rec["n_outer_test_predictions_aggregated"] = int(len(sub))
        rec["split_ids"] = ";".join(sorted(set(sub["split_id"].astype(str)))) if "split_id" in sub.columns else ""
        rec["y_true"] = sub["y_true"].dropna().iloc[0] if "y_true" in sub.columns and sub["y_true"].notna().any() else np.nan
        if str(rec["task_type"]) == "classification":
            rec["y_prob"] = float(pd.to_numeric(sub.get("y_prob", pd.Series(dtype=float)), errors="coerce").mean())
            rec["y_pred"] = int(rec["y_prob"] >= 0.5) if pd.notna(rec["y_prob"]) else np.nan
        else:
            rec["y_pred_value"] = float(pd.to_numeric(sub.get("y_pred_value", pd.Series(dtype=float)), errors="coerce").mean())
        rows.append(rec)
    return pd.DataFrame(rows)


def compute_metrics_from_predictions(pred_df: pd.DataFrame, args: argparse.Namespace, seed_offset: int = 0) -> pd.DataFrame:
    metrics_rows: List[Dict[str, Any]] = []
    if pred_df is None or len(pred_df) == 0:
        return pd.DataFrame()
    group_cols = ["dataset_key", "representation", "model_key", "task_type", "target_name"]
    for group_values, sub in pred_df.groupby(group_cols):
        dataset_key, representation, model_key, task_type, target_name = group_values
        metrics = classification_metrics(sub) if task_type == "classification" else regression_metrics(sub)
        metrics = add_bootstrap_metric_cis(
            base_metrics=metrics,
            pred_df=sub,
            task_type=task_type,
            n_bootstrap=args.bootstrap_n,
            random_seed=args.random_seed + seed_offset,
        )
        metrics.update({
            "dataset_key": dataset_key,
            "representation": representation,
            "model_key": model_key,
            "task_type": task_type,
            "target_name": target_name,
        })
        metrics_rows.append(metrics)
    return pd.DataFrame(metrics_rows)


def _best_row(metrics_df: pd.DataFrame, task_type: str, target_name: Optional[str] = None, dataset_key: Optional[str] = None) -> Optional[pd.Series]:
    if metrics_df is None or len(metrics_df) == 0:
        return None
    df = metrics_df[metrics_df["task_type"].astype(str) == task_type].copy()
    if target_name is not None and "target_name" in df.columns:
        df = df[df["target_name"].astype(str) == str(target_name)].copy()
    if dataset_key is not None:
        df = df[df["dataset_key"].astype(str) == str(dataset_key)].copy()
    if len(df) == 0:
        return None
    if task_type == "regression":
        return df.sort_values(["mae", "spearman_rho"], ascending=[True, False]).iloc[0]
    if target_name == TARGET_NAME_RELAPSE_STATUS:
        return df.sort_values(["auprc", "auroc", "brier"], ascending=[False, False, True]).iloc[0]
    return df.sort_values(["auroc", "auprc", "f1"], ascending=[False, False, False]).iloc[0]
