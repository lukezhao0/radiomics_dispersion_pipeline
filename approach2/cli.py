"""Nested evaluation CLI entry point."""

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
    DEFAULT_BOOTSTRAP_N,
    DEFAULT_STABILITY_THRESHOLD,
    RANDOM_SEED,
    TARGET_NAME_DISPERSION_HIGH_LOW,
    TARGET_NAME_DISPERSION_SCORE,
    TARGET_NAME_RELAPSE_STATUS,
)
from approach2.extraction import (
    MAX_TOKENS,
    Tee,
    _is_missing_text,
    _selected_report_text,
    _true_dispersion_high_low,
    build_chat_messages,
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

from approach2.eval_data import (
    ensure_case_id,
    get_target_frame,
    summarize_cohort_report_availability,
    write_mri_missing_case_summary,
)
from approach2.evaluation.plots import (
    rank_features_across_models,
    summarize_coefficient_sign_stability,
)
from approach2.features.normalize import ontology_table
from approach2.models_ml import get_model_specs
from approach2.orchestration import (
    _empty_split_result,
    _extend_aggregate_tables,
    _split_resume_marker,
    coordinate_parallelism,
    compute_metrics_from_predictions,
    deduplicate_outer_predictions,
    run_one_outer_split,
    write_failed_split_marker,
)
from approach2.extraction.config import DEFAULT_MODEL, ENV_PATH, configure_llm
from common.llm_models import SUPPORTED_MODELS
from common.reasoning_effort import DEFAULT_REASONING_EFFORT, REASONING_EFFORT_CHOICES
from approach2.splits import build_outer_splits, log_outer_split_summary, validate_outer_splits
from approach2.api.cost import (
    initialize_cost_tracker_for_resume,
    set_cost_persist_path,
    write_apriori_cost_estimate_json,
)
from approach2.reports import (
    generate_all_reports,
    pathology_metrics_on_mri_complete,
    permutation_test_relapse_metrics,
    regenerate_reports_and_plots,
    relapse_split_diagnostics,
    summarize_metrics,
    write_methodology_markdown,
)

# -----------------------------
# Main evaluation driver
# -----------------------------

def _add_bool_arg(parser: argparse.ArgumentParser, name: str, default: bool, help_text: str) -> None:
    dest = name.replace("-", "_")
    # For names such as enable-pathology-calibration, expose the natural pair
    # --enable-pathology-calibration / --disable-pathology-calibration.
    disable_name = name[len("enable-"):] if name.startswith("enable-") else name
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(f"--{name}", dest=dest, action="store_true", help=help_text)
    group.add_argument(f"--disable-{disable_name}", dest=dest, action="store_false", help=f"Disable: {help_text}")
    parser.set_defaults(**{dest: default})


def planned_llm_extraction_modes(args: argparse.Namespace) -> List[str]:
    """Return report modes requiring LLM extraction for this run."""
    return [
        mode for mode in ["mri", "path"]
        if (
            mode in args.modalities
            or "combined" in args.modalities
            or args.enable_pathology_calibration
            or args.enable_teacher_student
        )
    ]


def estimate_nested_pipeline_llm_cost(
    raw_df: pd.DataFrame,
    target_df: pd.DataFrame,
    outer_splits: Sequence[Tuple[np.ndarray, np.ndarray]],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """Build the exact prompts scheduled by nested extraction and estimate cost."""
    prompt_counts: List[int] = []
    prompt_modes: List[str] = []
    skipped_completed_splits = 0
    modes_to_run = planned_llm_extraction_modes(args)

    for split_num, (train_pos, _test_pos) in enumerate(outer_splits, 1):
        split_id = f"outer_split_{split_num:03d}"
        split_dir = os.path.join(args.out_dir, "outer_splits", split_id)
        if args.resume and args.skip_completed_splits and os.path.exists(_split_resume_marker(split_dir)):
            skipped_completed_splits += 1
            continue

        outer_train_case_df = target_df.iloc[train_pos].copy().reset_index(drop=True)
        for mode in modes_to_run:
            for row_index in outer_train_case_df["row_index"].astype(int).tolist():
                case = make_case_from_row(raw_df, int(row_index))
                if _is_missing_text(_selected_report_text(case, mode)):
                    continue
                prompt = build_user_prompt(case, mode)
                prompt_counts.append(estimate_prompt_tokens_from_messages(build_chat_messages(prompt)))
                prompt_modes.append(mode)

    estimate = summarize_apriori_cost_estimate(
        prompt_token_counts=prompt_counts,
        report_modes=prompt_modes,
        max_completion_tokens=MAX_TOKENS,
    )
    estimate["n_outer_splits"] = len(outer_splits)
    estimate["n_completed_splits_skipped_in_estimate"] = skipped_completed_splits
    estimate["planned_report_modes"] = modes_to_run
    return estimate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-path", type=str, required=True, help="Path to the raw CSV with MRI/pathology reports and true outcomes.")
    parser.add_argument("--out_dir", type=str, required=True, help="Directory to write nested-resampling outputs.")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        choices=list(SUPPORTED_MODELS),
        help="LLM deployment to use (gpt-5-nano uses SANDBOX_API_KEY; gpt-5 uses NEW_SECUREGPT_API_KEY).",
    )
    parser.add_argument(
        "--env-path",
        type=str,
        default=ENV_PATH,
        help="Path to .env containing model-specific API keys.",
    )
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        default=DEFAULT_REASONING_EFFORT,
        choices=list(REASONING_EFFORT_CHOICES),
        help="GPT-5 reasoning effort sent to the API (default: minimal; use 'none' to omit).",
    )
    parser.add_argument("--outer-scheme", type=str, default="repeated_mc", choices=["repeated_mc", "stratified_kfold"], help="Outer resampling design.")
    parser.add_argument("--outer-repeats", type=int, default=5, help="Number of repeated 80/20 Monte Carlo outer splits when --outer-scheme repeated_mc.")
    parser.add_argument("--outer-test-frac", type=float, default=0.20, help="Outer test fraction for repeated Monte Carlo splitting.")
    parser.add_argument("--outer-folds", type=int, default=5, help="Number of folds when --outer-scheme stratified_kfold.")
    parser.add_argument("--rediscovery-scheme", type=str, default="repeated_mc", choices=["repeated_mc", "stratified_kfold"], help="Training-only rediscovery design used to estimate lexicon selection frequency.")
    parser.add_argument("--rediscovery-repeats", type=int, default=25, help="Number of rediscovery Monte Carlo resamples when --rediscovery-scheme repeated_mc.")
    parser.add_argument("--rediscovery-test-frac", type=float, default=0.20, help="Test fraction for rediscovery Monte Carlo splits.")
    parser.add_argument("--rediscovery-folds", type=int, default=5, help="Number of folds when --rediscovery-scheme stratified_kfold.")
    parser.add_argument("--stability-threshold", type=float, default=0.60, help="USE DEFAULT_STABILITY_THRESHOLD FOR 0.20, RIGHT NOW HARD CODED 0.60- Selection-frequency threshold used to freeze the outer-split stable lexicon (default 0.20).")
    parser.add_argument("--target-stable-features-per-modality", type=int, default=0, help="Cap final stable phrase features per modality to this count (0 = no cap, use all stable). Ranking uses train-only selection frequency.")
    parser.add_argument("--min-phrase-cases", type=int, default=2, help="Minimum number of training cases in a rediscovery subset required for a phrase to count as rediscovered.")
    parser.add_argument("--min-group-cases", type=int, default=2, help="Minimum number of training cases in a rediscovery subset required for a group to count as rediscovered.")
    parser.add_argument("--modalities", nargs="+", default=["mri", "path", "combined"], choices=["mri", "path", "combined"], help="Modalities to evaluate.")
    parser.add_argument("--representations", nargs="+", default=["group_binary", "group_count", "group_status", "phrase_binary"], choices=["group_binary", "group_count", "group_status", "phrase_binary", "weighted_concept_score", "weighted_plus_group_status"], help="Feature representations to evaluate.")
    parser.add_argument("--rate-limit-sleep-s", type=float, default=0.25, help="Sleep interval between SecureGPT extraction calls.")
    parser.add_argument("--parallel-fold-workers", type=int, default=1, help="Maximum number of outer CV folds to run concurrently. Use 5 to run 5-fold CV folds simultaneously when resources/API limits allow.")
    parser.add_argument("--max-api-workers", type=int, default=2, help="Global cap on simultaneous Stanford AI Sandbox HTTP requests across all folds/modalities/cases.")
    parser.add_argument("--parallel-modality-workers", type=int, default=2, help="Maximum number of modalities to process concurrently within each outer split.")
    parser.add_argument("--ml-n-jobs", type=int, default=2, help="Number of local CPU workers for GridSearchCV in the classical ML stage after fold/modality coordination.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED, help="Base random seed.")
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        default=True,
        help="Disable resume behavior. By default, case checkpoints and completed split checkpoints are reused.",
    )
    parser.add_argument(
        "--force-reextract",
        action="store_true",
        default=False,
        help="Ignore per-case extraction checkpoints and call SecureGPT again. Completed split checkpoints are still skipped unless --no-skip-completed-splits is also used.",
    )
    parser.add_argument(
        "--no-skip-completed-splits",
        dest="skip_completed_splits",
        action="store_false",
        default=True,
        help="When resuming, recompute outer splits even if their completed split checkpoint exists.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        default=False,
        help="Accept the printed a-priori LLM cost estimate and skip the interactive confirmation prompt.",
    )
    parser.add_argument(
        "--regenerate-reports-only",
        action="store_true",
        default=False,
        help="Rebuild plots and HTML reports from saved nested_outer_* artifacts without rerunning extraction or model fitting.",
    )
    parser.add_argument("--bootstrap-n", type=int, default=DEFAULT_BOOTSTRAP_N, help="Number of case-level bootstrap resamples for final aggregate metric confidence intervals. Use 0 to disable.")
    parser.add_argument("--relapse-permutation-n", type=int, default=1000, help="Number of label permutations for relapse AUROC/AUPRC empirical p-values. Use 0 to disable.")

    _add_bool_arg(parser, "enable-pathology-calibration", True, "Enable leakage-aware pathology-informed MRI weighting.")
    _add_bool_arg(parser, "enable-teacher-student", True, "Enable MRI-only teacher-student multi-task ridge model.")
    parser.add_argument("--ontology-groups-mode", type=str, default="stable_plus_ontology", choices=["stable_only", "stable_plus_ontology"], help="Whether recoding should include all ontology groups in addition to stable groups.")
    parser.add_argument("--weighted-lexicon-min-selection-frequency", type=float, default=0.0, help="Minimum rediscovery frequency floor for weighted MRI concept lexicon.")
    parser.add_argument("--weighted-uncertain-value", type=float, default=0.5, help="Numeric value assigned to uncertain concept evidence in weighted scores.")
    parser.add_argument("--weighted-negated-value", type=float, default=0.0, help="Numeric value assigned to negated concept evidence in weighted scores.")
    parser.add_argument("--calibration-smoothing", type=float, default=0.5, help="Additive smoothing for cross-modal reliability estimates.")
    parser.add_argument("--weight-reliability-power", type=float, default=1.0, help="Exponent for pathology concordance in MRI concept weight formula.")
    parser.add_argument("--weight-stability-power", type=float, default=0.5, help="Exponent for rediscovery stability in MRI concept weight formula.")
    parser.add_argument("--weight-association-power", type=float, default=0.5, help="Exponent for univariate dispersion association in MRI concept weight formula.")
    _add_bool_arg(parser, "run-calibration-ablations", False, "Run randomized-pathology and mismatched-pairing calibration ablations.")
    parser.add_argument("--teacher-student-alpha", type=float, default=10.0, help="Fixed ridge alpha for teacher-student model.")
    parser.add_argument("--teacher-student-lambda-dispersion", type=float, default=1.0, help="Loss weight for true continuous dispersion target.")
    parser.add_argument("--teacher-student-lambda-teacher-score", type=float, default=0.5, help="Loss weight for pathology-teacher score target.")
    parser.add_argument("--teacher-student-lambda-path-concepts", type=float, default=0.25, help="Loss weight for pathology concept targets.")

    args = parser.parse_args()

    configure_llm(args.model, env_path=args.env_path, reasoning_effort=args.reasoning_effort)

    cost_report_path = os.path.join(args.out_dir, "llm_token_cost_report.json")
    set_cost_persist_path(cost_report_path, resume=args.resume)
    initialize_cost_tracker_for_resume(cost_report_path, resume=args.resume)

    args.max_api_workers = resolve_default_api_workers(args.max_api_workers)
    args.parallel_modality_workers = resolve_default_parallel_modality_workers(args.parallel_modality_workers)
    args.ml_n_jobs = resolve_default_ml_n_jobs(args.ml_n_jobs)
    args = coordinate_parallelism(args)

    os.makedirs(args.out_dir, exist_ok=True)
    if args.regenerate_reports_only:
        regenerate_reports_and_plots(args.out_dir, args)
        print("[DONE] Report and plot regeneration complete.")
        return

    log_dir = os.path.join(args.out_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "run_log_feature_discovery_nested_eval.txt")

    log_mode = "a" if args.resume and os.path.exists(log_path) else "w"
    with open(log_path, log_mode, encoding="utf-8") as log_f:
        tee_out = Tee(sys.__stdout__, log_f)
        tee_err = Tee(sys.__stderr__, log_f)

        with contextlib.redirect_stdout(tee_out), contextlib.redirect_stderr(tee_err):
            print("\n" + "#" * 100)
            print(f"[RUN_START] {datetime.now().isoformat(timespec='seconds')}")
            print(f"[RUN_START] model={args.model}")
            print(f"[RUN_START] reasoning_effort={args.reasoning_effort}")
            print(f"[RUN_START] env_path={args.env_path}")
            print(f"[RUN_START] resume={args.resume} force_reextract={args.force_reextract} skip_completed_splits={args.skip_completed_splits}")
            print("#" * 100)

            print(f"[LOG] Logging stdout/stderr to: {log_path}")
            print(
                f"[PARALLEL] parallel_fold_workers={args.parallel_fold_workers} "
                f"parallel_modality_workers={args.parallel_modality_workers} "
                f"max_api_workers_global={args.max_api_workers} ml_n_jobs={args.ml_n_jobs}"
            )
            print(
                f"[CONFIG] outer_scheme={args.outer_scheme} outer_repeats={args.outer_repeats} "
                f"outer_test_frac={args.outer_test_frac} outer_folds={args.outer_folds} "
                f"stability_threshold={args.stability_threshold} "
                f"target_stable_features_per_modality={args.target_stable_features_per_modality}"
            )
            print(f"[LOAD] Reading raw CSV: {args.csv_path}")
            raw_df = load_cases(args.csv_path)
            raw_df = ensure_case_id(raw_df)
            target_df = get_target_frame(raw_df)
            cohort_flags = summarize_cohort_report_availability(raw_df, target_df)
            print(
                f"[LOAD] Loaded target-eligible cases={cohort_flags['n_total_eligible_cases']} "
                f"usable_pathology={cohort_flags['n_usable_pathology']} "
                f"usable_mri={cohort_flags['n_usable_mri']} "
                f"missing_mri={cohort_flags['n_missing_mri']}"
            )

            outer_splits = build_outer_splits(
                y_binary=target_df["dispersion_true_high_low"].astype(int).values,
                scheme=args.outer_scheme,
                random_seed=args.random_seed,
                n_repeats=args.outer_repeats,
                test_frac=args.outer_test_frac,
                n_folds=args.outer_folds,
            )
            validate_outer_splits(outer_splits, args.outer_scheme, len(target_df))
            log_outer_split_summary(
                outer_splits,
                target_df["dispersion_true_high_low"].astype(int).values,
                args.outer_scheme,
                args.outer_repeats,
                args.outer_test_frac,
                args.outer_folds,
            )

            estimate = estimate_nested_pipeline_llm_cost(raw_df, target_df, outer_splits, args)
            print_apriori_cost_estimate_report(estimate, label="nested outer-training extraction pipeline")
            print(f"[A-PRIORI] n_outer_splits={estimate['n_outer_splits']} completed_splits_skipped={estimate['n_completed_splits_skipped_in_estimate']} planned_report_modes={estimate['planned_report_modes']}")
            write_apriori_cost_estimate_json(args.out_dir, estimate, label="nested outer-training extraction pipeline")
            confirm_cost_estimate_or_exit(estimate, assume_yes=args.yes)

            preflight_check()

            ontology_csv = os.path.join(args.out_dir, "shared_biological_concept_ontology.csv")
            ontology_table().to_csv(ontology_csv, index=False)
            print(f"[SAVE] Wrote ontology table: {ontology_csv}")

            methodology_md = write_methodology_markdown(args.out_dir, args)
            print(f"[SAVE] Wrote methodology summary: {methodology_md}")

            print(f"[OUTER] Generated n_outer_splits={len(outer_splits)}")

            all_phrase_freq_tables: List[pd.DataFrame] = []
            all_group_freq_tables: List[pd.DataFrame] = []
            all_stable_phrase_tables: List[pd.DataFrame] = []
            all_stable_group_tables: List[pd.DataFrame] = []
            all_mri_audit_tables: List[pd.DataFrame] = []
            all_mri_audit_summary_tables: List[pd.DataFrame] = []
            all_reliability_tables: List[pd.DataFrame] = []
            all_weighted_lexicon_tables: List[pd.DataFrame] = []

            prediction_tables: List[pd.DataFrame] = []
            fold_result_tables: List[pd.DataFrame] = []
            hyper_tables: List[pd.DataFrame] = []
            coef_tables: List[pd.DataFrame] = []
            error_tables: List[pd.DataFrame] = []
            mri_missing_summary_rows: List[Dict[str, Any]] = []

            model_specs = get_model_specs()
            standard_representations = [r for r in args.representations if not r.startswith("weighted")]
            weighted_representations = [r for r in args.representations if r.startswith("weighted")]
            if args.enable_pathology_calibration and not weighted_representations:
                weighted_representations = ["weighted_concept_score"]

            aggregate_lists = {
                "all_phrase_freq_tables": all_phrase_freq_tables,
                "all_group_freq_tables": all_group_freq_tables,
                "all_stable_phrase_tables": all_stable_phrase_tables,
                "all_stable_group_tables": all_stable_group_tables,
                "all_mri_audit_tables": all_mri_audit_tables,
                "all_mri_audit_summary_tables": all_mri_audit_summary_tables,
                "all_reliability_tables": all_reliability_tables,
                "all_weighted_lexicon_tables": all_weighted_lexicon_tables,
                "prediction_tables": prediction_tables,
                "fold_result_tables": fold_result_tables,
                "hyper_tables": hyper_tables,
                "coef_tables": coef_tables,
                "error_tables": error_tables,
            }

            split_jobs = [
                (split_num, train_pos, test_pos)
                for split_num, (train_pos, test_pos) in enumerate(outer_splits, 1)
            ]

            if args.parallel_fold_workers > 1 and len(split_jobs) > 1:
                fold_workers = min(args.parallel_fold_workers, len(split_jobs))
                print(
                    f"[OUTER_PARALLEL] Running outer folds concurrently with workers={fold_workers}. "
                    f"Each fold writes only to its own outer_splits/<split_id>/ directory."
                )
                with ThreadPoolExecutor(max_workers=fold_workers) as executor:
                    future_to_split = {
                        executor.submit(
                            run_one_outer_split,
                            split_num=split_num,
                            train_pos=train_pos,
                            test_pos=test_pos,
                            raw_df=raw_df,
                            target_df=target_df,
                            model_specs=model_specs,
                            standard_representations=standard_representations,
                            weighted_representations=weighted_representations,
                            args=args,
                        ): f"outer_split_{split_num:03d}"
                        for split_num, train_pos, test_pos in split_jobs
                    }
                    for future in as_completed(future_to_split):
                        split_id = future_to_split[future]
                        try:
                            split_result = future.result()
                        except Exception as e:
                            split_dir = os.path.join(args.out_dir, "outer_splits", split_id)
                            split_result = _empty_split_result(split_id)
                            split_result["error_tables"].append(write_failed_split_marker(split_dir, split_id, e))
                        _extend_aggregate_tables(split_result, aggregate_lists)
                        for row in split_result.get("mri_missing_summary_rows", []):
                            if isinstance(row, dict):
                                mri_missing_summary_rows.append(row)
                        print(f"[OUTER_PARALLEL] Collected results for {split_id}.")
            else:
                print("[OUTER_PARALLEL] Fold-level parallelism disabled; running outer folds sequentially.")
                for split_num, train_pos, test_pos in split_jobs:
                    split_result = run_one_outer_split(
                        split_num=split_num,
                        train_pos=train_pos,
                        test_pos=test_pos,
                        raw_df=raw_df,
                        target_df=target_df,
                        model_specs=model_specs,
                        standard_representations=standard_representations,
                        weighted_representations=weighted_representations,
                        args=args,
                    )
                    _extend_aggregate_tables(split_result, aggregate_lists)

                    _extend_aggregate_tables(split_result, aggregate_lists)
                    for row in split_result.get("mri_missing_summary_rows", []):
                        if isinstance(row, dict):
                            mri_missing_summary_rows.append(row)

            cohort_row = {
                "split_id": "cohort_overall",
                "dataset_key": "all",
                "split_role": "eligible",
                "n_before_filter": cohort_flags["n_total_eligible_cases"],
                "n_skipped_missing_mri": cohort_flags["n_missing_mri"],
                "n_after_filter": cohort_flags["n_usable_mri"],
                "drop_reason": "cohort_summary",
                "dropped_case_ids": cohort_flags["missing_mri_case_ids"],
            }
            mri_missing_summary_rows.append(cohort_row)
            write_mri_missing_case_summary(args.out_dir, mri_missing_summary_rows)
            with open(os.path.join(args.out_dir, "cohort_report_availability_summary.json"), "w", encoding="utf-8") as f:
                json.dump(cohort_flags, f, indent=2, sort_keys=True)
                f.write("\n")

            # Save aggregate lexicon/calibration/audit summaries.
            if all_phrase_freq_tables:
                phrase_freq_all = pd.concat(all_phrase_freq_tables, ignore_index=True)
                phrase_freq_all.to_csv(os.path.join(args.out_dir, "all_outer_phrase_rediscovery_frequencies.csv"), index=False)
            else:
                phrase_freq_all = pd.DataFrame()

            if all_group_freq_tables:
                group_freq_all = pd.concat(all_group_freq_tables, ignore_index=True)
                group_freq_all.to_csv(os.path.join(args.out_dir, "all_outer_group_rediscovery_frequencies.csv"), index=False)
            else:
                group_freq_all = pd.DataFrame()

            if all_stable_phrase_tables:
                stable_phrase_all = pd.concat(all_stable_phrase_tables, ignore_index=True)
                stable_phrase_summary = stable_phrase_all.groupby(["report_mode", "phrase_slug", "quote_norm", "canonical_group"]).size().reset_index(name="n_outer_splits_stable").sort_values(["report_mode", "n_outer_splits_stable"], ascending=[True, False])
                stable_phrase_summary.to_csv(os.path.join(args.out_dir, "stable_phrase_lexicon_outer_summary.csv"), index=False)
            else:
                stable_phrase_summary = pd.DataFrame()

            if all_stable_group_tables:
                stable_group_all = pd.concat(all_stable_group_tables, ignore_index=True)
                stable_group_summary = stable_group_all.groupby(["report_mode", "canonical_group"]).size().reset_index(name="n_outer_splits_stable").sort_values(["report_mode", "n_outer_splits_stable"], ascending=[True, False])
                stable_group_summary.to_csv(os.path.join(args.out_dir, "stable_group_lexicon_outer_summary.csv"), index=False)
            else:
                stable_group_summary = pd.DataFrame()

            if all_mri_audit_tables:
                pd.concat(all_mri_audit_tables, ignore_index=True).to_csv(os.path.join(args.out_dir, "mri_audit_case_table_all_outer_splits.csv"), index=False)
            if all_mri_audit_summary_tables:
                pd.concat(all_mri_audit_summary_tables, ignore_index=True).to_csv(os.path.join(args.out_dir, "mri_audit_density_summary_all_outer_splits.csv"), index=False)
            if all_reliability_tables:
                reliability_all = pd.concat(all_reliability_tables, ignore_index=True)
                reliability_all.to_csv(os.path.join(args.out_dir, "all_outer_mri_pathology_reliability_matrices.csv"), index=False)
            else:
                reliability_all = pd.DataFrame()
            if all_weighted_lexicon_tables:
                weighted_lexicon_all = pd.concat(all_weighted_lexicon_tables, ignore_index=True)
                weighted_lexicon_all.to_csv(os.path.join(args.out_dir, "all_outer_weighted_mri_lexicons.csv"), index=False)
            else:
                weighted_lexicon_all = pd.DataFrame()

            pred_all = pd.concat(prediction_tables, ignore_index=True) if prediction_tables else pd.DataFrame()
            if len(pred_all) and "target_name" not in pred_all.columns:
                pred_all["target_name"] = np.where(
                    pred_all["task_type"].astype(str) == "regression",
                    TARGET_NAME_DISPERSION_SCORE,
                    TARGET_NAME_DISPERSION_HIGH_LOW,
                )
                pred_all["target_col"] = np.where(
                    pred_all["task_type"].astype(str) == "regression",
                    "dispersion_true",
                    "dispersion_true_high_low",
                )
            pred_case_all = deduplicate_outer_predictions(pred_all)
            fold_results_all = pd.concat(fold_result_tables, ignore_index=True) if fold_result_tables else pd.DataFrame()
            hyper_all = pd.concat(hyper_tables, ignore_index=True) if hyper_tables else pd.DataFrame()
            coef_all = pd.concat(coef_tables, ignore_index=True) if coef_tables else pd.DataFrame()
            error_all = pd.concat(error_tables, ignore_index=True) if error_tables else pd.DataFrame()

            pred_out = os.path.join(args.out_dir, "nested_outer_predictions_all.csv")
            pred_case_out = os.path.join(args.out_dir, "nested_outer_predictions_case_deduplicated.csv")
            fold_out = os.path.join(args.out_dir, "nested_outer_fold_metrics_all.csv")
            hyper_out = os.path.join(args.out_dir, "nested_outer_hyperparameters_all.csv")
            coef_out = os.path.join(args.out_dir, "nested_outer_feature_coefficients_all.csv")
            error_out = os.path.join(args.out_dir, "nested_outer_split_errors.csv")

            pred_all.to_csv(pred_out, index=False)
            pred_case_all.to_csv(pred_case_out, index=False)
            fold_results_all.to_csv(fold_out, index=False)
            hyper_all.to_csv(hyper_out, index=False)
            coef_all.to_csv(coef_out, index=False)
            error_all.to_csv(error_out, index=False)

            print(f"[SAVE] Wrote raw per-split predictions: {pred_out}")
            print(f"[SAVE] Wrote case-deduplicated predictions: {pred_case_out}")
            print(f"[SAVE] Wrote fold-level metrics: {fold_out}")
            print(f"[SAVE] Wrote hyperparameter summary: {hyper_out}")
            print(f"[SAVE] Wrote coefficient summary: {coef_out}")
            if len(error_all):
                print(f"[WARN] One or more outer folds failed. Error summary: {error_out}")

            metrics_df = compute_metrics_from_predictions(pred_case_all, args)
            metrics_csv = os.path.join(args.out_dir, "nested_outer_metrics_summary.csv")
            metrics_df.to_csv(metrics_csv, index=False)
            print(f"[SAVE] Wrote case-deduplicated metrics summary: {metrics_csv}")

            feature_rank_cls_df = rank_features_across_models([coef_all[coef_all["task_type"] == "classification"].assign(task_type="classification")] if len(coef_all) else [], task_type="classification")
            feature_rank_reg_df = rank_features_across_models([coef_all[coef_all["task_type"] == "regression"].assign(task_type="regression")] if len(coef_all) else [], task_type="regression")

            feature_rank_cls_csv = os.path.join(args.out_dir, "nested_feature_ranking_classification.csv")
            feature_rank_reg_csv = os.path.join(args.out_dir, "nested_feature_ranking_regression.csv")
            feature_rank_cls_df.to_csv(feature_rank_cls_csv, index=False)
            feature_rank_reg_df.to_csv(feature_rank_reg_csv, index=False)
            print(f"[SAVE] Wrote classification feature ranking: {feature_rank_cls_csv}")
            print(f"[SAVE] Wrote regression feature ranking: {feature_rank_reg_csv}")

            sign_stability_df = summarize_coefficient_sign_stability(coef_all)
            sign_stability_csv = os.path.join(args.out_dir, "nested_feature_sign_stability.csv")
            sign_stability_df.to_csv(sign_stability_csv, index=False)
            print(f"[SAVE] Wrote coefficient sign stability: {sign_stability_csv}")

            relapse_balance_df = relapse_split_diagnostics(target_df, outer_splits, args.out_dir)
            permutation_df = permutation_test_relapse_metrics(pred_case_all, metrics_df, args, args.out_dir)
            path_mri_subset_metrics_df = pathology_metrics_on_mri_complete(pred_case_all, raw_df, args)
            if len(path_mri_subset_metrics_df):
                path_mri_subset_metrics_df.to_csv(os.path.join(args.out_dir, "pathology_only_mri_complete_subset_metrics.csv"), index=False)

            summary_txt = os.path.join(args.out_dir, "nested_resampling_summary.txt")
            summary = summarize_metrics(metrics_df)
            with open(summary_txt, "w", encoding="utf-8") as f:
                f.write(summary)

            generate_all_reports(args.out_dir, csv_path=args.csv_path, force=True)

            print(f"[SAVE] Wrote summary text: {summary_txt}")
            print("\n" + summary)
            print_cumulative_report(label="full nested pipeline (final, resume-stable)")
            write_cost_tracker_json(args.out_dir)
            print("[DONE] Nested resampling evaluation complete.")


if __name__ == "__main__":
    main()
