"""Per-config inference loop and aggregate summary."""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Tuple

import pandas as pd

from . import config
from .api import (
    get_default_tracker,
    load_cost_tracker_snapshot,
    print_cumulative_report,
    reset_cost_tracker,
    save_cumulative_report_json_shim,
)
from .checkpoint.predictions import (
    append_prediction_jsonl,
    load_existing_case_predictions,
    predictions_dict_to_dataframe,
    write_predictions_csv,
    write_predictions_jsonl,
)
from .checkpoint.resume import (
    config_completed_marker_path,
    load_completed_config_checkpoint,
    save_completed_config_checkpoint,
)
from .evaluation.runner import evaluate_and_plot
from .inference import predict_case
from .models import RunConfig
from .prompts.tokens import make_case_token
from .schema.records import build_pred_record
from .splits import write_run_config, write_skipped_cases
from .text_utils import modality_display_name


def run_one_config(
    rc: RunConfig,
    *,
    resume: bool = True,
    skip_completed_configs: bool = True,
    force_rerun_cases: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    os.makedirs(rc.run_out_dir, exist_ok=True)
    predictions_csv = os.path.join(rc.run_out_dir, "predictions_testing_cases.csv")
    predictions_jsonl = os.path.join(rc.run_out_dir, "predictions_testing_cases.jsonl")
    cost_json = os.path.join(rc.run_out_dir, "token_cost_report.json")
    completed_marker = config_completed_marker_path(rc.run_out_dir)

    write_run_config(rc)
    write_skipped_cases(rc)

    if resume and skip_completed_configs:
        loaded = load_completed_config_checkpoint(rc.run_out_dir, rc)
        if loaded is not None:
            pred_df, eval_summary = loaded
            return pred_df, eval_summary

    if not resume or force_rerun_cases:
        if os.path.isfile(completed_marker):
            os.remove(completed_marker)
            print(f"[RESUME] Removed completed checkpoint: {completed_marker}")

    prior_cost = load_cost_tracker_snapshot(cost_json) if resume and not force_rerun_cases else None
    reset_cost_tracker()
    get_default_tracker().configure_persist(
        cost_json,
        prior=prior_cost if resume else None,
    )

    existing_by_row: Dict[int, Dict[str, Any]] = {}
    if resume and not force_rerun_cases:
        existing_by_row, source, warnings = load_existing_case_predictions(rc.run_out_dir, rc)
        for w in warnings:
            print(f"[RESUME][WARN] {w}")
        if existing_by_row:
            print(
                f"[RESUME] Loaded {len(existing_by_row)}/{len(rc.test_cases_with_idxs)} "
                f"existing case predictions from {source or 'disk'}"
            )

    if not resume or force_rerun_cases:
        open(predictions_jsonl, "w", encoding="utf-8").close()

    print("=" * 80)
    print(f"[RUN] {rc.shotset_name} / {rc.modality}")
    print(f"[RUN] Output directory: {rc.run_out_dir}")
    print(f"[RUN] Training rows: high={rc.high_rows}, low={rc.low_rows}")
    print(f"[RUN] Test cases: {len(rc.test_cases_with_idxs)}")
    print(f"[RUN] Skipped missing MRI: {len(rc.skipped_missing_mri)}")
    print(f"[RUN] Resume enabled: {resume} | force_rerun_cases: {force_rerun_cases}")
    print(f"[RUN] Cases already present: {len(existing_by_row)}")
    print(f"[RUN] Cases pending API: {len(rc.test_cases_with_idxs) - len(existing_by_row)}")
    print("=" * 80)

    n_new_api_calls = 0
    t0 = time.time()

    for n, (idx, test_case) in enumerate(rc.test_cases_with_idxs, 1):
        validation_token = make_case_token(test_case, idx, rc.modality)
        print("-" * 80)
        print(
            f"[CASE] {n}/{len(rc.test_cases_with_idxs)} "
            f"row_index={idx} case_id={test_case.case_id} modality={rc.modality}"
        )

        if resume and not force_rerun_cases and idx in existing_by_row:
            pred_record = existing_by_row[idx]
            print(
                "[RESUME] Skipping API call; reusing saved prediction: "
                f"dispersion_pred={pred_record['dispersion_score_pred']:.2f} "
                f"dispersion_high_low_pred={pred_record['dispersion_high_low_pred']} "
                f"relapse_pred={pred_record['relapse_pred']}"
            )
            continue

        print(f"[CASE] preop_MRI_chars={len(test_case.preop_mri)} path_report_chars={len(test_case.path_report)}")
        pred_obj = predict_case(rc.training_block, test_case, row_index=idx, modality=rc.modality)
        pred_record = build_pred_record(rc, idx, test_case, pred_obj, validation_token)
        existing_by_row[idx] = pred_record
        append_prediction_jsonl(predictions_jsonl, pred_record)
        n_new_api_calls += 1

        print(
            "[CASE] Prediction OK: "
            f"dispersion_pred={pred_record['dispersion_score_pred']:.2f} "
            f"dispersion_high_low_pred={pred_record['dispersion_high_low_pred']} "
            f"relapse_pred={pred_record['relapse_pred']} "
            f"token_match={pred_record['retrieval_token_exact_match']}"
        )
        time.sleep(config.RATE_LIMIT_SLEEP_S)

    elapsed = time.time() - t0
    print(
        f"[DONE] Inference pass complete for {rc.shotset_name}/{rc.modality} in {elapsed / 60:.2f} minutes. "
        f"new_api_calls={n_new_api_calls}"
    )

    if len(existing_by_row) != len(rc.test_cases_with_idxs):
        missing = [idx for idx, _ in rc.test_cases_with_idxs if idx not in existing_by_row]
        raise RuntimeError(
            f"Incomplete predictions for {rc.shotset_name}/{rc.modality}. "
            f"Missing row indices: {missing}"
        )

    ordered_records = [existing_by_row[idx] for idx, _ in rc.test_cases_with_idxs]
    write_predictions_jsonl(predictions_jsonl, ordered_records)
    pred_df = predictions_dict_to_dataframe(existing_by_row, rc)
    write_predictions_csv(predictions_csv, pred_df)
    print(f"[SAVE] Wrote predictions JSONL: {predictions_jsonl}")
    print(f"[SAVE] Wrote predictions CSV: {predictions_csv}")

    print_cumulative_report()
    save_cumulative_report_json_shim(cost_json, prior=prior_cost if resume else None)
    get_default_tracker().configure_persist(None)
    print(f"[COST] Wrote token/cost report: {cost_json}")

    title_suffix = f"{rc.shotset_name} / {modality_display_name(rc.modality)}"
    eval_summary = evaluate_and_plot(pred_df, rc.run_out_dir, title_suffix)
    save_completed_config_checkpoint(rc.run_out_dir, rc, n_new_api_calls)
    return pred_df, eval_summary


def save_aggregate_summary(root_out_dir: str, summaries: List[Dict[str, Any]]) -> None:
    rows = []
    for s in summaries:
        metrics = s.get("metrics", {})
        rows.append({
            "shotset_name": s["shotset_name"],
            "modality": s["modality"],
            "n_predictions": metrics.get("n_rows"),
            "n_skipped_missing_mri": s.get("n_skipped_missing_mri"),
            "dispersion_mae": metrics.get("dispersion_regression", {}).get("mae"),
            "dispersion_rmse": metrics.get("dispersion_regression", {}).get("rmse"),
            "dispersion_spearman_rho": metrics.get("dispersion_regression", {}).get("spearman_rho"),
            "dispersion_high_low_accuracy": metrics.get("dispersion_high_low", {}).get("accuracy"),
            "dispersion_high_low_f1": metrics.get("dispersion_high_low", {}).get("f1"),
            "dispersion_high_low_auroc": metrics.get("dispersion_high_low", {}).get("auroc"),
            "dispersion_high_low_auprc": metrics.get("dispersion_high_low", {}).get("auprc"),
            "relapse_accuracy": metrics.get("relapse_label", {}).get("accuracy"),
            "relapse_f1": metrics.get("relapse_label", {}).get("f1"),
            "needle_single_token_rate": metrics.get("needle_retrieval", {}).get("single_token_rate"),
        })
    df = pd.DataFrame(rows)
    path = os.path.join(root_out_dir, "all_tiers_metrics_summary.csv")
    df.to_csv(path, index=False)
    print(f"[SUMMARY] Wrote aggregate metrics summary: {path}")


def refresh_evaluations_from_predictions(root_out_dir: str) -> int:
    """Re-run evaluate_and_plot from saved predictions CSVs (no API calls)."""
    from .evaluation.results_report import discover_config_dirs

    summaries: List[Dict[str, Any]] = []
    n_refreshed = 0
    for shotset_name, modality, config_dir in discover_config_dirs(root_out_dir):
        pred_csv = os.path.join(config_dir, "predictions_testing_cases.csv")
        if not os.path.isfile(pred_csv):
            print(f"[RE-EVAL] Skipping {shotset_name}/{modality}: no predictions_testing_cases.csv")
            continue
        pred_df = pd.read_csv(pred_csv)
        run_cfg_path = os.path.join(config_dir, "run_config.json")
        run_cfg: Dict[str, Any] = {}
        if os.path.isfile(run_cfg_path):
            with open(run_cfg_path, "r", encoding="utf-8") as f:
                run_cfg = json.load(f)
        title_suffix = f"{shotset_name} / {modality_display_name(modality)}"
        print(f"[RE-EVAL] Refreshing metrics: {title_suffix}")
        eval_summary = evaluate_and_plot(pred_df, config_dir, title_suffix)
        summaries.append({
            "shotset_name": shotset_name,
            "modality": modality,
            "n_skipped_missing_mri": run_cfg.get("n_skipped_missing_mri", 0),
            "metrics": eval_summary,
        })
        n_refreshed += 1
    if summaries:
        save_aggregate_summary(root_out_dir, summaries)
    return n_refreshed
