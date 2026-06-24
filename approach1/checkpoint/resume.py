"""Resume planning and completed-config checkpoint management."""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .. import config
from ..api import get_default_tracker
from ..evaluation.runner import evaluate_and_plot
from ..io_atomic import atomic_write_json
from ..models import RunConfig
from ..text_utils import modality_display_name
from .fingerprint import build_config_fingerprint, config_fingerprints_compatible
from .predictions import (
    load_existing_case_predictions,
    predictions_dict_to_dataframe,
)


def config_resume_dir(run_out_dir: str) -> str:
    path = os.path.join(run_out_dir, config.RESUME_CHECKPOINT_SUBDIR)
    os.makedirs(path, exist_ok=True)
    return path


def config_completed_marker_path(run_out_dir: str) -> str:
    return os.path.join(config_resume_dir(run_out_dir), "COMPLETED.json")


def is_config_checkpoint_complete(run_out_dir: str, rc: RunConfig) -> bool:
    marker_path = config_completed_marker_path(run_out_dir)
    if not os.path.isfile(marker_path):
        return False
    try:
        with open(marker_path, "r", encoding="utf-8") as f:
            marker = json.load(f)
    except Exception:
        return False
    ok, _ = config_fingerprints_compatible(marker.get("fingerprint", {}), rc)
    if not ok:
        return False
    by_row, _, _ = load_existing_case_predictions(run_out_dir, rc)
    return len(by_row) == len(rc.test_cases_with_idxs)


def save_completed_config_checkpoint(
    run_out_dir: str,
    rc: RunConfig,
    n_new_api_calls: int,
) -> None:
    tracker = get_default_tracker()
    marker = {
        "completed_at": datetime.now().isoformat(timespec="seconds"),
        "fingerprint": build_config_fingerprint(rc),
        "predictions_csv": os.path.join(run_out_dir, "predictions_testing_cases.csv"),
        "predictions_jsonl": os.path.join(run_out_dir, "predictions_testing_cases.jsonl"),
        "metrics_json": os.path.join(run_out_dir, "evaluation_metrics_summary.json"),
        "cost_json": os.path.join(run_out_dir, "token_cost_report.json"),
        "n_test_cases": len(rc.test_cases_with_idxs),
        "n_new_api_calls_last_session": n_new_api_calls,
        "session_estimated_cost_usd": float(tracker.to_dict()["estimated_cost_usd"]),
    }
    marker_path = config_completed_marker_path(run_out_dir)
    atomic_write_json(marker_path, marker)
    print(f"[RESUME] Wrote completed config checkpoint: {marker_path}")


def load_completed_config_checkpoint(
    run_out_dir: str,
    rc: RunConfig,
) -> Optional[Tuple[pd.DataFrame, Dict[str, Any]]]:
    marker_path = config_completed_marker_path(run_out_dir)
    if not os.path.isfile(marker_path):
        return None

    try:
        with open(marker_path, "r", encoding="utf-8") as f:
            marker = json.load(f)
    except Exception as exc:
        print(f"[RESUME] Ignoring unreadable completed checkpoint {marker_path}: {exc}")
        return None

    ok, msg = config_fingerprints_compatible(marker.get("fingerprint", {}), rc)
    if not ok:
        print(f"[RESUME] Ignoring completed checkpoint due to fingerprint mismatch: {msg}")
        return None

    by_row, source, warnings = load_existing_case_predictions(run_out_dir, rc)
    for w in warnings:
        print(f"[RESUME][WARN] {w}")
    if len(by_row) != len(rc.test_cases_with_idxs):
        print(
            f"[RESUME] Completed marker exists but only {len(by_row)}/{len(rc.test_cases_with_idxs)} "
            "valid case records were found; will rerun this config."
        )
        return None

    pred_csv = marker.get("predictions_csv") or os.path.join(run_out_dir, "predictions_testing_cases.csv")
    if not os.path.isfile(pred_csv):
        print(f"[RESUME] Completed marker exists but predictions CSV is missing: {pred_csv}")
        return None

    pred_df = predictions_dict_to_dataframe(by_row, rc)
    metrics_json = marker.get("metrics_json") or os.path.join(run_out_dir, "evaluation_metrics_summary.json")
    eval_summary: Dict[str, Any] = {}
    if os.path.isfile(metrics_json):
        try:
            with open(metrics_json, "r", encoding="utf-8") as f:
                eval_summary = json.load(f)
        except Exception as exc:
            print(f"[RESUME][WARN] Failed to load metrics JSON ({metrics_json}): {exc}")

    title_suffix = f"{rc.shotset_name} / {modality_display_name(rc.modality)}"
    if not eval_summary:
        print(f"[RESUME] Metrics JSON missing; recomputing evaluation for {title_suffix}")
        eval_summary = evaluate_and_plot(pred_df, run_out_dir, title_suffix)
    else:
        print(f"[RESUME] Loaded completed config from checkpoint ({source or 'marker'}): {marker_path}")
    return pred_df, eval_summary


def summarize_resume_plan(
    run_configs: List[RunConfig],
    *,
    resume: bool,
    skip_completed_configs: bool,
    force_rerun_cases: bool,
) -> Dict[str, Any]:
    summary = {
        "resume_enabled": resume,
        "skip_completed_configs": skip_completed_configs,
        "force_rerun_cases": force_rerun_cases,
        "n_configs_total": len(run_configs),
        "n_configs_skip_complete": 0,
        "n_configs_resume_cases": 0,
        "n_configs_refinalize": 0,
        "n_configs_run_fresh": 0,
        "n_cases_total": 0,
        "n_cases_already_done": 0,
        "n_cases_pending_api": 0,
        "per_config": [],
    }
    if not resume:
        for rc in run_configs:
            n_test = len(rc.test_cases_with_idxs)
            summary["n_cases_total"] += n_test
            summary["n_cases_pending_api"] += n_test
            summary["n_configs_run_fresh"] += 1
            summary["per_config"].append({
                "shotset_name": rc.shotset_name,
                "modality": rc.modality,
                "status": "fresh_no_resume",
                "n_done": 0,
                "n_pending": n_test,
            })
        return summary

    for rc in run_configs:
        n_test = len(rc.test_cases_with_idxs)
        summary["n_cases_total"] += n_test
        if skip_completed_configs and is_config_checkpoint_complete(rc.run_out_dir, rc):
            summary["n_configs_skip_complete"] += 1
            summary["n_cases_already_done"] += n_test
            status = "skip_complete"
            n_done = n_test
            n_pending = 0
        else:
            by_row, _, _ = (
                ({}, "", [])
                if force_rerun_cases
                else load_existing_case_predictions(rc.run_out_dir, rc)
            )
            n_done = len(by_row)
            n_pending = n_test - n_done
            summary["n_cases_already_done"] += n_done
            summary["n_cases_pending_api"] += n_pending
            if n_done > 0 and n_pending > 0:
                summary["n_configs_resume_cases"] += 1
                status = "resume_partial"
            elif n_done == 0:
                summary["n_configs_run_fresh"] += 1
                status = "fresh"
            else:
                summary["n_configs_refinalize"] += 1
                status = "all_cases_present_refinalize"
        summary["per_config"].append({
            "shotset_name": rc.shotset_name,
            "modality": rc.modality,
            "status": status,
            "n_done": n_done,
            "n_pending": n_pending,
        })
    return summary


def print_resume_plan(summary: Dict[str, Any]) -> None:
    print("\n[RESUME PLAN]")
    print(f"resume_enabled:           {summary['resume_enabled']}")
    print(f"skip_completed_configs:   {summary['skip_completed_configs']}")
    print(f"force_rerun_cases:        {summary['force_rerun_cases']}")
    print(f"configs total:            {summary['n_configs_total']}")
    print(f"configs skip complete:    {summary['n_configs_skip_complete']}")
    print(f"configs resume partial:   {summary['n_configs_resume_cases']}")
    print(f"configs refinalize only:  {summary['n_configs_refinalize']}")
    print(f"configs run fresh:        {summary['n_configs_run_fresh']}")
    print(f"cases total:              {summary['n_cases_total']}")
    print(f"cases already done:       {summary['n_cases_already_done']}")
    print(f"cases pending API:        {summary['n_cases_pending_api']}")
    for item in summary["per_config"]:
        print(
            f"  {item['shotset_name']}/{item['modality']}: "
            f"status={item['status']} done={item['n_done']} pending={item['n_pending']}"
        )
