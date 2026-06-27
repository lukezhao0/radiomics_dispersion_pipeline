"""Tests for Approach 2 standalone report generation."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

PIPELINE_ROOT = Path(__file__).resolve().parents[2]


def test_report_cli_help():
    proc = subprocess.run(
        [sys.executable, str(PIPELINE_ROOT / "approach2_generate_reports.py"), "--help"],
        capture_output=True,
        text=True,
        cwd=PIPELINE_ROOT,
    )
    assert proc.returncode == 0
    assert "--run-dir" in proc.stdout
    assert "--force" in proc.stdout


def test_generate_reports_on_complete_run(tmp_path):
    complete_run = PIPELINE_ROOT.parent / "sabcs" / "securegpt_dispersion_approach2"
    if not complete_run.is_dir():
        pytest.skip("Complete Approach 2 run directory not available")

    from approach2.reports import generate_all_reports

    out_dir = tmp_path / "reports_out"
    out_dir.mkdir()
    for name in [
        "nested_outer_metrics_summary.csv",
        "nested_outer_predictions_case_deduplicated.csv",
        "nested_outer_fold_metrics_all.csv",
        "nested_outer_feature_coefficients_all.csv",
        "nested_feature_sign_stability.csv",
        "all_outer_phrase_rediscovery_frequencies.csv",
        "all_outer_group_rediscovery_frequencies.csv",
        "stable_phrase_lexicon_outer_summary.csv",
        "stable_group_lexicon_outer_summary.csv",
        "relapse_class_balance_by_split.csv",
        "nested_resampling_summary.txt",
    ]:
        src = complete_run / name
        if src.is_file():
            (out_dir / name).write_bytes(src.read_bytes())

    paths = generate_all_reports(str(out_dir), force=True)
    assert Path(paths["results_html"]).is_file()
    assert Path(paths["interpretability_html"]).is_file()
    assert Path(paths["missed_case_html"]).is_file()
    assert (out_dir / "report_plots").is_dir()


def test_generate_reports_minimal_synthetic(tmp_path):
    from approach2.reports import generate_all_reports

    out_dir = tmp_path
    metrics = pd.DataFrame([{
        "target_name": "dispersion_score",
        "dataset_key": "mri",
        "representation": "group_binary",
        "model_key": "ridge_regression",
        "task_type": "regression",
        "n": 10,
        "mae": 12.5,
        "rmse": 15.0,
        "spearman_rho": 0.4,
        "r2": 0.1,
    }])
    preds = pd.DataFrame([{
        "case_id": "c1",
        "row_index": 0,
        "split_id": "outer_split_001",
        "dataset_key": "mri",
        "representation": "group_binary",
        "model_key": "ridge_regression",
        "task_type": "regression",
        "target_name": "dispersion_score",
        "y_true": 50.0,
        "y_pred_value": 48.0,
    }])
    metrics.to_csv(out_dir / "nested_outer_metrics_summary.csv", index=False)
    preds.to_csv(out_dir / "nested_outer_predictions_case_deduplicated.csv", index=False)
    (out_dir / "llm_cost_estimate_apriori_initial.json").write_text(json.dumps({
        "n_calls": 756,
        "estimated_prompt_tokens": 2965776,
        "estimated_completion_cap_tokens": 12096000,
        "no_cache_estimated_cost_usd": 4.9866888,
        "cache_aware_estimated_cost_usd": 4.92920384,
        "cache_aware_estimated_cached_tokens": 1437124,
        "cache_aware_estimated_cache_savings_usd": 0.05748496,
        "n_completed_splits_skipped_in_estimate": 0,
        "n_calls_skipped_existing_checkpoints": 0,
        "estimate_kind": "full_pipeline_initial",
    }), encoding="utf-8")
    (out_dir / "llm_cost_estimate_apriori.json").write_text(json.dumps({
        "n_calls": 282,
        "estimated_prompt_tokens": 1101897,
        "estimated_completion_cap_tokens": 4512000,
        "no_cache_estimated_cost_usd": 1.85989485,
        "cache_aware_estimated_cost_usd": 1.83854765,
        "cache_aware_estimated_cached_tokens": 533680,
        "cache_aware_estimated_cache_savings_usd": 0.0213472,
        "n_completed_splits_skipped_in_estimate": 2,
        "n_calls_skipped_existing_checkpoints": 169,
        "estimate_kind": "session_remaining_work",
    }), encoding="utf-8")
    (out_dir / "llm_token_cost_report.json").write_text(json.dumps({
        "calls": 755,
        "prompt_tokens": 2969665,
        "completion_tokens": 8040058,
        "total_tokens": 11009723,
        "estimated_cost_usd": 3.31048533,
        "cached_tokens": 1350528,
        "uncached_prompt_tokens": 1619137,
        "reasoning_tokens": 6475424,
        "estimated_cache_savings_usd": 0.05402112,
        "cost_type": "post_run_actual",
    }), encoding="utf-8")

    paths = generate_all_reports(str(out_dir), force=True)
    assert Path(paths["results_html"]).is_file()
    results_html = (out_dir / "automated_results_report.html").read_text(encoding="utf-8")
    assert "Cost estimate vs actual" in results_html
    assert "756.0" in results_html or "756.0000" in results_html
    assert "282.0" not in results_html
    assert "cost_estimate_vs_actual_usd.png" in results_html
    assert Path(paths["interpretability_html"]).is_file()
    assert Path(paths["missed_case_html"]).is_file()


def test_flowchart_png_exists():
    png = PIPELINE_ROOT / "docs" / "approach2_pipeline_flowchart.png"
    assert png.is_file()
    assert png.stat().st_size > 1000
