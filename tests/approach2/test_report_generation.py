"""Tests for Approach 2 standalone report generation."""

from __future__ import annotations

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

    paths = generate_all_reports(str(out_dir), force=True)
    assert Path(paths["results_html"]).is_file()
    assert Path(paths["interpretability_html"]).is_file()
    assert Path(paths["missed_case_html"]).is_file()


def test_flowchart_png_exists():
    png = PIPELINE_ROOT / "docs" / "approach2_pipeline_flowchart.png"
    assert png.is_file()
    assert png.stat().st_size > 1000
