"""Tests for Approach 1 bootstrap confidence intervals."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from approach1.evaluation.bootstrap import (
    BOOTSTRAP_CSV_FILENAME,
    BOOTSTRAP_JSON_FILENAME,
    compute_and_save_bootstrap_cis,
    compute_approach1_bootstrap_cis,
)
from approach1.evaluation.metrics import prepare_predictions_for_eval

PIPELINE_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = PIPELINE_ROOT / "scripts" / "bootstrap_approach1_metrics.py"
EXISTING_CSV = (
    PIPELINE_ROOT.parent
    / "sabcs"
    / "securegpt_dispersion_approach1_pipeline_062726"
    / "shotset_high_0_2_low_101_102"
    / "mri_plus_pathology"
    / "predictions_testing_cases.csv"
)


def _classification_df(n: int = 20, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, 2, size=n)
    y_pred = y_true.copy()
    y_pred[rng.choice(n, size=max(1, n // 5), replace=False)] = 1 - y_pred[rng.choice(n, size=max(1, n // 5), replace=False)]
    scores = y_true.astype(float) + rng.normal(0, 0.1, size=n)
    dispersion_true = rng.uniform(20, 200, size=n)
    dispersion_score = dispersion_true + rng.normal(0, 10, size=n)
    return pd.DataFrame(
        {
            "dispersion_true": dispersion_true,
            "dispersion_true_high_low": (dispersion_true >= 85).astype(int),
            "dispersion_score_pred": dispersion_score,
            "dispersion_high_low_pred": (dispersion_score >= 85).astype(int),
            "relapse_true": y_true,
            "relapse_pred": y_pred,
        }
    )


def test_bootstrap_returns_all_metric_tasks(tmp_path):
    df = prepare_predictions_for_eval(_classification_df())
    results = compute_approach1_bootstrap_cis(df, n_bootstrap=50, random_seed=123)
    tasks = {r.task for r in results}
    assert tasks == {"continuous_dispersion", "high_low_dispersion", "relapse_prediction"}
    metrics = {(r.task, r.metric) for r in results}
    assert ("continuous_dispersion", "spearman_rho") in metrics
    assert ("high_low_dispersion", "auroc") in metrics
    assert ("relapse_prediction", "sensitivity") in metrics
    for r in results:
        assert r.n_bootstrap_requested == 50
        assert 0 <= r.n_bootstrap_valid <= 50


def test_degenerate_bootstrap_single_class_does_not_crash():
    df = prepare_predictions_for_eval(
        pd.DataFrame(
            {
                "dispersion_true": [100.0, 90.0, 95.0],
                "dispersion_score_pred": [95.0, 88.0, 92.0],
                "dispersion_high_low_pred": [1, 1, 1],
                "relapse_true": [1, 1, 1],
                "relapse_pred": [1, 0, 1],
            }
        )
    )
    results = compute_approach1_bootstrap_cis(df, n_bootstrap=100, random_seed=7)
    auroc_rows = [r for r in results if r.metric == "auroc"]
    assert len(auroc_rows) == 2
    for row in auroc_rows:
        assert row.point_estimate is None or np.isnan(row.point_estimate) or row.point_estimate >= 0
        assert row.n_bootstrap_valid < row.n_bootstrap_requested or "Skipped" in row.notes


def test_save_bootstrap_output_files(tmp_path):
    df = prepare_predictions_for_eval(_classification_df(n=30, seed=1))
    summary = compute_and_save_bootstrap_cis(df, str(tmp_path), n_bootstrap=25, random_seed=99)
    csv_path = tmp_path / BOOTSTRAP_CSV_FILENAME
    json_path = tmp_path / BOOTSTRAP_JSON_FILENAME
    assert csv_path.is_file()
    assert json_path.is_file()
    out_df = pd.read_csv(csv_path)
    assert {"task", "metric", "point_estimate", "ci_lower", "ci_upper"}.issubset(out_df.columns)
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert "metadata" in payload
    assert len(payload["metrics"]) == summary["n_metrics"]


@pytest.mark.skipif(not EXISTING_CSV.is_file(), reason="Existing pipeline predictions CSV not present")
def test_offline_script_on_existing_csv(tmp_path):
    out_dir = tmp_path / "bootstrap_out"
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            str(EXISTING_CSV),
            "--outdir",
            str(out_dir),
            "--bootstrap-n",
            "100",
            "--bootstrap-seed",
            "42",
        ],
        cwd=str(PIPELINE_ROOT),
        check=True,
        capture_output=True,
        text=True,
    )
    assert "[BOOTSTRAP]" in proc.stdout
    csv_path = out_dir / BOOTSTRAP_CSV_FILENAME
    assert csv_path.is_file()
    df = pd.read_csv(csv_path)
    assert len(df) == 12
    assert len(pd.read_csv(EXISTING_CSV)) == 82


def test_relapse_metrics_use_binary_pred_column():
    df = prepare_predictions_for_eval(_classification_df(n=40, seed=3))
    results = compute_approach1_bootstrap_cis(df, n_bootstrap=30, random_seed=5)
    relapse_auroc = next(r for r in results if r.task == "relapse_prediction" and r.metric == "auroc")
    assert np.isfinite(relapse_auroc.point_estimate)
