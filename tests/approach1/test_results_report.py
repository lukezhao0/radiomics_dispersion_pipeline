"""HTML results report generation tests."""

from __future__ import annotations

import json
import os

import pandas as pd

from approach1.evaluation.results_report import (
    REPORT_FILENAME,
    build_approach1_results_html,
    discover_config_dirs,
)


def _write_minimal_config(tmp_path, shotset: str, modality: str) -> str:
    cfg_dir = tmp_path / shotset / modality
    cfg_dir.mkdir(parents=True)
    run_config = {
        "shotset_name": shotset,
        "high_rows": [0, 2],
        "low_rows": [101, 102],
        "training_rows": [0, 2, 101, 102],
        "modality": modality,
        "n_test_cases": 2,
        "n_skipped_missing_mri": 0,
    }
    (cfg_dir / "run_config.json").write_text(json.dumps(run_config), encoding="utf-8")
    metrics = {
        "n_rows": 2,
        "dispersion_regression": {"mae": 10.5, "rmse": 12.0, "spearman_rho": 0.4},
        "dispersion_high_low": {"accuracy": 0.5, "f1": 0.5, "confusion_matrix": [[1, 0], [1, 0]]},
        "relapse_label": {"accuracy": 1.0, "f1": 1.0, "confusion_matrix": [[2, 0], [0, 0]]},
        "needle_retrieval": {"single_token_rate": 1.0, "single_token_failures": 0},
    }
    (cfg_dir / "evaluation_metrics_summary.json").write_text(json.dumps(metrics), encoding="utf-8")
    (cfg_dir / "evaluation_metrics_from_csv.txt").write_text("Dispersion score (regression):\n  N_used = 2", encoding="utf-8")
    run_config["apriori_cost"] = {
        "n_calls": 2,
        "prompt_tokens_estimated_total": 10000,
        "cached_prompt_tokens_estimated": 2000,
        "uncached_prompt_tokens_estimated": 8000,
        "completion_tokens_upper_estimated": 32000,
        "estimated_no_cache_cost_usd_upper": 0.5,
        "estimated_cache_adjusted_cost_usd_upper": 0.45,
        "estimated_cache_savings_usd": 0.05,
    }
    (cfg_dir / "run_config.json").write_text(json.dumps(run_config), encoding="utf-8")
    token_cost = {
        "cumulative": {
            "calls": 2,
            "prompt_tokens": 9500,
            "cached_tokens": 1800,
            "uncached_prompt_tokens": 7700,
            "completion_tokens": 800,
            "reasoning_tokens": 0,
            "total_tokens": 10300,
            "estimated_cost_usd": 0.12,
            "estimated_cache_savings_usd": 0.004,
        }
    }
    (cfg_dir / "token_cost_report.json").write_text(json.dumps(token_cost), encoding="utf-8")
    pd.DataFrame([
        {
            "case_id": "SYNTH_001",
            "row_index": 3,
            "dispersion_true": 120.0,
            "dispersion_score_pred": 110.0,
            "dispersion_high_low_pred": 1,
            "relapse_true": 0,
            "relapse_pred": 0,
            "retrieval_token_exact_match": 1,
            "reasoning_summary": "Example reasoning.",
        }
    ]).to_csv(cfg_dir / "predictions_testing_cases.csv", index=False)
    return str(cfg_dir)


def test_discover_config_dirs(tmp_path) -> None:
    _write_minimal_config(tmp_path, "shotset_a", "mri_only")
    found = discover_config_dirs(str(tmp_path))
    assert len(found) == 1
    assert found[0][0] == "shotset_a"
    assert found[0][1] == "mri_only"


def test_build_approach1_results_html(tmp_path) -> None:
    _write_minimal_config(tmp_path, "shotset_a", "mri_only")
    pd.DataFrame([{
        "shotset_name": "shotset_a",
        "modality": "mri_only",
        "n_predictions": 2,
        "dispersion_mae": 10.5,
    }]).to_csv(tmp_path / "all_tiers_metrics_summary.csv", index=False)

    out = build_approach1_results_html(str(tmp_path))
    assert out.endswith(REPORT_FILENAME)
    assert os.path.isfile(out)
    html = open(out, encoding="utf-8").read()
    assert "Approach 1 Results Review" in html
    assert "shotset_a" in html
    assert "Dispersion MAE" in html
    assert "Metric glossary" in html
    assert "Cost estimate vs actual" in html
    assert "cost_estimate_vs_actual_tokens.png" in html
