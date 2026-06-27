"""Tests for shared cost estimate vs actual comparison helpers."""

from __future__ import annotations

import json
import os

import pandas as pd
import pytest

from common.cost_comparison import (
    APPROACH1_APRIORI,
    APPROACH2_APRIORI,
    APRIORI_INITIAL_FILENAME,
    aggregate_apriori_from_run_configs,
    build_cost_comparison_summary_df,
    extract_actual_cumulative,
    generate_cost_comparison_plots,
    is_partial_resume_apriori_estimate,
    load_approach2_apriori_for_comparison,
    normalize_apriori_estimate,
)


def test_normalize_apriori_approach1() -> None:
    raw = {
        "n_calls": 10,
        "prompt_tokens_estimated_total": 1000,
        "cached_prompt_tokens_estimated": 400,
        "uncached_prompt_tokens_estimated": 600,
        "completion_tokens_upper_estimated": 160000,
        "estimated_no_cache_cost_usd_upper": 1.0,
        "estimated_cache_adjusted_cost_usd_upper": 0.8,
        "estimated_cache_savings_usd": 0.2,
    }
    norm = normalize_apriori_estimate(raw, flavor=APPROACH1_APRIORI)
    assert norm["n_calls"] == 10
    assert norm["prompt_tokens"] == 1000
    assert norm["completion_tokens_cap"] == 160000
    assert norm["cache_aware_cost_usd"] == pytest.approx(0.8)


def test_normalize_apriori_approach2() -> None:
    raw = {
        "n_calls": 5,
        "estimated_prompt_tokens": 500,
        "estimated_completion_cap_tokens": 80000,
        "no_cache_estimated_cost_usd": 0.5,
        "cache_aware_estimated_cost_usd": 0.4,
        "cache_aware_estimated_cached_tokens": 100,
        "cache_aware_estimated_cache_savings_usd": 0.05,
    }
    norm = normalize_apriori_estimate(raw, flavor=APPROACH2_APRIORI)
    assert norm["prompt_tokens"] == 500
    assert norm["cache_aware_cost_usd"] == pytest.approx(0.4)


def test_extract_actual_cumulative_nested_and_flat() -> None:
    nested = {"cumulative": {"calls": 3, "prompt_tokens": 100, "estimated_cost_usd": 0.12}}
    flat = {"calls": 2, "prompt_tokens": 50, "estimated_cost_usd": 0.05}
    assert extract_actual_cumulative(nested)["calls"] == 3
    assert extract_actual_cumulative(flat)["prompt_tokens"] == 50


def test_build_cost_comparison_summary_includes_delta() -> None:
    apriori = normalize_apriori_estimate(
        {"n_calls": 1, "prompt_tokens_estimated_total": 100, "completion_tokens_upper_estimated": 1000,
         "estimated_no_cache_cost_usd_upper": 1.0, "estimated_cache_adjusted_cost_usd_upper": 0.9,
         "estimated_cache_savings_usd": 0.1, "cached_prompt_tokens_estimated": 0,
         "uncached_prompt_tokens_estimated": 100},
        flavor=APPROACH1_APRIORI,
    )
    actual = {"calls": 1, "prompt_tokens": 95, "completion_tokens": 200, "total_tokens": 295,
              "estimated_cost_usd": 0.5, "cached_tokens": 0, "uncached_prompt_tokens": 95,
              "reasoning_tokens": 0, "estimated_cache_savings_usd": 0.0}
    df = build_cost_comparison_summary_df(apriori, actual)
    assert "Estimated cost (USD)" in df["metric"].values
    assert df.loc[df["metric"] == "Actual minus cache-aware estimate (USD)", "actual"].iloc[0] == pytest.approx(-0.4)


def test_aggregate_apriori_from_run_configs() -> None:
    payloads = [
        {"apriori_cost": {"n_calls": 2, "prompt_tokens_estimated_total": 100,
                          "cached_prompt_tokens_estimated": 10, "uncached_prompt_tokens_estimated": 90,
                          "completion_tokens_upper_estimated": 2000,
                          "estimated_no_cache_cost_usd_upper": 0.1,
                          "estimated_cache_adjusted_cost_usd_upper": 0.09,
                          "estimated_cache_savings_usd": 0.01}},
        {"apriori_cost": {"n_calls": 3, "prompt_tokens_estimated_total": 150,
                          "cached_prompt_tokens_estimated": 20, "uncached_prompt_tokens_estimated": 130,
                          "completion_tokens_upper_estimated": 3000,
                          "estimated_no_cache_cost_usd_upper": 0.2,
                          "estimated_cache_adjusted_cost_usd_upper": 0.18,
                          "estimated_cache_savings_usd": 0.02}},
    ]
    total = aggregate_apriori_from_run_configs(payloads)
    assert total["n_calls"] == 5
    assert total["prompt_tokens_estimated_total"] == 250


def test_generate_cost_comparison_plots(tmp_path) -> None:
    apriori = normalize_apriori_estimate(
        {"n_calls": 1, "prompt_tokens_estimated_total": 1000, "completion_tokens_upper_estimated": 5000,
         "estimated_no_cache_cost_usd_upper": 1.0, "estimated_cache_adjusted_cost_usd_upper": 0.8,
         "estimated_cache_savings_usd": 0.2, "cached_prompt_tokens_estimated": 200,
         "uncached_prompt_tokens_estimated": 800},
        flavor=APPROACH1_APRIORI,
    )
    actual = extract_actual_cumulative({
        "calls": 1, "prompt_tokens": 900, "completion_tokens": 400, "total_tokens": 1300,
        "estimated_cost_usd": 0.3, "cached_tokens": 150, "uncached_prompt_tokens": 750,
        "reasoning_tokens": 0, "estimated_cache_savings_usd": 0.01,
    })
    paths = generate_cost_comparison_plots(str(tmp_path), apriori, actual)
    assert len(paths) >= 2
    for path in paths:
        assert os.path.isfile(path)


def test_is_partial_resume_apriori_estimate() -> None:
    assert not is_partial_resume_apriori_estimate({"n_calls": 10})
    assert is_partial_resume_apriori_estimate({"n_calls": 10, "n_completed_splits_skipped_in_estimate": 1})
    assert is_partial_resume_apriori_estimate({"n_calls": 10, "n_calls_skipped_existing_checkpoints": 3})


def test_load_approach2_apriori_prefers_initial(tmp_path) -> None:
    initial = {
        "n_calls": 756,
        "estimated_prompt_tokens": 100,
        "estimated_completion_cap_tokens": 1000,
        "no_cache_estimated_cost_usd": 4.0,
        "cache_aware_estimated_cost_usd": 3.5,
        "cache_aware_estimated_cached_tokens": 10,
        "cache_aware_estimated_cache_savings_usd": 0.5,
    }
    session = {
        "n_calls": 282,
        "estimated_prompt_tokens": 50,
        "estimated_completion_cap_tokens": 500,
        "no_cache_estimated_cost_usd": 1.0,
        "cache_aware_estimated_cost_usd": 0.8,
        "cache_aware_estimated_cached_tokens": 5,
        "cache_aware_estimated_cache_savings_usd": 0.2,
        "n_completed_splits_skipped_in_estimate": 2,
    }
    (tmp_path / APRIORI_INITIAL_FILENAME).write_text(json.dumps(initial), encoding="utf-8")
    (tmp_path / "llm_cost_estimate_apriori.json").write_text(json.dumps(session), encoding="utf-8")
    loaded = load_approach2_apriori_for_comparison(str(tmp_path))
    assert loaded["n_calls"] == 756
    assert normalize_apriori_estimate(loaded, flavor=APPROACH2_APRIORI)["cache_aware_cost_usd"] == pytest.approx(3.5)
