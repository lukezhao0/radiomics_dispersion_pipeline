"""Regression tests for CLI helper wiring after modularization."""

from __future__ import annotations

import argparse
import json
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from approach2.cli import estimate_nested_pipeline_llm_cost
from approach2.eval_data import get_target_frame
from approach2.extraction.config import configure_llm
from approach2.extraction.pipeline import case_extraction_checkpoint_available
from approach2.splits import build_outer_splits


def _synthetic_cases_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "case_id": [f"case_{i}" for i in range(6)],
            "index_side": ["left"] * 6,
            "preop_MRI_text": [f"residual enhancement focus {i}" for i in range(6)],
            "path_report_text": [f"residual invasive carcinoma {i}" for i in range(6)],
            "dispersion_invasive_DCIS_geographic": [90, 40, 85, 30, 88, 55],
            "relapse": [1, 0, 1, 0, 1, 0],
        }
    )


def _estimate_args(tmp_path, **kwargs):
    defaults = dict(
        out_dir=str(tmp_path),
        resume=True,
        skip_completed_splits=True,
        force_reextract=False,
        modalities=["mri", "path", "combined"],
        enable_pathology_calibration=True,
        enable_teacher_student=True,
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def test_estimate_nested_pipeline_llm_cost_honor_resume_skips_false_counts_all(tmp_path):
    raw_df = _synthetic_cases_df()
    target_df = get_target_frame(raw_df)
    outer_splits = build_outer_splits(
        y_binary=target_df["dispersion_true_high_low"].astype(int).values,
        scheme="stratified_kfold",
        random_seed=17,
        n_repeats=1,
        test_frac=0.2,
        n_folds=3,
    )
    os.environ.setdefault("SANDBOX_API_KEY", "test-key")
    configure_llm("gpt-5-nano")
    args = _estimate_args(tmp_path)
    resumed = estimate_nested_pipeline_llm_cost(raw_df, target_df, outer_splits, args)
    full = estimate_nested_pipeline_llm_cost(
        raw_df, target_df, outer_splits, args, honor_resume_skips=False
    )
    assert full["n_calls"] >= resumed["n_calls"]
    assert full["n_completed_splits_skipped_in_estimate"] == 0
    assert full.get("n_calls_skipped_existing_checkpoints", 0) == 0


def test_estimate_nested_pipeline_llm_cost_smoke(tmp_path):
    raw_df = _synthetic_cases_df()
    target_df = get_target_frame(raw_df)
    outer_splits = build_outer_splits(
        y_binary=target_df["dispersion_true_high_low"].astype(int).values,
        scheme="stratified_kfold",
        random_seed=17,
        n_repeats=1,
        test_frac=0.2,
        n_folds=3,
    )
    os.environ.setdefault("SANDBOX_API_KEY", "test-key")
    configure_llm("gpt-5-nano")
    estimate = estimate_nested_pipeline_llm_cost(
        raw_df, target_df, outer_splits, _estimate_args(tmp_path)
    )
    assert estimate["n_outer_splits"] == len(outer_splits)
    assert estimate["n_calls"] > 0
    assert estimate["estimated_prompt_tokens"] > 0
    assert estimate["model"] == "gpt-5-nano"
    assert estimate["price_per_1M_input_tokens"] == pytest.approx(0.05)


def test_estimate_nested_pipeline_llm_cost_uses_selected_model_pricing(tmp_path, monkeypatch):
    monkeypatch.setenv("SANDBOX_API_KEY", "sandbox-key")
    monkeypatch.setenv("NEW_SECUREGPT_API_KEY", "securegpt-key")

    raw_df = _synthetic_cases_df()
    target_df = get_target_frame(raw_df)
    outer_splits = build_outer_splits(
        y_binary=target_df["dispersion_true_high_low"].astype(int).values,
        scheme="stratified_kfold",
        random_seed=17,
        n_repeats=1,
        test_frac=0.2,
        n_folds=3,
    )

    configure_llm("gpt-5")
    gpt5 = estimate_nested_pipeline_llm_cost(
        raw_df, target_df, outer_splits, _estimate_args(tmp_path)
    )
    configure_llm("gpt-5-nano")
    nano = estimate_nested_pipeline_llm_cost(
        raw_df, target_df, outer_splits, _estimate_args(tmp_path)
    )

    assert gpt5["model"] == "gpt-5"
    assert nano["model"] == "gpt-5-nano"
    assert gpt5["no_cache_estimated_cost_usd"] > nano["no_cache_estimated_cost_usd"]
    assert (
        gpt5["no_cache_estimated_cost_usd"] / nano["no_cache_estimated_cost_usd"]
        == pytest.approx(25.0, rel=1e-6)
    )


def test_estimate_nested_pipeline_llm_cost_skips_existing_case_checkpoints(tmp_path):
    raw_df = _synthetic_cases_df()
    target_df = get_target_frame(raw_df)
    outer_splits = build_outer_splits(
        y_binary=target_df["dispersion_true_high_low"].astype(int).values,
        scheme="stratified_kfold",
        random_seed=17,
        n_repeats=1,
        test_frac=0.2,
        n_folds=3,
    )
    os.environ.setdefault("SANDBOX_API_KEY", "test-key")
    configure_llm("gpt-5-nano")

    split_dir = tmp_path / "outer_splits" / "outer_split_001" / "mri"
    split_dir.mkdir(parents=True)
    row_index = int(target_df.iloc[outer_splits[0][0][0]]["row_index"])
    case_id = str(target_df.iloc[outer_splits[0][0][0]]["case_id"])
    checkpoint_root = split_dir / "_case_checkpoints" / "mri" / "outer_split_001" / "train"
    checkpoint_root.mkdir(parents=True)
    checkpoint_path = checkpoint_root / f"row_{row_index:06d}__case_{case_id}__mode_mri.json"
    checkpoint_path.write_text(
        json.dumps(
            {
                "case_id": case_id,
                "row_index": row_index,
                "report_mode": "mri",
                "outer_split_id": "outer_split_001",
                "outer_split_role": "train",
            }
        ),
        encoding="utf-8",
    )
    assert case_extraction_checkpoint_available(
        checkpoint_dir=str(split_dir),
        row_index=row_index,
        case_id=case_id,
        report_mode="mri",
        split_id="outer_split_001",
        split_role="train",
    )

    fresh = estimate_nested_pipeline_llm_cost(
        raw_df, target_df, outer_splits, _estimate_args(tmp_path)
    )
    forced = estimate_nested_pipeline_llm_cost(
        raw_df,
        target_df,
        outer_splits,
        _estimate_args(tmp_path, force_reextract=True),
    )

    assert fresh["n_calls_skipped_existing_checkpoints"] >= 1
    assert fresh["n_calls"] < forced["n_calls"]
