"""Regression tests for CLI helper wiring after modularization."""

from __future__ import annotations

import argparse
from types import SimpleNamespace

import numpy as np
import pandas as pd

from approach2.cli import estimate_nested_pipeline_llm_cost
from approach2.eval_data import get_target_frame
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
    args = SimpleNamespace(
        out_dir=str(tmp_path),
        resume=True,
        skip_completed_splits=True,
        modalities=["mri", "path", "combined"],
        enable_pathology_calibration=True,
        enable_teacher_student=True,
    )
    estimate = estimate_nested_pipeline_llm_cost(raw_df, target_df, outer_splits, args)
    assert estimate["n_outer_splits"] == len(outer_splits)
    assert estimate["n_calls"] > 0
    assert estimate["estimated_prompt_tokens"] > 0
