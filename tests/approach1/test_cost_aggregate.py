"""Tests for approach1 pipeline-level cost aggregation."""

from __future__ import annotations

import json

import pytest

from approach1.api.cost import (
    CONFIG_COST_FILENAME,
    aggregate_pipeline_cost_report,
    empty_cost_tracker,
    save_pipeline_cost_report,
)
from approach1.models import Case, RunConfig


def _make_rc(tmp_path, modality: str, calls: int) -> RunConfig:
    run_out_dir = tmp_path / modality
    run_out_dir.mkdir(parents=True)
    cumulative = empty_cost_tracker()
    cumulative["calls"] = calls
    cumulative["estimated_cost_usd"] = float(calls) / 100.0
    payload = {
        "cumulative": cumulative,
        "session": cumulative,
        "prior_sessions": empty_cost_tracker(),
    }
    with open(run_out_dir / CONFIG_COST_FILENAME, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    return RunConfig(
        shotset_name="shot_a",
        high_rows=[0, 1],
        low_rows=[2, 3],
        training_rows=[0, 1, 2, 3],
        modality=modality,
        run_out_dir=str(run_out_dir),
        training_block="BLOCK",
        test_cases_with_idxs=[(5, Case("c1", "m", "p", "left", 1.0, 0))],
        skipped_missing_mri=[],
        apriori_cost={"n_calls": 1},
    )


def test_aggregate_pipeline_cost_report_sums_configs(tmp_path) -> None:
    rc1 = _make_rc(tmp_path, "mri_only", 1)
    rc2 = _make_rc(tmp_path, "pathology_only", 2)
    merged = aggregate_pipeline_cost_report([rc1, rc2])
    assert merged["calls"] == 3
    assert merged["estimated_cost_usd"] == pytest.approx(0.03)


def test_save_pipeline_cost_report_writes_root_file(tmp_path) -> None:
    rc = _make_rc(tmp_path, "mri_only", 4)
    path = save_pipeline_cost_report(str(tmp_path), [rc])
    payload = json.loads(open(path, encoding="utf-8").read())
    assert payload["cumulative"]["calls"] == 4
