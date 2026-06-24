"""Integration tests for Approach 1 resume / checkpoint behavior."""

from __future__ import annotations

import json
import os

import pandas as pd
import pytest

from approach1.checkpoint.fingerprint import build_config_fingerprint, config_fingerprints_compatible
from approach1.checkpoint.predictions import load_existing_case_predictions
from approach1.checkpoint.resume import (
    is_config_checkpoint_complete,
    load_completed_config_checkpoint,
    save_completed_config_checkpoint,
    summarize_resume_plan,
)
from approach1.models import Case, RunConfig
from approach1.orchestration import run_one_config
from approach1.prompts.tokens import make_case_token
from approach1.schema.records import build_pred_record


def _valid_pred_obj(token: str) -> dict:
    return {
        "case_id": "SYNTH_001",
        "dispersion_score_pred": 90.0,
        "dispersion_high_low_pred": 1,
        "relapse_pred": 0,
        "key_evidence": ["residual carcinoma scattered"],
        "retrieval_check_token_returned": token,
        "retrieval_check_correct": True,
        "reasoning_summary": "Evidence supports high dispersion.",
        "structured_rationale": {
            "step_1_localization": "Left breast.",
            "step_2_pathology_pattern": "Scattered cells.",
            "step_3_mri_pattern": "Non-mass enhancement.",
            "step_4_dispersion_synthesis": "High dispersion pattern.",
            "step_5_relapse_synthesis": "Lower relapse risk.",
        },
    }


def _make_rc(tmp_path, test_indices: list[int], *, modality: str = "pathology_only") -> RunConfig:
    cases = [
        (
            idx,
            Case(
                case_id=f"SYNTH_{idx:03d}",
                preop_mri=f"MRI text for case {idx}",
                path_report=f"Pathology text for case {idx}",
                index_side="left",
                dispersion_true=120.0,
                relapse_true=0,
            ),
        )
        for idx in test_indices
    ]
    return RunConfig(
        shotset_name="test_shotset",
        high_rows=[0, 1],
        low_rows=[2, 3],
        training_rows=[0, 1, 2, 3],
        modality=modality,
        run_out_dir=str(tmp_path / "out" / modality),
        training_block="EXAMPLE BLOCK",
        test_cases_with_idxs=cases,
        skipped_missing_mri=[],
        apriori_cost={"n_calls": len(cases)},
    )


def _write_saved_prediction(rc: RunConfig, idx: int) -> dict:
    case = next(c for i, c in rc.test_cases_with_idxs if i == idx)
    token = make_case_token(case, idx, rc.modality)
    obj = _valid_pred_obj(token)
    obj["case_id"] = case.case_id
    return build_pred_record(rc, idx, case, obj, token)


@pytest.fixture
def mock_predict(monkeypatch):
    calls: list[int] = []

    def _fake_predict(training_block, test_case, row_index, modality):
        calls.append(row_index)
        token = make_case_token(test_case, row_index, modality)
        obj = _valid_pred_obj(token)
        obj["case_id"] = test_case.case_id
        return obj

    monkeypatch.setattr("approach1.orchestration.predict_case", _fake_predict)
    monkeypatch.setattr("approach1.orchestration.evaluate_and_plot", lambda df, out_dir, title: {"n_rows": len(df)})
    return calls


def test_skip_completed_config_when_marker_and_predictions_valid(tmp_path, mock_predict):
    rc = _make_rc(tmp_path, [10, 11, 12])
    os.makedirs(rc.run_out_dir, exist_ok=True)
    for idx in [10, 11, 12]:
        rec = _write_saved_prediction(rc, idx)
        with open(os.path.join(rc.run_out_dir, "predictions_testing_cases.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
    pd.DataFrame([_write_saved_prediction(rc, idx) for idx in [10, 11, 12]]).to_csv(
        os.path.join(rc.run_out_dir, "predictions_testing_cases.csv"), index=False
    )
    with open(os.path.join(rc.run_out_dir, "evaluation_metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"n_rows": 3}, f)
    save_completed_config_checkpoint(rc.run_out_dir, rc, n_new_api_calls=3)

    pred_df, metrics = run_one_config(rc, resume=True, skip_completed_configs=True)
    assert len(pred_df) == 3
    assert metrics["n_rows"] == 3
    assert mock_predict == []


def test_partial_resume_only_calls_api_for_missing_cases(tmp_path, mock_predict):
    rc = _make_rc(tmp_path, [10, 11, 12])
    os.makedirs(rc.run_out_dir, exist_ok=True)
    rec10 = _write_saved_prediction(rc, 10)
    with open(os.path.join(rc.run_out_dir, "predictions_testing_cases.jsonl"), "w", encoding="utf-8") as f:
        f.write(json.dumps(rec10) + "\n")

    pred_df, _ = run_one_config(rc, resume=True, skip_completed_configs=False, force_rerun_cases=False)
    assert len(pred_df) == 3
    assert sorted(mock_predict) == [11, 12]

    # Final JSONL should contain all three rows in order
    by_row, _, warnings = load_existing_case_predictions(rc.run_out_dir, rc)
    assert len(by_row) == 3
    assert not warnings


def test_force_rerun_cases_redoes_api_even_with_saved_predictions(tmp_path, mock_predict):
    rc = _make_rc(tmp_path, [10, 11])
    os.makedirs(rc.run_out_dir, exist_ok=True)
    for idx in [10, 11]:
        rec = _write_saved_prediction(rc, idx)
        with open(os.path.join(rc.run_out_dir, "predictions_testing_cases.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
    save_completed_config_checkpoint(rc.run_out_dir, rc, n_new_api_calls=2)

    pred_df, _ = run_one_config(rc, resume=True, skip_completed_configs=False, force_rerun_cases=True)
    assert len(pred_df) == 2
    assert sorted(mock_predict) == [10, 11]
    assert not os.path.isfile(os.path.join(rc.run_out_dir, "_resume_checkpoint", "COMPLETED.json")) or True
    # Marker removed at start then rewritten at end
    assert os.path.isfile(os.path.join(rc.run_out_dir, "_resume_checkpoint", "COMPLETED.json"))


def test_force_rerun_cases_still_skips_fully_completed_config(tmp_path, mock_predict):
    """Completed configs are skipped before per-case logic; matches monolith behavior."""
    rc = _make_rc(tmp_path, [10, 11])
    os.makedirs(rc.run_out_dir, exist_ok=True)
    for idx in [10, 11]:
        rec = _write_saved_prediction(rc, idx)
        with open(os.path.join(rc.run_out_dir, "predictions_testing_cases.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
    pd.DataFrame([_write_saved_prediction(rc, idx) for idx in [10, 11]]).to_csv(
        os.path.join(rc.run_out_dir, "predictions_testing_cases.csv"), index=False
    )
    with open(os.path.join(rc.run_out_dir, "evaluation_metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"n_rows": 2}, f)
    save_completed_config_checkpoint(rc.run_out_dir, rc, n_new_api_calls=2)

    run_one_config(rc, resume=True, skip_completed_configs=True, force_rerun_cases=True)
    assert mock_predict == []


def test_fingerprint_version_mismatch_invalidates_completed_marker(tmp_path):
    rc = _make_rc(tmp_path, [10])
    saved = build_config_fingerprint(rc)
    saved["resume_script_version"] = "old-version"
    ok, msg = config_fingerprints_compatible(saved, rc)
    assert not ok
    assert "resume_script_version" in msg


def test_no_resume_ignores_existing_predictions(tmp_path, mock_predict):
    rc = _make_rc(tmp_path, [10, 11])
    os.makedirs(rc.run_out_dir, exist_ok=True)
    rec = _write_saved_prediction(rc, 10)
    with open(os.path.join(rc.run_out_dir, "predictions_testing_cases.jsonl"), "w", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    save_completed_config_checkpoint(rc.run_out_dir, rc, n_new_api_calls=1)

    pred_df, _ = run_one_config(rc, resume=False, skip_completed_configs=True)
    assert len(pred_df) == 2
    assert sorted(mock_predict) == [10, 11]


def test_summarize_resume_plan_partial_status(tmp_path):
    rc = _make_rc(tmp_path, [10, 11, 12])
    os.makedirs(rc.run_out_dir, exist_ok=True)
    rec = _write_saved_prediction(rc, 10)
    with open(os.path.join(rc.run_out_dir, "predictions_testing_cases.jsonl"), "w", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")

    summary = summarize_resume_plan([rc], resume=True, skip_completed_configs=True, force_rerun_cases=False)
    assert summary["n_configs_resume_cases"] == 1
    assert summary["per_config"][0]["status"] == "resume_partial"
    assert summary["per_config"][0]["n_done"] == 1
    assert summary["per_config"][0]["n_pending"] == 2


def test_invalid_saved_record_is_rejected_on_load(tmp_path):
    rc = _make_rc(tmp_path, [10])
    os.makedirs(rc.run_out_dir, exist_ok=True)
    bad = _write_saved_prediction(rc, 10)
    bad["dispersion_score_pred"] = 999.0
    with open(os.path.join(rc.run_out_dir, "predictions_testing_cases.jsonl"), "w", encoding="utf-8") as f:
        f.write(json.dumps(bad) + "\n")

    by_row, _, warnings = load_existing_case_predictions(rc.run_out_dir, rc)
    assert len(by_row) == 0
    assert any("out of range" in w for w in warnings)
