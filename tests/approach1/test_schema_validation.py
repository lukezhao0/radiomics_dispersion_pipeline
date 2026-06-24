"""Schema validation tests."""

from __future__ import annotations

from approach1.schema.prediction import extract_json_from_text, validate_prediction_obj


def _valid_obj(token: str) -> dict:
    return {
        "case_id": "SYNTH_001",
        "dispersion_score_pred": 90.0,
        "dispersion_high_low_pred": 1,
        "relapse_pred": 0,
        "key_evidence": ["scattered residual carcinoma"],
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


def test_validate_prediction_obj_accepts_valid():
    token = "CTXCHK_TEST"
    ok, msg = validate_prediction_obj(_valid_obj(token), "SYNTH_001", token)
    assert ok, msg


def test_validate_prediction_obj_rejects_score_out_of_range():
    token = "CTXCHK_TEST"
    obj = _valid_obj(token)
    obj["dispersion_score_pred"] = 500.0
    ok, msg = validate_prediction_obj(obj, "SYNTH_001", token)
    assert not ok
    assert "out of range" in msg


def test_validate_prediction_obj_rejects_inconsistent_high_low():
    token = "CTXCHK_TEST"
    obj = _valid_obj(token)
    obj["dispersion_score_pred"] = 50.0
    obj["dispersion_high_low_pred"] = 1
    ok, msg = validate_prediction_obj(obj, "SYNTH_001", token)
    assert not ok
    assert "inconsistent" in msg


def test_validate_prediction_obj_rejects_wrong_token():
    token = "CTXCHK_TEST"
    obj = _valid_obj(token)
    obj["retrieval_check_token_returned"] = "WRONG"
    obj["retrieval_check_correct"] = True
    ok, msg = validate_prediction_obj(obj, "SYNTH_001", token)
    assert not ok


def test_extract_json_from_text_embedded():
    raw = 'Some preamble {"dispersion_score_pred": 10, "x": 1} trailing'
    # minimal won't parse full schema but should extract JSON
    result = extract_json_from_text('{"a": 1}')
    assert result == {"a": 1}
