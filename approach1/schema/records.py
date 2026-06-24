"""Saved prediction record construction and validation."""

from __future__ import annotations

import ast
import json
from typing import Any, Dict, List, Tuple

import numpy as np

from ..config import DISPERSION_HIGH_THRESHOLD
from ..models import Case, RunConfig
from ..prompts.tokens import make_case_token
from ..text_utils import has_report_text
from .prediction import validate_prediction_obj


def parse_jsonish_list(x: Any) -> List[str]:
    if isinstance(x, list):
        return [str(v) for v in x]
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []
    s = str(x).strip()
    if not s:
        return []
    try:
        val = json.loads(s)
        if isinstance(val, list):
            return [str(v) for v in val]
    except Exception:
        pass
    try:
        val = ast.literal_eval(s)
        if isinstance(val, list):
            return [str(v) for v in val]
    except Exception:
        pass
    return [s]


def parse_jsonish_dict(x: Any) -> Dict[str, Any]:
    if isinstance(x, dict):
        return x
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return {}
    s = str(x).strip()
    if not s:
        return {}
    try:
        val = json.loads(s)
        if isinstance(val, dict):
            return val
    except Exception:
        pass
    try:
        val = ast.literal_eval(s)
        if isinstance(val, dict):
            return val
    except Exception:
        pass
    return {}


def normalize_pred_record(record: Dict[str, Any]) -> Dict[str, Any]:
    rec = dict(record)
    if isinstance(rec.get("key_evidence"), str):
        rec["key_evidence"] = parse_jsonish_list(rec["key_evidence"])
    if isinstance(rec.get("structured_rationale"), str):
        rec["structured_rationale"] = parse_jsonish_dict(rec["structured_rationale"])
    if "retrieval_check_correct_reported" in rec and "retrieval_check_correct" not in rec:
        rec["retrieval_check_correct"] = bool(rec["retrieval_check_correct_reported"])
    return rec


def build_pred_record(
    rc: RunConfig,
    idx: int,
    test_case: Case,
    pred_obj: Dict[str, Any],
    validation_token: str,
) -> Dict[str, Any]:
    returned_token = str(pred_obj["retrieval_check_token_returned"])
    token_match = int(returned_token == validation_token)
    dispersion_true_high_low = (
        int(test_case.dispersion_true >= DISPERSION_HIGH_THRESHOLD)
        if test_case.dispersion_true is not None
        else np.nan
    )
    return {
        "shotset_name": rc.shotset_name,
        "modality": rc.modality,
        "case_id": test_case.case_id,
        "row_index": idx,
        "index_side": test_case.index_side,
        "has_preop_mri": has_report_text(test_case.preop_mri),
        "has_path_report": has_report_text(test_case.path_report),
        "dispersion_true": test_case.dispersion_true,
        "dispersion_true_high_low": dispersion_true_high_low,
        "relapse_true": test_case.relapse_true,
        "dispersion_score_pred": float(pred_obj["dispersion_score_pred"]),
        "dispersion_high_low_pred": int(pred_obj["dispersion_high_low_pred"]),
        "relapse_pred": int(pred_obj["relapse_pred"]),
        "key_evidence": pred_obj["key_evidence"],
        "retrieval_token_expected": validation_token,
        "retrieval_check_token_returned": returned_token,
        "retrieval_check_correct_reported": bool(pred_obj["retrieval_check_correct"]),
        "retrieval_token_exact_match": token_match,
        "reasoning_summary": str(pred_obj["reasoning_summary"]),
        "structured_rationale": pred_obj["structured_rationale"],
    }


def validate_saved_pred_record(
    record: Dict[str, Any],
    rc: RunConfig,
    idx: int,
    test_case: Case,
) -> Tuple[bool, str]:
    rec = normalize_pred_record(record)
    try:
        row_index = int(rec.get("row_index", -1))
    except (TypeError, ValueError):
        return False, "row_index is not an integer"
    if row_index != int(idx):
        return False, f"row_index mismatch: got {row_index} expected {idx}"
    if str(rec.get("shotset_name", "")) != str(rc.shotset_name):
        return False, "shotset_name mismatch"
    if str(rec.get("modality", "")) != str(rc.modality):
        return False, "modality mismatch"
    if str(rec.get("case_id", "")) != str(test_case.case_id):
        return False, f"case_id mismatch: got {rec.get('case_id')} expected {test_case.case_id}"

    expected_token = make_case_token(test_case, idx, rc.modality)
    if str(rec.get("retrieval_token_expected", "")) != expected_token:
        return False, "retrieval_token_expected does not match current prompt/token logic"

    obj = {
        "dispersion_score_pred": rec.get("dispersion_score_pred"),
        "dispersion_high_low_pred": rec.get("dispersion_high_low_pred"),
        "relapse_pred": rec.get("relapse_pred"),
        "key_evidence": rec.get("key_evidence"),
        "retrieval_check_token_returned": rec.get("retrieval_check_token_returned"),
        "retrieval_check_correct": rec.get("retrieval_check_correct", rec.get("retrieval_check_correct_reported")),
        "reasoning_summary": rec.get("reasoning_summary"),
        "structured_rationale": rec.get("structured_rationale"),
    }
    ok, msg = validate_prediction_obj(
        obj,
        expected_case_id=test_case.case_id,
        expected_token=expected_token,
    )
    if not ok:
        return False, msg
    return True, ""
