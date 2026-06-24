"""LLM prediction JSON validation and parsing."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from ..config import DISPERSION_HIGH_THRESHOLD
from .quote_helpers import (
    _quote_present_in_report,
    _repair_quote_to_exact_report_span,
    _word_count,
)

KEY_EVIDENCE_MAX_WORDS = 35


def _sanitize_key_evidence_quote(
    quote: Any,
    report_text: str,
    item_name: str,
) -> Tuple[Optional[str], List[str], int]:
    """Normalize, repair, or drop one key_evidence quote before hard validation."""
    warnings: List[str] = []
    repaired = 0

    if not isinstance(quote, str):
        return None, [f"{item_name} dropped: quote is not a string"], repaired

    quote = quote.strip()
    if not quote:
        return None, [f"{item_name} dropped: empty quote"], repaired

    if _word_count(quote) > KEY_EVIDENCE_MAX_WORDS or not _quote_present_in_report(quote, report_text):
        repaired_quote, similarity, overlap = _repair_quote_to_exact_report_span(quote, report_text)
        if repaired_quote and _word_count(repaired_quote) <= KEY_EVIDENCE_MAX_WORDS:
            repaired = 1
            warnings.append(
                f"{item_name} repaired to exact report substring "
                f"(similarity={similarity:.3f}, token_overlap={overlap:.3f})"
            )
            return repaired_quote, warnings, repaired
        return None, [
            f"{item_name} dropped: quote not found and could not be repaired; "
            f"best_similarity={similarity:.3f}, best_token_overlap={overlap:.3f}; "
            f"quote={quote[:160]!r}"
        ], repaired

    return quote, warnings, repaired


def sanitize_prediction_obj_for_validation(
    obj: Dict[str, Any],
    report_text: str,
) -> Dict[str, Any]:
    """Repair/drop invalid key_evidence quotes before strict object validation."""
    obj = dict(obj)
    validation_warnings: List[str] = []
    n_repaired = 0
    n_dropped = 0

    raw_items = obj.get("key_evidence", [])
    if not isinstance(raw_items, list):
        validation_warnings.append("key_evidence was not a list; replacing with empty list")
        obj["key_evidence"] = []
    else:
        cleaned: List[str] = []
        for i, quote in enumerate(raw_items):
            cleaned_quote, warnings, repaired = _sanitize_key_evidence_quote(
                quote=quote,
                report_text=report_text,
                item_name=f"key_evidence[{i}]",
            )
            validation_warnings.extend(warnings)
            n_repaired += repaired
            if cleaned_quote is None:
                n_dropped += 1
            else:
                cleaned.append(cleaned_quote)
        obj["key_evidence"] = cleaned

    obj["validation_warnings"] = validation_warnings
    obj["n_validation_warnings"] = len(validation_warnings)
    obj["n_repaired_key_evidence_quotes"] = n_repaired
    obj["n_dropped_key_evidence_quotes"] = n_dropped
    return obj


def validate_prediction_obj(
    obj: Dict[str, Any],
    expected_case_id: str,
    expected_token: str,
    report_text: Optional[str] = None,
) -> Tuple[bool, str]:
    required = [
        "dispersion_score_pred",
        "dispersion_high_low_pred",
        "relapse_pred",
        "key_evidence",
        "retrieval_check_token_returned",
        "retrieval_check_correct",
        "reasoning_summary",
        "structured_rationale",
    ]
    for k in required:
        if k not in obj:
            return False, f"Missing key: {k}"

    if "case_id" in obj and str(obj["case_id"]) != str(expected_case_id):
        return False, f"case_id mismatch: got {obj['case_id']} expected {expected_case_id}"

    try:
        dsp = float(obj["dispersion_score_pred"])
    except Exception:
        return False, "dispersion_score_pred is not a number"
    if not (0.0 <= dsp <= 450.0):
        return False, f"dispersion_score_pred out of range [0,450]: {dsp}"

    try:
        dhl = int(obj["dispersion_high_low_pred"])
    except Exception:
        return False, "dispersion_high_low_pred is not an integer"
    if dhl not in (0, 1):
        return False, f"dispersion_high_low_pred must be 0/1, got {dhl}"

    try:
        rp = int(obj["relapse_pred"])
    except Exception:
        return False, "relapse_pred is not an integer"
    if rp not in (0, 1):
        return False, f"relapse_pred must be 0/1, got {rp}"

    ke = obj["key_evidence"]
    if not isinstance(ke, list):
        return False, "key_evidence must be a list"
    if len(ke) > 6:
        return False, f"key_evidence must have <= 6 items, got {len(ke)}"
    for i, q in enumerate(ke):
        if not isinstance(q, str):
            return False, f"key_evidence[{i}] must be a string"
        if not q.strip():
            return False, f"key_evidence[{i}] is empty"
        if _word_count(q) > KEY_EVIDENCE_MAX_WORDS:
            return False, f"key_evidence[{i}] exceeds {KEY_EVIDENCE_MAX_WORDS} words"
        if report_text and not _quote_present_in_report(q, report_text):
            return False, f"key_evidence[{i}] is not found in report text"

    returned_token = obj["retrieval_check_token_returned"]
    if not isinstance(returned_token, str) or not returned_token.strip():
        return False, "retrieval_check_token_returned must be a non-empty string"

    token_correct = obj["retrieval_check_correct"]
    if token_correct not in [True, False]:
        return False, "retrieval_check_correct must be boolean"

    if not isinstance(obj["reasoning_summary"], str) or not obj["reasoning_summary"].strip():
        return False, "reasoning_summary must be a non-empty string"

    sr = obj["structured_rationale"]
    if not isinstance(sr, dict):
        return False, "structured_rationale must be an object"
    for subk in [
        "step_1_localization",
        "step_2_pathology_pattern",
        "step_3_mri_pattern",
        "step_4_dispersion_synthesis",
        "step_5_relapse_synthesis",
    ]:
        if subk not in sr:
            return False, f"structured_rationale missing key: {subk}"
        if not isinstance(sr[subk], str) or not sr[subk].strip():
            return False, f"structured_rationale[{subk}] must be a non-empty string"

    expected_correct = returned_token == expected_token
    if bool(token_correct) != expected_correct:
        return False, (
            "retrieval_check_correct inconsistent with returned token "
            f"(expected {expected_correct}, got {token_correct})"
        )

    if dhl != int(dsp >= DISPERSION_HIGH_THRESHOLD):
        return False, (
            f"dispersion_high_low_pred inconsistent with dispersion_score_pred: "
            f"score={dsp}, label={dhl}, expected={int(dsp >= DISPERSION_HIGH_THRESHOLD)}"
        )

    return True, "ok"


def extract_json_from_text(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        raise ValueError("No JSON object found in model output.")
    return json.loads(m.group(0))
