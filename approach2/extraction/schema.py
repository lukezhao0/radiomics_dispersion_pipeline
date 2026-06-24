"""Extraction JSON schema validation and normalization."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .text_helpers import (
    _normalize_ws,
    _quote_present_in_report,
    _repair_quote_to_exact_report_span,
    _word_count,
)

def _extract_json_from_text(text: str) -> Dict[str, Any]:
    text = (text or "").strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    fenced = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
    fenced = re.sub(r"\s*```$", "", fenced.strip())
    try:
        return json.loads(fenced)
    except Exception:
        pass

    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        raise ValueError("No JSON object found in model output.")
    return json.loads(m.group(0))


def _coerce_candidate_concepts(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(v).strip() for v in x if str(v).strip()]
    if isinstance(x, str):
        parts = re.split(r"[,;|]", x)
        return [p.strip() for p in parts if p.strip()]
    return []


def _coerce_float_or_none(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        v = float(x)
    except Exception:
        return None
    if np.isnan(v):
        return None
    return v


def _normalize_phrase_item_schema(item: Dict[str, Any], report_mode: str) -> Dict[str, Any]:
    """Fill optional fields so older extraction outputs remain downstream-compatible."""
    item = dict(item)
    quote = str(item.get("quote", "")).strip()
    concept = str(item.get("concept", "")).strip()
    item.setdefault("normalized_phrase", re.sub(r"\s+", " ", quote.lower()).strip())
    item["candidate_concepts"] = _coerce_candidate_concepts(item.get("candidate_concepts"))
    if not item["candidate_concepts"] and concept:
        item["candidate_concepts"] = [concept]
    item.setdefault("section", "unknown")
    item.setdefault("directness", "direct" if report_mode == "path" else "unknown")
    item.setdefault("directly_asserts_tumor", None)
    item.setdefault("imaging_pattern_only", None)
    item.setdefault("biological_ambiguity", "unknown")
    item.setdefault("mapping_confidence", None)
    item["mapping_confidence"] = _coerce_float_or_none(item.get("mapping_confidence"))
    item.setdefault("quantitative_attributes", {})
    if not isinstance(item["quantitative_attributes"], dict):
        item["quantitative_attributes"] = {}
    return item

def _sanitize_phrase_item_for_validation(
    item: Dict[str, Any],
    report_text: str,
    item_name: str,
    report_mode: str,
) -> Tuple[Optional[Dict[str, Any]], List[str], int]:
    """Normalize, repair, or drop one phrase item before hard validation.

    A single invalid model phrase should not invalidate the entire case. If the
    quote is a near miss, repair it to an exact report substring. If it cannot be
    repaired, drop only that phrase item and preserve a warning.
    """
    warnings: List[str] = []
    repaired = 0

    if not isinstance(item, dict):
        return None, [f"{item_name} dropped: item is not an object"], repaired

    item = _normalize_phrase_item_schema(item, report_mode=report_mode)

    # Normalize controlled vocab fields conservatively.
    item["polarity"] = str(item.get("polarity", "affirmed")).strip().lower()
    item["certainty"] = str(item.get("certainty", "certain")).strip().lower()
    if item["polarity"] not in {"affirmed", "negated", "uncertain"}:
        warnings.append(f"{item_name}.polarity coerced from {item.get('polarity')!r} to 'uncertain'")
        item["polarity"] = "uncertain"
    if item["certainty"] not in {"certain", "uncertain"}:
        warnings.append(f"{item_name}.certainty coerced from {item.get('certainty')!r} to 'uncertain'")
        item["certainty"] = "uncertain"

    concept = str(item.get("concept", "")).strip()
    if not concept:
        return None, [f"{item_name} dropped: missing/empty concept"], repaired

    quote = str(item.get("quote", "")).strip()
    if not quote:
        return None, [f"{item_name} dropped: missing/empty quote"], repaired

    if _word_count(quote) > 35 or not _quote_present_in_report(quote, report_text):
        repaired_quote, similarity, overlap = _repair_quote_to_exact_report_span(quote, report_text)
        if repaired_quote and _word_count(repaired_quote) <= 35:
            item["original_quote_before_repair"] = quote
            item["quote"] = repaired_quote
            item["quote_repair_status"] = "repaired_to_exact_report_substring"
            item["quote_repair_similarity"] = similarity
            item["quote_repair_token_overlap"] = overlap
            repaired = 1
            warnings.append(
                f"{item_name}.quote repaired to exact report substring "
                f"(similarity={similarity:.3f}, token_overlap={overlap:.3f})"
            )
        else:
            return None, [
                f"{item_name} dropped: quote not found and could not be repaired; "
                f"best_similarity={similarity:.3f}, best_token_overlap={overlap:.3f}; "
                f"quote={quote[:160]!r}"
            ], repaired

    return item, warnings, repaired


def _sanitize_extraction_obj_for_validation(
    obj: Dict[str, Any],
    report_text: str,
    report_mode: str,
) -> Dict[str, Any]:
    """Repair/drop invalid phrase-level outputs before strict object validation."""
    obj = dict(obj)
    validation_warnings: List[str] = []
    n_repaired = 0
    n_dropped = 0

    for list_key in ["seed_aligned_phrases", "denovo_candidate_phrases"]:
        raw_items = obj.get(list_key, [])
        if not isinstance(raw_items, list):
            validation_warnings.append(f"{list_key} was not a list; replacing with empty list")
            obj[list_key] = []
            continue

        cleaned: List[Dict[str, Any]] = []
        for i, item in enumerate(raw_items):
            cleaned_item, warnings, repaired = _sanitize_phrase_item_for_validation(
                item=item,
                report_text=report_text,
                item_name=f"{list_key}[{i}]",
                report_mode=report_mode,
            )
            validation_warnings.extend(warnings)
            n_repaired += int(repaired)
            if cleaned_item is None:
                n_dropped += 1
            else:
                cleaned.append(cleaned_item)
        obj[list_key] = cleaned

    obj["validation_warnings"] = validation_warnings
    obj["n_validation_warnings"] = len(validation_warnings)
    obj["n_repaired_phrase_quotes"] = n_repaired
    obj["n_dropped_phrase_items"] = n_dropped
    return obj

# -----------------------------
# Validation
# -----------------------------

def _validate_phrase_item(item: Dict[str, Any], report_text: str, item_name: str) -> Tuple[bool, str]:
    required = ["quote", "concept", "polarity", "certainty"]
    for k in required:
        if k not in item:
            return False, f"{item_name} missing key: {k}"

    quote = item["quote"]
    if not isinstance(quote, str) or not quote.strip():
        return False, f"{item_name}.quote must be a non-empty string"
    if _word_count(quote) > 35:
        return False, f"{item_name}.quote exceeds 35 words"
    if not _quote_present_in_report(quote, report_text):
        return False, f"{item_name}.quote is not found in report text"

    concept = item["concept"]
    if not isinstance(concept, str) or not concept.strip():
        return False, f"{item_name}.concept must be a non-empty string"

    polarity = str(item["polarity"]).strip().lower()
    if polarity not in {"affirmed", "negated", "uncertain"}:
        return False, f"{item_name}.polarity must be affirmed/negated/uncertain"

    certainty = str(item["certainty"]).strip().lower()
    if certainty not in {"certain", "uncertain"}:
        return False, f"{item_name}.certainty must be certain/uncertain"

    if "candidate_concepts" in item and not isinstance(item["candidate_concepts"], list):
        return False, f"{item_name}.candidate_concepts must be a list when supplied"

    return True, "ok"


def _validate_quantitative_attributes(q: Dict[str, Any]) -> Tuple[bool, str]:
    required = [
        "extent_cm",
        "largest_focus_cm",
        "margin_distance_mm",
        "lvi_present",
        "dcis_burden",
        "nme_present",
        "satellite_lesions_present",
        "multifocal_present",
        "multicentric_present",
        "residual_disease_minimal",
        "single_localized_residual",
        "diffuse_scattered_residual",
    ]
    for k in required:
        if k not in q:
            return False, f"quantitative_attributes missing key: {k}"
    return True, "ok"


def _normalize_extraction_obj(obj: Dict[str, Any], report_mode: str) -> Dict[str, Any]:
    obj = dict(obj)
    for list_key in ["seed_aligned_phrases", "denovo_candidate_phrases"]:
        items = obj.get(list_key, [])
        if isinstance(items, list):
            obj[list_key] = [
                _normalize_phrase_item_schema(item, report_mode) if isinstance(item, dict) else item
                for item in items
            ]
    return obj


def _validate_extraction_obj(
    obj: Dict[str, Any],
    expected_case_id: str,
    report_text: str,
    report_mode: str,
) -> Tuple[bool, str]:
    required = [
        "case_id",
        "report_mode",
        "selected_report_field",
        "selected_report_missing",
        "seed_aligned_phrases",
        "denovo_candidate_phrases",
        "quantitative_attributes",
        "report_level_summary",
    ]
    for k in required:
        if k not in obj:
            return False, f"Missing key: {k}"

    if str(obj["case_id"]) != str(expected_case_id):
        return False, f"case_id mismatch: got {obj['case_id']} expected {expected_case_id}"

    if str(obj["report_mode"]).strip().lower() != report_mode:
        return False, f"report_mode mismatch: got {obj['report_mode']} expected {report_mode}"

    if bool(obj["selected_report_missing"]):
        return False, "selected_report_missing must be false for a non-missing report extraction"

    for list_key in ["seed_aligned_phrases", "denovo_candidate_phrases"]:
        if not isinstance(obj[list_key], list):
            return False, f"{list_key} must be a list"
        for i, item in enumerate(obj[list_key]):
            if not isinstance(item, dict):
                return False, f"{list_key}[{i}] must be an object"
            ok, msg = _validate_phrase_item(item, report_text, f"{list_key}[{i}]")
            if not ok:
                return False, msg

    qa = obj["quantitative_attributes"]
    if not isinstance(qa, dict):
        return False, "quantitative_attributes must be an object"
    ok, msg = _validate_quantitative_attributes(qa)
    if not ok:
        return False, msg

    rls = obj["report_level_summary"]
    if not isinstance(rls, dict):
        return False, "report_level_summary must be an object"

    return True, "ok"
