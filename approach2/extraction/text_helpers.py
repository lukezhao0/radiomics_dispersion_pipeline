"""Quote matching and text helpers for extraction."""

from __future__ import annotations

import json
import re
import unicodedata
from difflib import SequenceMatcher
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

def _safe_text(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and np.isnan(x):
        return ""
    return str(x)


def _word_count(s: str) -> int:
    return len([w for w in re.split(r"\s+", s.strip()) if w])


_MISSING_TEXT_PLACEHOLDERS = frozenset({
    "",
    "na",
    "n/a",
    "nan",
    "none",
    "null",
    "missing",
    "not available",
    "not applicable",
    "[missing]",
    "no mri",
    "no report",
})


def _is_missing_text(x: Any) -> bool:
    s = _safe_text(x).strip()
    if not s:
        return True
    return s.lower() in _MISSING_TEXT_PLACEHOLDERS


def _normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip()


def _normalize_for_quote_match(s: str) -> str:
    """Normalize text for robust quote matching without changing stored quotes."""
    s = unicodedata.normalize("NFKC", str(s or ""))
    replacements = {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u00a0": " ",
    }
    for old, new in replacements.items():
        s = s.replace(old, new)
    s = re.sub(r"\s+", " ", s)
    return s.strip().lower()


def _quote_present_in_report(quote: str, report_text: str) -> bool:
    """Return True if quote is present under exact or normalized matching."""
    if not quote or not report_text:
        return False
    q_raw = _normalize_ws(quote)
    r_raw = _normalize_ws(report_text)
    if not q_raw:
        return False
    if q_raw.lower() in r_raw.lower():
        return True
    q_norm = _normalize_for_quote_match(q_raw)
    r_norm = _normalize_for_quote_match(r_raw)
    return bool(q_norm and q_norm in r_norm)


def _token_set_overlap(a: str, b: str) -> float:
    a_tokens = set(re.findall(r"[a-z0-9]+", _normalize_for_quote_match(a)))
    b_tokens = set(re.findall(r"[a-z0-9]+", _normalize_for_quote_match(b)))
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / max(1, len(a_tokens | b_tokens))


def _iter_report_token_windows(report_text: str, target_word_count: int) -> Sequence[str]:
    """Yield exact substrings from the original report near the target quote length."""
    tokens = [(m.group(0), m.start(), m.end()) for m in re.finditer(r"\S+", report_text or "")]
    if not tokens:
        return []

    target_word_count = max(1, int(target_word_count))
    min_len = max(2, target_word_count - 5)
    max_len = min(35, target_word_count + 8)
    if min_len > max_len:
        min_len = max_len

    windows: List[str] = []
    for width in range(min_len, max_len + 1):
        if width > len(tokens):
            continue
        for i in range(0, len(tokens) - width + 1):
            start = tokens[i][1]
            end = tokens[i + width - 1][2]
            span = report_text[start:end]
            if span.strip():
                windows.append(span)
    return windows


def _repair_quote_to_exact_report_span(
    quote: str,
    report_text: str,
    min_similarity: float = 0.84,
    min_token_overlap: float = 0.55,
) -> Tuple[Optional[str], float, float]:
    """Try to repair a non-exact model quote to an exact report substring.

    Returns (repaired_quote, sequence_similarity, token_overlap). The repaired
    quote is always copied exactly from report_text.
    """
    quote = _normalize_ws(quote)
    if not quote or not report_text:
        return None, 0.0, 0.0

    if _quote_present_in_report(quote, report_text):
        return quote, 1.0, 1.0

    q_norm = _normalize_for_quote_match(quote)
    q_words = _word_count(quote)
    best_span: Optional[str] = None
    best_similarity = 0.0
    best_overlap = 0.0
    best_score = -1.0

    # First try line/sentence-level candidates because they are faster and often
    # preserve clinically meaningful spans.
    candidates: List[str] = []
    for part in re.split(r"[\n\r]+|(?<=[.;:])\s+", report_text or ""):
        part = part.strip()
        if 2 <= _word_count(part) <= 35:
            candidates.append(part)

    # Add token windows around the same length as the failed quote.
    candidates.extend(_iter_report_token_windows(report_text, q_words))

    seen = set()
    for cand in candidates:
        cand = cand.strip()
        if not cand or cand in seen:
            continue
        seen.add(cand)
        cand_norm = _normalize_for_quote_match(cand)
        if not cand_norm:
            continue
        similarity = SequenceMatcher(None, q_norm, cand_norm).ratio()
        overlap = _token_set_overlap(q_norm, cand_norm)
        score = 0.75 * similarity + 0.25 * overlap
        if score > best_score:
            best_score = score
            best_span = cand
            best_similarity = similarity
            best_overlap = overlap

    if (
        best_span is not None
        and best_similarity >= min_similarity
        and best_overlap >= min_token_overlap
        and _quote_present_in_report(best_span, report_text)
    ):
        return best_span, float(best_similarity), float(best_overlap)

    return None, float(best_similarity), float(best_overlap)
