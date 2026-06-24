"""Deterministic validation tokens for needle-in-the-haystack checks."""

from __future__ import annotations

import hashlib
import re

from ..models import Case
from ..text_utils import normalize_side


def case_id_for_token(case_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_\-]+", "_", str(case_id)).strip("_")
    return safe or "case"


def make_case_token(test_case: Case, row_index: int, modality: str) -> str:
    side = normalize_side(test_case.index_side)
    base = f"{test_case.case_id}|row_{row_index}|side_{side}|modality_{modality}"
    digest = hashlib.sha256(base.encode("utf-8")).hexdigest()[:10].upper()
    cid = case_id_for_token(test_case.case_id)
    return f"CTXCHK_{digest}_CASE_{cid}_ROW_{row_index}_SIDE_{side}_MODALITY_{modality}"
