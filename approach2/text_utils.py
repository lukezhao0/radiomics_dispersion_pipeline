"""Text normalization, slugging, and parallelism default helpers for approach2."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Optional

import pandas as pd

from .config import NEGATION_PATTERNS, UNCERTAINTY_PATTERNS


def resolve_default_ml_n_jobs(requested_jobs: Optional[int]) -> int:
    if requested_jobs is not None:
        return max(1, int(requested_jobs))
    cpu_count = os.cpu_count() or 1
    return max(1, min(4, cpu_count - 2))


def resolve_default_parallel_modality_workers(requested_workers: Optional[int]) -> int:
    if requested_workers is not None:
        return max(1, int(requested_workers))
    cpu_count = os.cpu_count() or 1
    return 2 if cpu_count >= 8 else 1


def resolve_default_api_workers(requested_workers: Optional[int]) -> int:
    if requested_workers is not None:
        return max(1, int(requested_workers))
    cpu_count = os.cpu_count() or 1
    return 2 if cpu_count >= 8 else 1


def parse_jsonish(x: Any) -> Any:
    if isinstance(x, (dict, list)):
        return x
    if pd.isna(x):
        return None
    s = str(x).strip()
    if not s:
        return None
    try:
        return json.loads(s)
    except Exception:
        pass
    return None


def normalize_text(s: str) -> str:
    s = str(s or "").lower()
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def make_slug(s: str) -> str:
    s = normalize_text(s)
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s[:120] if s else "empty"


def detect_negation(text: str) -> bool:
    t = normalize_text(text)
    return any(re.search(p, t) for p in NEGATION_PATTERNS)


def detect_uncertainty(text: str) -> bool:
    t = normalize_text(text)
    return any(re.search(p, t) for p in UNCERTAINTY_PATTERNS)


def clean_phrase_for_display(s: str) -> str:
    s = str(s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s
