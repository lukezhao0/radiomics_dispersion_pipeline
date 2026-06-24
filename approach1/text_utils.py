"""Text normalization and helper utilities."""

from __future__ import annotations

import math
import re
from typing import Any, List, Optional

import numpy as np
from sklearn.metrics import mean_squared_error

from .config import STOPWORDS


def safe_text(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and np.isnan(x):
        return ""
    return str(x)


def has_report_text(x: Any) -> bool:
    s = safe_text(x).strip()
    if not s:
        return False
    return s.lower() not in {"nan", "none", "null", "na", "n/a"}


def shorten_for_prompt(text: str) -> str:
    return text


def word_count(s: str) -> int:
    return len([w for w in re.split(r"\s+", s.strip()) if w])


def normalize_side(side: Optional[str]) -> str:
    s = (side or "").strip().lower()
    return s if s in {"left", "right"} else "unknown"


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(math.sqrt(mean_squared_error(y_true, y_pred)))


def normalize_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s\-\/]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def tokenize_quote(s: str) -> List[str]:
    s = normalize_text(s)
    return [t for t in s.split() if len(t) >= 3 and t not in STOPWORDS]


def modality_display_name(modality: str) -> str:
    return {
        "mri_only": "MRI only",
        "pathology_only": "Pathology only",
        "mri_plus_pathology": "MRI + pathology",
    }[modality]


def modality_requires_mri(modality: str) -> bool:
    return modality in {"mri_only", "mri_plus_pathology"}


def modality_uses_pathology(modality: str) -> bool:
    return modality in {"pathology_only", "mri_plus_pathology"}
