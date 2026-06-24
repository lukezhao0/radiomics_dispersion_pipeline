"""Case model and CSV loading for extraction."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .config import REPORT_CONFIG
from .text_helpers import _is_missing_text, _safe_text

@dataclass
class Case:
    case_id: str
    preop_mri: str
    path_report: str
    index_side: Optional[str] = None
    dispersion_true: Optional[float] = None
    relapse_true: Optional[int] = None


def _selected_report_text(case: Case, report_mode: str) -> str:
    if report_mode == "mri":
        return case.preop_mri
    if report_mode == "path":
        return case.path_report
    raise ValueError(f"Unsupported report_mode: {report_mode}")


def _selected_report_field(report_mode: str) -> str:
    return REPORT_CONFIG[report_mode]["field"]


def _selected_report_label(report_mode: str) -> str:
    return REPORT_CONFIG[report_mode]["label"]


def _true_dispersion_high_low(x: Any, threshold: float = 85.0) -> float:
    try:
        v = float(x)
    except Exception:
        return np.nan
    if np.isnan(v):
        return np.nan
    return float(int(v >= threshold))

def load_cases(csv_path: str) -> pd.DataFrame:
    print(f"[DATA] Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)

    required_cols = {
        "preop_MRI_text",
        "path_report_text",
        "index_side",
        "dispersion_invasive_DCIS_geographic",
        "relapse",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")

    if "case_id" not in df.columns:
        df = df.copy()
        df["case_id"] = [f"row_{i}" for i in range(len(df))]

    print(f"[DATA] Loaded dataframe: rows={len(df)} cols={len(df.columns)}")
    return df


def make_case_from_row(df: pd.DataFrame, idx: int) -> Case:
    row = df.iloc[idx]
    return Case(
        case_id=str(row["case_id"]),
        preop_mri=_safe_text(row["preop_MRI_text"]),
        path_report=_safe_text(row["path_report_text"]),
        index_side=_safe_text(row["index_side"]),
        dispersion_true=float(row["dispersion_invasive_DCIS_geographic"])
        if pd.notna(row["dispersion_invasive_DCIS_geographic"])
        else None,
        relapse_true=int(row["relapse"]) if pd.notna(row["relapse"]) else None,
    )


def make_missing_extraction_record(
    test_case: Case,
    row_index: int,
    report_mode: str,
    split_id: Optional[str] = None,
    split_role: Optional[str] = None,
) -> Dict[str, Any]:
    selected_field = _selected_report_field(report_mode)
    selected_text = _selected_report_text(test_case, report_mode)
    return {
        "case_id": test_case.case_id,
        "row_index": row_index,
        "report_mode": report_mode,
        "selected_report_field": selected_field,
        "selected_report_text": selected_text,
        "selected_report_missing": 1,
        "selected_report_missing_reason": f"Missing {selected_field}; no extraction generated.",
        "has_preop_mri": int(not _is_missing_text(test_case.preop_mri)),
        "has_path_report": int(not _is_missing_text(test_case.path_report)),
        "dispersion_true": test_case.dispersion_true,
        "dispersion_true_high_low": _true_dispersion_high_low(test_case.dispersion_true),
        "relapse_true": test_case.relapse_true,
        "outer_split_id": split_id or "",
        "outer_split_role": split_role or "",
        "seed_aligned_phrases": [],
        "denovo_candidate_phrases": [],
        "quantitative_attributes": {
            "extent_cm": None,
            "largest_focus_cm": None,
            "margin_distance_mm": None,
            "lvi_present": None,
            "dcis_burden": None,
            "nme_present": None,
            "satellite_lesions_present": None,
            "multifocal_present": None,
            "multicentric_present": None,
            "residual_disease_minimal": None,
            "single_localized_residual": None,
            "diffuse_scattered_residual": None,
        },
        "report_level_summary": {
            "distribution_pattern": "unknown",
            "distribution_evidence_quote": "",
            "localization_vs_scatter_note": "Selected modality missing; no lexical extraction performed.",
        },
    }
