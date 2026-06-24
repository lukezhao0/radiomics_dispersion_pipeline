"""Core data structures for the approach1 pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


@dataclass
class Case:
    case_id: str
    preop_mri: str
    path_report: str
    index_side: str | None = None
    dispersion_true: float | None = None
    relapse_true: int | None = None


@dataclass
class RunConfig:
    shotset_name: str
    high_rows: List[int]
    low_rows: List[int]
    training_rows: List[int]
    modality: str
    run_out_dir: str
    training_block: str
    test_cases_with_idxs: List[Tuple[int, Case]]
    skipped_missing_mri: List[Tuple[int, Case]]
    apriori_cost: Dict[str, Any]
