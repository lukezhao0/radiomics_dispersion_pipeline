"""
Leakage-aware nested clinical NLP/ML pipeline for breast tumor dispersiveness.

Phase-1 modular package extracted from the original monolithic approach2.py script.
Orchestration, fitting, and reporting remain in approach2.py for now.
"""

from __future__ import annotations

from .config import (
    AMBIGUITY_GROUPS,
    CANONICAL_GROUP_PATTERNS,
    COEF_ZERO_TOL,
    DEFAULT_BOOTSTRAP_N,
    DISPERSION_TRUE_HIGH_THRESHOLD,
    DISTRIBUTION_GROUPS,
    EPS,
    INNER_CV_MAX_SPLITS,
    META_COLS,
    NEGATION_PATTERNS,
    RANDOM_SEED,
    SHARED_CONCEPT_ONTOLOGY,
    SPATIAL_MORPH_RESPONSE_GROUPS,
    TARGET_NAME_DISPERSION_HIGH_LOW,
    TARGET_NAME_DISPERSION_SCORE,
    TARGET_NAME_RELAPSE_STATUS,
    UNCERTAINTY_PATTERNS,
)
from .io_atomic import atomic_write_df, safe_read_csv_if_exists
from .metrics import (
    calibration_intercept_slope,
    rmse,
    safe_pearson,
    safe_spearman,
)
from .models import LowInfoFeatureFilter, ModelSpec
from .text_utils import (
    clean_phrase_for_display,
    detect_negation,
    detect_uncertainty,
    make_slug,
    normalize_text,
    parse_jsonish,
    resolve_default_api_workers,
    resolve_default_ml_n_jobs,
    resolve_default_parallel_modality_workers,
)

__all__ = [
    "AMBIGUITY_GROUPS",
    "CANONICAL_GROUP_PATTERNS",
    "COEF_ZERO_TOL",
    "DEFAULT_BOOTSTRAP_N",
    "DISPERSION_TRUE_HIGH_THRESHOLD",
    "DISTRIBUTION_GROUPS",
    "EPS",
    "INNER_CV_MAX_SPLITS",
    "META_COLS",
    "NEGATION_PATTERNS",
    "RANDOM_SEED",
    "SHARED_CONCEPT_ONTOLOGY",
    "SPATIAL_MORPH_RESPONSE_GROUPS",
    "TARGET_NAME_DISPERSION_HIGH_LOW",
    "TARGET_NAME_DISPERSION_SCORE",
    "TARGET_NAME_RELAPSE_STATUS",
    "UNCERTAINTY_PATTERNS",
    "LowInfoFeatureFilter",
    "ModelSpec",
    "atomic_write_df",
    "calibration_intercept_slope",
    "clean_phrase_for_display",
    "detect_negation",
    "detect_uncertainty",
    "make_slug",
    "normalize_text",
    "parse_jsonish",
    "resolve_default_api_workers",
    "resolve_default_ml_n_jobs",
    "resolve_default_parallel_modality_workers",
    "rmse",
    "safe_pearson",
    "safe_read_csv_if_exists",
    "safe_spearman",
]
