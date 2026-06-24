"""Phrase explosion and extraction CSV normalization."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import html
import json
import math
import os
import re
import sys
import time
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, HuberRegressor, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    mutual_info_score,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold, StratifiedShuffleSplit, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR, SVC

from approach2.config import (
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
from approach2.extraction import (
    MAX_TOKENS,
    Tee,
    _is_missing_text,
    _selected_report_text,
    _true_dispersion_high_low,
    build_html_report,
    build_user_prompt,
    confirm_cost_estimate_or_exit,
    configure_global_api_concurrency,
    df_to_html_table,
    estimate_prompt_tokens_from_messages,
    extract_subset_records,
    html_paragraph,
    html_plot_block,
    html_section,
    load_cases,
    make_case_from_row,
    preflight_check,
    print_apriori_cost_estimate_report,
    print_cumulative_report,
    summarize_apriori_cost_estimate,
    write_cost_tracker_json,
    write_extractions,
)
from approach2.io_atomic import atomic_write_df as _atomic_write_df
from approach2.io_atomic import safe_read_csv_if_exists as _safe_read_csv_if_exists
from approach2.metrics import calibration_intercept_slope, rmse, safe_pearson, safe_spearman
from approach2.models import LowInfoFeatureFilter, ModelSpec
from approach2.text_utils import (
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

def _coerce_candidate_concepts(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [make_slug(str(v)) for v in x if str(v).strip()]
    if isinstance(x, str):
        return [make_slug(p) for p in re.split(r"[,;|]", x) if p.strip()]
    return []


def infer_shared_concepts(raw_concept: str, quote: str, candidate_concepts: Optional[Sequence[str]] = None) -> List[str]:
    concepts: List[str] = []
    for c in candidate_concepts or []:
        c_slug = make_slug(str(c))
        if c_slug in SHARED_CONCEPT_ONTOLOGY and c_slug not in concepts:
            concepts.append(c_slug)

    search_text = normalize_text(f"{raw_concept} || {quote}")
    for group, patterns in CANONICAL_GROUP_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, search_text):
                if group not in concepts:
                    concepts.append(group)
                break

    if not concepts:
        concepts.append("other_candidate_feature")
    return concepts


def infer_canonical_group(raw_concept: str, quote: str, candidate_concepts: Optional[Sequence[str]] = None) -> str:
    return infer_shared_concepts(raw_concept, quote, candidate_concepts)[0]

def load_extractions_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in [
        "seed_aligned_phrases",
        "denovo_candidate_phrases",
        "quantitative_attributes",
        "report_level_summary",
    ]:
        if col in df.columns:
            df[col] = df[col].apply(parse_jsonish)
    return df


def ontology_table() -> pd.DataFrame:
    rows = []
    for concept, spec in SHARED_CONCEPT_ONTOLOGY.items():
        rows.append({
            "concept_name": concept,
            "category": spec.get("category", ""),
            "definition": spec.get("definition", ""),
            "mri_examples": spec.get("mri_examples", ""),
            "path_examples": spec.get("path_examples", ""),
            "regex_patterns": "; ".join(spec.get("patterns", [])),
        })
    return pd.DataFrame(rows)


# -----------------------------
# Extraction normalization
# -----------------------------

def explode_phrase_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for _, rec in df.iterrows():
        if int(rec.get("selected_report_missing", 0)) == 1:
            continue

        for source_col, source_name in [
            ("seed_aligned_phrases", "seed"),
            ("denovo_candidate_phrases", "denovo"),
        ]:
            items = rec.get(source_col)
            if not isinstance(items, list):
                continue

            for item in items:
                if not isinstance(item, dict):
                    continue

                quote = clean_phrase_for_display(item.get("quote", ""))
                raw_concept = clean_phrase_for_display(item.get("concept", ""))
                normalized_phrase = clean_phrase_for_display(item.get("normalized_phrase", "")) or normalize_text(quote)
                candidate_concepts = _coerce_candidate_concepts(item.get("candidate_concepts"))
                polarity = str(item.get("polarity", "")).strip().lower()
                certainty = str(item.get("certainty", "")).strip().lower()
                laterality = str(item.get("laterality", "unknown")).strip().lower()
                span_type = str(item.get("span_type", "unknown")).strip().lower()
                section = str(item.get("section", "unknown")).strip().lower()
                directness = str(item.get("directness", "unknown")).strip().lower()
                biological_ambiguity = str(item.get("biological_ambiguity", "unknown")).strip().lower()
                mapping_confidence = pd.to_numeric(item.get("mapping_confidence", np.nan), errors="coerce")
                directly_asserts_tumor = item.get("directly_asserts_tumor", None)
                imaging_pattern_only = item.get("imaging_pattern_only", None)

                if not quote:
                    continue

                polarity_final = polarity if polarity in {"affirmed", "negated", "uncertain"} else "affirmed"
                certainty_final = certainty if certainty in {"certain", "uncertain"} else "certain"

                if polarity_final == "affirmed" and detect_negation(quote):
                    polarity_final = "negated"
                if certainty_final == "certain" and detect_uncertainty(quote):
                    certainty_final = "uncertain"
                if polarity_final == "affirmed" and certainty_final == "uncertain":
                    polarity_final = "uncertain"

                mapped_concepts = infer_shared_concepts(raw_concept, quote, candidate_concepts)
                canonical_group = mapped_concepts[0]
                phrase_slug = make_slug(quote)

                rows.append({
                    "case_id": rec["case_id"],
                    "row_index": int(rec["row_index"]),
                    "report_mode": rec["report_mode"],
                    "dispersion_true": rec["dispersion_true"],
                    "dispersion_true_high_low": rec["dispersion_true_high_low"],
                    "relapse_true": rec["relapse_true"],
                    "source": source_name,
                    "quote": quote,
                    "quote_norm": normalize_text(quote),
                    "normalized_phrase": normalized_phrase,
                    "phrase_slug": phrase_slug,
                    "raw_concept": raw_concept,
                    "candidate_concepts": ";".join(candidate_concepts),
                    "mapped_concepts": ";".join(mapped_concepts),
                    "canonical_group": canonical_group,
                    "polarity": polarity_final,
                    "certainty": certainty_final,
                    "laterality": laterality,
                    "span_type": span_type,
                    "section": section,
                    "directness": directness,
                    "biological_ambiguity": biological_ambiguity,
                    "mapping_confidence": mapping_confidence,
                    "directly_asserts_tumor": directly_asserts_tumor,
                    "imaging_pattern_only": imaging_pattern_only,
                })

    out = pd.DataFrame(rows)
    if len(out) == 0:
        return out

    out["is_affirmed"] = ((out["polarity"] == "affirmed") & (out["certainty"] == "certain")).astype(int)
    out["is_negated"] = (out["polarity"] == "negated").astype(int)
    out["is_uncertain"] = ((out["certainty"] == "uncertain") | (out["polarity"] == "uncertain")).astype(int)
    out["is_seed"] = (out["source"] == "seed").astype(int)
    out["is_denovo"] = (out["source"] == "denovo").astype(int)
    return out
