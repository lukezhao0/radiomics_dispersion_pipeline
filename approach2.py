#!/usr/bin/env python3
"""
Leakage-aware nested evaluation with pathology-informed MRI lexical refinement.

SUMMARY OF EDITED VERSION
=========================
This edited file keeps the original nested outer/inner evaluation design, but
adds the minimum machinery needed for the requested pathology-informed MRI
refinement project:

1. MRI underperformance audit
   - report length, recoded phrase/concept counts, uncertainty/negation counts,
     concept-category densities, section flags, high/low comparisons, and relapse
     comparisons when relapse labels exist.

2. Shared biological concept ontology
   - explicit concept definitions and regex mapping rules for concepts expressible
     in MRI and pathology language.
   - stable lexicons are still produced, but recoding can also force ontology
     concepts into the feature space so concept-level MRI features can be tested
     even if the original thresholded stable lexicon would have dropped them.

3. Pathology-as-teacher calibration
   - inside each outer-training split only, compute MRI-concept to pathology-
     concept concordance: P(g|m), P(m|g), P(g|not m), delta, lift, odds ratio,
     mutual information, and support counts.
   - convert those concordances into pathology-informed MRI concept weights.
   - apply frozen weights to train/test MRI reports to construct MRI-only weighted
     concept-score matrices. Outer-test pathology is never used for primary
     calibration.

4. Weighted lexicon / concept-score models
   - existing thresholded group/phrase matrices remain available.
   - new dataset keys are added when calibration is enabled:
       * mri_pathcal_weighted
       * optional randomized / mismatched calibration ablations

5. Teacher-student MRI model
   - trains a simple multi-task ridge MRI student inside each outer split using
     MRI features only as inputs, but training targets include true continuous
     dispersion, training-only pathology-teacher predictions, and pathology
     concept outputs.
   - produces held-out MRI-only continuous predictions plus thresholded high/low
     predictions.

6. Interpretation outputs
   - feature coefficient tables are preserved.
   - additional coefficient sign-consistency summaries are written.
   - all calibration, weighting, audit, and methodology documentation files are
     written to disk.

Recommended primary run:

python approach2_3.py \
  --csv-path /Users/lukezhao/projects/onc/PROCESSED_TRIMMED_path_cases_MRI_status.csv \
  --out_dir /Users/lukezhao/projects/onc/sabcs/securegpt_dispersion_pathology_informed_eval \
  --enable-pathology-calibration \
  --enable-teacher-student \
  --modalities mri path combined \
  --representations group_binary group_count group_status phrase_binary \
  --max-api-workers 2 \
  --parallel-modality-workers 2 \
  --ml-n-jobs 1 \
  --yes

python approach2_3.py \
  --csv-path /Users/lukezhao/projects/onc/PROCESSED_TRIMMED_path_cases_MRI_status.csv \
  --out_dir /Users/lukezhao/projects/onc/sabcs/securegpt_dispersion_approach2_noMax \
  --enable-pathology-calibration \
  --enable-teacher-student \
  --modalities mri path combined \
  --representations group_binary group_count group_status phrase_binary \
  --max-api-workers 2 \
  --parallel-modality-workers 2 \
  --ml-n-jobs 1 \
  --yes
  

Optional ablation run:

python feature_discovery_eval_ML.py \
  --csv-path /path/to/cases.csv \
  --out_dir /path/to/out \
  --enable-pathology-calibration \
  --enable-teacher-student \
  --run-calibration-ablations

Important leakage rule:
For the primary calibrated MRI features, every cross-modal reliability metric,
weight, teacher model, and ontology-calibration decision is learned only from the
outer-training cases. The outer-test MRI reports are recoded using frozen rules
and predicted using MRI-derived inputs only.
"""

from __future__ import annotations

import argparse
import contextlib
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
import os
import re
import sys
import time
import traceback
import html
import hashlib
from datetime import datetime
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.base import BaseEstimator, TransformerMixin
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
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    StratifiedKFold,
    StratifiedShuffleSplit,
    cross_val_predict,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR, SVC

from approach2_3_aux import (
    MAX_TOKENS,
    Tee,
    _is_missing_text,
    _selected_report_text,
    _true_dispersion_high_low,
    build_html_report,
    build_user_prompt,
    confirm_cost_estimate_or_exit,
    df_to_html_table,
    estimate_prompt_tokens_from_messages,
    build_chat_messages,
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
    configure_global_api_concurrency,
    extract_subset_records,
    write_extractions,
)

DISPERSION_TRUE_HIGH_THRESHOLD = 85.0
RANDOM_SEED = 17
INNER_CV_MAX_SPLITS = 5
COEF_ZERO_TOL = 1e-8
EPS = 1e-12
DEFAULT_BOOTSTRAP_N = 1000

TARGET_NAME_DISPERSION_SCORE = "dispersion_score"
TARGET_NAME_DISPERSION_HIGH_LOW = "dispersion_high_low"
TARGET_NAME_RELAPSE_STATUS = "relapse_status"

NEGATION_PATTERNS = [
    r"\bno\b",
    r"\bnot\b",
    r"\bwithout\b",
    r"\babsent\b",
    r"\bnegative for\b",
    r"\bfree of\b",
    r"\bneither\b",
    r"\bnone\b",
    r"\bresolved\b",
    r"\bresolution of\b",
]

UNCERTAINTY_PATTERNS = [
    r"\bpossible\b",
    r"\bpossibly\b",
    r"\bprobable\b",
    r"\blikely\b",
    r"\bsuspicious\b",
    r"\bsuspected\b",
    r"\bcannot exclude\b",
    r"\bmay represent\b",
    r"\bindeterminate\b",
    r"\buncertain\b",
    r"\bfavored\b",
    r"\bprobably\b",
    r"\bpresumably\b",
]

# Shared ontology. The concept names are intentionally stable identifiers used
# throughout recoding, reliability analysis, weighting, and interpretation.
SHARED_CONCEPT_ONTOLOGY: Dict[str, Dict[str, Any]] = {
    "extent_span": {
        "definition": "Long span, large extent, or broad area of abnormality/tumor bed/disease.",
        "category": "spatial_morphology_response",
        "mri_examples": "spanning 6 cm; extent of enhancement; large area of NME",
        "path_examples": "residual disease spanning; tumor bed measures; extent of DCIS",
        "patterns": [r"\bextent\b", r"\bspan\b", r"\bspanning\b", r"\bmeasur", r"\blarge area\b", r"\bbroad\b", r"\bdiffuse\b", r"\bextensive\b", r"\bcm\b", r"\bmm\b"],
    },
    "multiplicity": {
        "definition": "Multiple foci/sites suggesting more than one localized residual focus.",
        "category": "spatial_morphology_response",
        "mri_examples": "multiple foci; multifocal enhancement; satellite foci",
        "path_examples": "multiple residual foci; multifocal residual carcinoma",
        "patterns": [r"\bmultifocal\b", r"\bmultiple\b", r"\bseveral\b", r"\bsatellite", r"\badditional foci\b", r"\bseparate foci\b", r"\bsmall foci\b"],
    },
    "multicentricity_separate_sites": {
        "definition": "Disease or abnormality in separate regions/sites/quadrants.",
        "category": "spatial_morphology_response",
        "mri_examples": "multicentric disease; separate sites",
        "path_examples": "separate foci; widely separated residual disease",
        "patterns": [r"\bmulticentric\b", r"\bseparate sites\b", r"\bseparate areas\b", r"\bwidely separated\b", r"\bseparated by\b", r"\bdifferent quadrant"],
    },
    "distribution_linear_segmental_regional": {
        "definition": "Linear, segmental, ductal, or regional distribution pattern.",
        "category": "spatial_morphology_response",
        "mri_examples": "segmental NME; linear enhancement; ductal distribution",
        "path_examples": "ductal distribution; regional spread; geographic disease",
        "patterns": [r"\bsegmental\b", r"\blinear\b", r"\bregional\b", r"\bductal distribution\b", r"\bclumped\b", r"\bgeographic\b"],
    },
    "fragmentation_scattered_patchy_discontinuous": {
        "definition": "Patchy, scattered, discontinuous, or fragmented residual abnormality/disease.",
        "category": "spatial_morphology_response",
        "mri_examples": "patchy enhancement; scattered foci; discontinuous enhancement",
        "path_examples": "scattered microscopic foci; discontinuous residual carcinoma",
        "patterns": [r"\bdiscontinuous\b", r"\bdiscontinuity\b", r"\bpatchy\b", r"\bskip\b", r"\bscattered\b", r"\bfragment", r"\bintermixed\b"],
    },
    "residual_tumor_presence": {
        "definition": "Residual disease/tumor or residual enhancement after neoadjuvant therapy.",
        "category": "residual_disease",
        "mri_examples": "residual enhancement; persistent NME",
        "path_examples": "residual viable carcinoma; residual invasive carcinoma; residual DCIS",
        "patterns": [r"\bresidual\b", r"\bpersistent\b", r"\bviable carcinoma\b", r"\bresidual carcinoma\b", r"\bresidual disease\b", r"\bremaining\b"],
    },
    "non_mass_enhancement": {
        "definition": "MRI non-mass enhancement pattern, including NME abbreviation.",
        "category": "spatial_morphology_response",
        "mri_examples": "non-mass enhancement; NME",
        "path_examples": "not usually literal, may align to broad/ductal/DCIS concepts",
        "patterns": [r"\bnon[- ]mass enhancement\b", r"\bnme\b", r"\bnon mass\b"],
    },
    "invasive_disease": {
        "definition": "Invasive carcinoma/disease component.",
        "category": "pathology_specific",
        "mri_examples": "often indirect, e.g. enhancing mass suspicious for invasive disease",
        "path_examples": "residual invasive carcinoma; invasive ductal carcinoma",
        "patterns": [r"\binvasive\b", r"\binvasion\b", r"\binfiltrating carcinoma\b"],
    },
    "in_situ_disease_dcis": {
        "definition": "DCIS or in-situ component/burden.",
        "category": "pathology_specific",
        "mri_examples": "NME in ductal distribution, often indirect",
        "path_examples": "DCIS; ductal carcinoma in situ; extensive intraductal component",
        "patterns": [r"\bdcis\b", r"\bductal carcinoma in situ\b", r"\bin situ\b", r"\bintraductal\b"],
    },
    "treatment_response": {
        "definition": "Language describing response magnitude after therapy.",
        "category": "spatial_morphology_response",
        "mri_examples": "decreased enhancement; near complete imaging response",
        "path_examples": "treatment effect; residual cellularity; partial response",
        "patterns": [r"\bdecrease\b", r"\bdecreased\b", r"\binterval decrease\b", r"\bshrink", r"\breduced\b", r"\bsmaller\b", r"\bresponse\b", r"\bcomplete response\b", r"\bnear complete\b", r"\bpartial response\b"],
    },
    "treatment_effect_tumor_bed": {
        "definition": "Treatment effect, tumor bed, fibrosis/scarring, or therapy-related changes.",
        "category": "response_treatment_effect",
        "mri_examples": "post-treatment change; treatment-related enhancement",
        "path_examples": "tumor bed; treatment effect; fibrosis; residual cellularity",
        "patterns": [r"\btreatment effect\b", r"\btumou?r bed\b", r"\bfibrosis\b", r"\btherapy[- ]related\b", r"\bpost[- ]treatment\b", r"\bcellularity\b"],
    },
    "localized_compact_residual": {
        "definition": "Single, focal, compact, localized residual abnormality/disease.",
        "category": "spatial_morphology_response",
        "mri_examples": "single residual mass; focal enhancement",
        "path_examples": "single focus; localized residual carcinoma",
        "patterns": [r"\blocalized\b", r"\bcompact\b", r"\bsingle\b", r"\bsolitary\b", r"\bdominant mass\b", r"\bfocal\b", r"\bsingle residual\b", r"\bone focus\b"],
    },
    "diffuse_scattered_residual": {
        "definition": "Diffuse, scattered, broad, or patchy residual disease/enhancement.",
        "category": "spatial_morphology_response",
        "mri_examples": "diffuse residual enhancement; broad NME",
        "path_examples": "scattered residual disease; widely distributed residual tumor",
        "patterns": [r"\bdiffuse residual\b", r"\bpatchy residual\b", r"\bscattered residual\b", r"\bmultiple residual\b", r"\bdiscontinuous residual\b", r"\bwidely distributed\b"],
    },
    "lymphovascular_invasion": {
        "definition": "Lymphovascular invasion.",
        "category": "pathology_specific",
        "mri_examples": "not directly visible in routine report language",
        "path_examples": "lymphovascular invasion; LVI",
        "patterns": [r"\blymphovascular invasion\b", r"\blvi\b"],
    },
    "margin_proximity": {
        "definition": "Margin proximity/positivity, a pathology-localization and residual extent signal.",
        "category": "pathology_specific",
        "mri_examples": "not usually direct; relationship to surgical margins may be absent",
        "path_examples": "margin; mm from margin; close margin; positive margin",
        "patterns": [r"\bmargin\b", r"\bmm from\b", r"\bclose margin\b", r"\bpositive margin\b", r"\bclear margin\b"],
    },
    "benign_or_nonspecific_enhancement": {
        "definition": "MRI enhancement described as nonspecific, probably benign, or background-like.",
        "category": "ambiguity_noise",
        "mri_examples": "background parenchymal enhancement; probably benign; nonspecific enhancement",
        "path_examples": "not direct pathology correlate; used as ambiguity penalty",
        "patterns": [r"\bnonspecific\b", r"\bprobably benign\b", r"\bfavored benign\b", r"\bbackground parenchymal enhancement\b", r"\bbpe\b", r"\bbenign enhancement\b"],
    },
}

CANONICAL_GROUP_PATTERNS = {k: v["patterns"] for k, v in SHARED_CONCEPT_ONTOLOGY.items()}

SPATIAL_MORPH_RESPONSE_GROUPS = {
    k for k, v in SHARED_CONCEPT_ONTOLOGY.items()
    if v["category"] in {"spatial_morphology_response", "residual_disease", "response_treatment_effect"}
}
DISTRIBUTION_GROUPS = {
    "multiplicity",
    "multicentricity_separate_sites",
    "distribution_linear_segmental_regional",
    "fragmentation_scattered_patchy_discontinuous",
    "diffuse_scattered_residual",
    "non_mass_enhancement",
}
AMBIGUITY_GROUPS = {"benign_or_nonspecific_enhancement"}

META_COLS = {
    "case_id",
    "row_index",
    "dispersion_true",
    "dispersion_true_high_low",
    "relapse_true",
    "split_id",
    "split_role",
    "report_mode",
}


# -----------------------------
# Helper classes
# -----------------------------

class LowInfoFeatureFilter(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        min_non_missing: int = 2,
        min_nonzero: int = 1,
        min_unique_non_missing: int = 2,
    ):
        self.min_non_missing = min_non_missing
        self.min_nonzero = min_nonzero
        self.min_unique_non_missing = min_unique_non_missing

    def fit(self, X: Any, y: Any = None) -> "LowInfoFeatureFilter":
        X_df = pd.DataFrame(X).copy()
        self.feature_names_in_ = list(X_df.columns)

        non_missing = X_df.notna().sum(axis=0)
        nonzero = (X_df.fillna(0.0) != 0).sum(axis=0)
        unique_non_missing = X_df.nunique(dropna=True)

        keep = (
            (non_missing >= self.min_non_missing)
            & (nonzero >= self.min_nonzero)
            & (unique_non_missing >= self.min_unique_non_missing)
        )

        if int(keep.sum()) == 0:
            keep = pd.Series(True, index=X_df.columns)

        self.selected_features_ = list(X_df.columns[keep.values])
        self.support_mask_ = keep.values.astype(bool)
        return self

    def transform(self, X: Any) -> pd.DataFrame:
        X_df = pd.DataFrame(X).copy()
        if X_df.shape[1] == len(getattr(self, "feature_names_in_", [])):
            X_df.columns = self.feature_names_in_
        return X_df.loc[:, self.selected_features_]

    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> np.ndarray:
        return np.asarray(self.selected_features_, dtype=object)


@dataclass
class ModelSpec:
    key: str
    task_type: str
    family: str
    estimator_name: str
    scoring: str
    supports_probability: bool
    notes: str


# -----------------------------
# Generic helpers
# -----------------------------

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


def safe_spearman(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    if len(x) < 3 or len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return np.nan, np.nan
    rho, p = spearmanr(x, y)
    return float(rho), float(p)


def safe_pearson(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    if len(x) < 3 or len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return np.nan, np.nan
    r, p = pearsonr(x, y)
    return float(r), float(p)


def rmse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def ensure_case_id(df: pd.DataFrame) -> pd.DataFrame:
    if "case_id" not in df.columns:
        df = df.copy()
        df["case_id"] = [f"row_{i}" for i in range(len(df))]
    return df


def get_target_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = ensure_case_id(df).copy()
    out["row_index"] = list(range(len(out)))
    out["dispersion_true"] = pd.to_numeric(
        out["dispersion_invasive_DCIS_geographic"], errors="coerce"
    )
    out["dispersion_true_high_low"] = out["dispersion_true"].apply(_true_dispersion_high_low)
    out["relapse_true"] = pd.to_numeric(out["relapse"], errors="coerce")
    out = out[out["dispersion_true"].notna() & out["dispersion_true_high_low"].notna()].copy()
    out["dispersion_true_high_low"] = out["dispersion_true_high_low"].astype(int)
    out["row_index"] = out["row_index"].astype(int)
    return out




def _raw_df_with_row_index(raw_df: pd.DataFrame) -> pd.DataFrame:
    out = ensure_case_id(raw_df).copy()
    if "row_index" not in out.columns:
        out["row_index"] = list(range(len(out)))
    return out


def _mri_missing_row_indices(raw_df: pd.DataFrame) -> set:
    raw = _raw_df_with_row_index(raw_df)
    if "preop_MRI_text" not in raw.columns:
        return set()
    return set(raw.loc[raw["preop_MRI_text"].apply(_is_missing_text), "row_index"].astype(int).tolist())


def dataset_requires_mri_report(dataset_key: str) -> bool:
    """Return True for datasets whose features cannot be interpreted without MRI text."""
    dataset_key = str(dataset_key)
    return (
        dataset_key == "mri"
        or dataset_key == "combined"
        or dataset_key.startswith("mri_pathcal")
        or dataset_key.startswith("mri_teacher_student")
    )


def filter_missing_mri_for_dataset(
    dataset_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    dataset_key: str,
    split_id: str,
) -> pd.DataFrame:
    """Drop MRI-missing cases from MRI-derived evaluations.

    Pathology-only evaluations intentionally keep these cases because every case is
    expected to have a pathology report. MRI-only, combined MRI+pathology,
    pathology-calibrated MRI, and teacher-student MRI evaluations are not valid
    without the MRI report, so those rows are removed from both train and test
    portions before model fitting.
    """
    if not dataset_requires_mri_report(dataset_key) or len(dataset_df) == 0:
        return dataset_df

    missing_rows = _mri_missing_row_indices(raw_df)
    if not missing_rows or "row_index" not in dataset_df.columns:
        return dataset_df

    before = len(dataset_df)
    out = dataset_df[~dataset_df["row_index"].astype(int).isin(missing_rows)].copy().reset_index(drop=True)
    skipped = before - len(out)
    if skipped > 0:
        print(
            f"[MISSING_MRI] split={split_id} dataset={dataset_key}: "
            f"skipped {skipped} case rows with missing preop_MRI_text."
        )
    return out

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


# -----------------------------
# Split generation
# -----------------------------

def build_outer_splits(
    y_binary: np.ndarray,
    scheme: str,
    random_seed: int,
    n_repeats: int,
    test_frac: float,
    n_folds: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    indices = np.arange(len(y_binary))
    splits: List[Tuple[np.ndarray, np.ndarray]] = []

    if scheme == "repeated_mc":
        for rep in range(n_repeats):
            rs = random_seed + rep
            splitter = StratifiedShuffleSplit(
                n_splits=1,
                test_size=test_frac,
                random_state=rs,
            )
            for train_idx, test_idx in splitter.split(indices, y_binary):
                splits.append((train_idx, test_idx))
    elif scheme == "stratified_kfold":
        splitter = StratifiedKFold(
            n_splits=n_folds,
            shuffle=True,
            random_state=random_seed,
        )
        for train_idx, test_idx in splitter.split(indices, y_binary):
            splits.append((train_idx, test_idx))
    else:
        raise ValueError(f"Unsupported outer scheme: {scheme}")

    return splits


def build_rediscovery_subsplits(
    case_ids: np.ndarray,
    y_binary: np.ndarray,
    scheme: str,
    random_seed: int,
    n_repeats: int,
    test_frac: float,
    n_folds: int,
) -> List[np.ndarray]:
    indices = np.arange(len(case_ids))
    train_subsets: List[np.ndarray] = []

    if len(indices) < 4 or len(np.unique(y_binary)) < 2:
        return [indices]

    if scheme == "repeated_mc":
        n_repeats = max(1, n_repeats)
        for rep in range(n_repeats):
            rs = random_seed + rep
            splitter = StratifiedShuffleSplit(
                n_splits=1,
                test_size=test_frac,
                random_state=rs,
            )
            for train_idx, _ in splitter.split(indices, y_binary):
                train_subsets.append(train_idx)
    elif scheme == "stratified_kfold":
        n_folds = min(n_folds, len(indices))
        n_folds = max(2, n_folds)
        splitter = StratifiedKFold(
            n_splits=n_folds,
            shuffle=True,
            random_state=random_seed,
        )
        for train_idx, _ in splitter.split(indices, y_binary):
            train_subsets.append(train_idx)
    else:
        raise ValueError(f"Unsupported rediscovery scheme: {scheme}")

    if not train_subsets:
        train_subsets = [indices]
    return train_subsets


# -----------------------------
# Rediscovery and frozen lexicon
# -----------------------------

def build_stable_lexicon_from_training_extractions(
    train_extractions_df: pd.DataFrame,
    train_phrase_df: pd.DataFrame,
    rediscovery_scheme: str,
    rediscovery_repeats: int,
    rediscovery_test_frac: float,
    rediscovery_folds: int,
    stability_threshold: float,
    min_phrase_cases: int,
    min_group_cases: int,
    random_seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    case_meta = (
        train_extractions_df[["case_id", "dispersion_true_high_low"]]
        .drop_duplicates()
        .copy()
    )
    case_ids = case_meta["case_id"].astype(str).values
    y_binary = case_meta["dispersion_true_high_low"].astype(int).values

    rediscovery_subsets = build_rediscovery_subsplits(
        case_ids=case_ids,
        y_binary=y_binary,
        scheme=rediscovery_scheme,
        random_seed=random_seed,
        n_repeats=rediscovery_repeats,
        test_frac=rediscovery_test_frac,
        n_folds=rediscovery_folds,
    )
    n_subsplits = len(rediscovery_subsets)
    print(f"[REDISCOVERY] Built {n_subsplits} inner rediscovery subsets.")

    if len(train_phrase_df) == 0:
        empty_phrase = pd.DataFrame(columns=["phrase_slug", "quote_norm", "canonical_group", "n_rows", "selected_count", "mean_support_cases", "selection_frequency", "stable"])
        empty_group = pd.DataFrame(columns=["canonical_group", "n_rows", "selected_count", "mean_support_cases", "selection_frequency", "stable"])
        return empty_phrase, empty_group, empty_phrase.copy(), empty_group.copy()

    phrase_counts = Counter()
    group_counts = Counter()
    phrase_support_total = Counter()
    group_support_total = Counter()

    phrase_lookup = (
        train_phrase_df.groupby(["phrase_slug", "quote_norm", "canonical_group"])
        .size()
        .reset_index(name="n_rows")
    )
    group_lookup = (
        train_phrase_df.groupby(["canonical_group"])
        .size()
        .reset_index(name="n_rows")
    )

    case_id_arr = np.asarray(case_ids)

    for sub_idx, train_subset_idx in enumerate(rediscovery_subsets, 1):
        subset_case_ids = set(case_id_arr[train_subset_idx].tolist())
        subset_phrases = train_phrase_df[train_phrase_df["case_id"].isin(subset_case_ids)].copy()

        if len(subset_phrases) == 0:
            continue

        phrase_support = (
            subset_phrases.groupby("phrase_slug")["case_id"]
            .nunique()
            .to_dict()
        )
        group_support = (
            subset_phrases.groupby("canonical_group")["case_id"]
            .nunique()
            .to_dict()
        )

        for phrase_slug, support in phrase_support.items():
            phrase_support_total[phrase_slug] += int(support)
            if int(support) >= min_phrase_cases:
                phrase_counts[phrase_slug] += 1

        for group, support in group_support.items():
            group_support_total[group] += int(support)
            if int(support) >= min_group_cases:
                group_counts[group] += 1

        print(
            f"[REDISCOVERY] subset={sub_idx}/{n_subsplits} "
            f"n_cases={len(subset_case_ids)} "
            f"n_phrase_candidates={len(phrase_support)} "
            f"n_group_candidates={len(group_support)}"
        )

    phrase_freq_df = phrase_lookup.copy()
    phrase_freq_df["selected_count"] = phrase_freq_df["phrase_slug"].map(lambda x: phrase_counts.get(x, 0))
    phrase_freq_df["mean_support_cases"] = phrase_freq_df["phrase_slug"].map(
        lambda x: phrase_support_total.get(x, 0) / max(1, n_subsplits)
    )
    phrase_freq_df["selection_frequency"] = phrase_freq_df["selected_count"] / max(1, n_subsplits)
    phrase_freq_df["stable"] = (phrase_freq_df["selection_frequency"] >= stability_threshold).astype(int)
    phrase_freq_df = phrase_freq_df.sort_values(
        ["stable", "selection_frequency", "mean_support_cases", "n_rows"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)

    group_freq_df = group_lookup.copy()
    group_freq_df["selected_count"] = group_freq_df["canonical_group"].map(lambda x: group_counts.get(x, 0))
    group_freq_df["mean_support_cases"] = group_freq_df["canonical_group"].map(
        lambda x: group_support_total.get(x, 0) / max(1, n_subsplits)
    )
    group_freq_df["selection_frequency"] = group_freq_df["selected_count"] / max(1, n_subsplits)
    group_freq_df["stable"] = (group_freq_df["selection_frequency"] >= stability_threshold).astype(int)
    group_freq_df = group_freq_df.sort_values(
        ["stable", "selection_frequency", "mean_support_cases", "n_rows"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)

    stable_phrase_df = phrase_freq_df[phrase_freq_df["stable"] == 1].copy()
    stable_group_df = group_freq_df[group_freq_df["stable"] == 1].copy()

    print(
        f"[REDISCOVERY] Stable lexicon summary: "
        f"n_stable_phrases={len(stable_phrase_df)} "
        f"n_stable_groups={len(stable_group_df)} "
        f"threshold={stability_threshold:.3f}"
    )

    return phrase_freq_df, group_freq_df, stable_phrase_df, stable_group_df


# -----------------------------
# Re-coding with frozen lexicon and ontology groups
# -----------------------------

def _phrase_context_status(report_norm: str, phrase_norm: str) -> Tuple[int, int, int, str]:
    idx = report_norm.find(phrase_norm)
    if idx < 0:
        return 0, 0, 0, ""
    start = max(0, idx - 80)
    end = min(len(report_norm), idx + len(phrase_norm) + 80)
    context = report_norm[start:end]
    if detect_negation(context):
        return 0, 1, 0, context
    if detect_uncertainty(context):
        return 0, 0, 1, context
    return 1, 0, 0, context


def _pattern_group_counts(report_norm: str, patterns: Sequence[str]) -> Tuple[int, int, int, str]:
    present = 0
    negated = 0
    uncertain = 0
    support = ""

    for pat in patterns:
        for match in re.finditer(pat, report_norm):
            start = max(0, match.start() - 80)
            end = min(len(report_norm), match.end() + 80)
            context = report_norm[start:end]
            support = context
            if detect_negation(context):
                negated += 1
            elif detect_uncertainty(context):
                uncertain += 1
            else:
                present += 1

    return present, negated, uncertain, support


def recode_cases_with_frozen_lexicon(
    raw_df: pd.DataFrame,
    outer_case_df: pd.DataFrame,
    report_mode: str,
    stable_phrase_df: pd.DataFrame,
    stable_group_df: pd.DataFrame,
    split_id: str,
    train_case_ids: Sequence[str],
    extra_group_names: Optional[Sequence[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    raw_df = ensure_case_id(raw_df).copy()
    raw_df["row_index"] = list(range(len(raw_df)))

    train_case_id_set = set(str(x) for x in train_case_ids)
    outer_row_indices = set(int(x) for x in outer_case_df["row_index"].tolist())

    phrase_rows: List[Dict[str, Any]] = []
    group_rows: List[Dict[str, Any]] = []

    stable_phrase_records = stable_phrase_df.to_dict("records") if len(stable_phrase_df) else []
    stable_groups = stable_group_df["canonical_group"].astype(str).tolist() if len(stable_group_df) else []
    if extra_group_names:
        stable_groups = sorted(set(stable_groups).union(set(map(str, extra_group_names))))

    outer_raw = raw_df[raw_df["row_index"].isin(outer_row_indices)].copy()

    for _, row in outer_raw.iterrows():
        case_id = str(row["case_id"])
        row_index = int(row["row_index"])
        report_text = _selected_report_text(
            type("TmpCase", (), {
                "preop_mri": row.get("preop_MRI_text", ""),
                "path_report": row.get("path_report_text", ""),
            })(),
            report_mode,
        )
        report_norm = normalize_text(report_text)
        missing = int(_is_missing_text(report_text))
        split_role = "train" if case_id in train_case_id_set else "test"
        y_score = pd.to_numeric(row["dispersion_invasive_DCIS_geographic"], errors="coerce")
        y_highlow = _true_dispersion_high_low(y_score)
        relapse_true = pd.to_numeric(row["relapse"], errors="coerce")

        for phrase_rec in stable_phrase_records:
            phrase_slug = str(phrase_rec["phrase_slug"])
            phrase_norm = str(phrase_rec["quote_norm"])
            canonical_group = str(phrase_rec["canonical_group"])

            if missing:
                present = negated = uncertain = 0
                support_text = ""
            else:
                present, negated, uncertain, support_text = _phrase_context_status(report_norm, phrase_norm)

            phrase_rows.append({
                "case_id": case_id,
                "row_index": row_index,
                "report_mode": report_mode,
                "split_id": split_id,
                "split_role": split_role,
                "dispersion_true": y_score,
                "dispersion_true_high_low": y_highlow,
                "relapse_true": relapse_true,
                "feature_slug": phrase_slug,
                "feature_type": "phrase",
                "feature_group": canonical_group,
                "present": float(present),
                "count": float(present),
                "negated_count": float(negated),
                "uncertain_count": float(uncertain),
                "support_text": support_text,
                "selected_report_missing": missing,
            })

        for group in stable_groups:
            if missing:
                present = negated = uncertain = 0
                support_text = ""
            else:
                present, negated, uncertain, support_text = _pattern_group_counts(
                    report_norm, CANONICAL_GROUP_PATTERNS.get(group, [])
                )

            group_rows.append({
                "case_id": case_id,
                "row_index": row_index,
                "report_mode": report_mode,
                "split_id": split_id,
                "split_role": split_role,
                "dispersion_true": y_score,
                "dispersion_true_high_low": y_highlow,
                "relapse_true": relapse_true,
                "feature_slug": group,
                "feature_type": "group",
                "feature_group": group,
                "present": float(int(present > 0)),
                "count": float(present),
                "negated_count": float(negated),
                "uncertain_count": float(uncertain),
                "support_text": support_text,
                "selected_report_missing": missing,
            })

    return pd.DataFrame(phrase_rows), pd.DataFrame(group_rows)


# -----------------------------
# MRI audit outputs
# -----------------------------

def _report_section_flags(text: str) -> Dict[str, int]:
    t = normalize_text(text)
    section_patterns = {
        "has_findings_section": r"\bfindings\b",
        "has_impression_section": r"\bimpression\b",
        "has_comparison_section": r"\bcomparison\b|\bcompared with\b|\bprior\b",
        "has_response_language": r"\bresponse\b|\bdecreased\b|\bresolved\b|\bpersistent\b|\bresidual\b",
        "has_measurement_language": r"\b\d+(?:\.\d+)?\s*(?:cm|mm)\b",
    }
    return {name: int(bool(re.search(pat, t))) for name, pat in section_patterns.items()}


def compute_mri_audit_table(
    raw_df: pd.DataFrame,
    outer_case_df: pd.DataFrame,
    phrase_prov_df: pd.DataFrame,
    group_prov_df: pd.DataFrame,
    split_id: str,
) -> pd.DataFrame:
    raw_df = ensure_case_id(raw_df).copy()
    raw_df["row_index"] = list(range(len(raw_df)))
    outer_rows = raw_df[raw_df["row_index"].isin(set(outer_case_df["row_index"].astype(int)))].copy()

    phrase_counts = pd.DataFrame()
    if len(phrase_prov_df):
        phrase_counts = phrase_prov_df.groupby(["case_id", "row_index"]).agg(
            n_phrase_features=("feature_slug", "nunique"),
            n_phrase_present=("present", "sum"),
            n_phrase_uncertain=("uncertain_count", "sum"),
            n_phrase_negated=("negated_count", "sum"),
        ).reset_index()

    group_counts = pd.DataFrame()
    if len(group_prov_df):
        tmp = group_prov_df.copy()
        tmp["is_spatial_morph_response"] = tmp["feature_slug"].isin(SPATIAL_MORPH_RESPONSE_GROUPS).astype(int)
        tmp["is_distribution"] = tmp["feature_slug"].isin(DISTRIBUTION_GROUPS).astype(int)
        tmp["is_ambiguity"] = tmp["feature_slug"].isin(AMBIGUITY_GROUPS).astype(int)
        tmp["spatial_count"] = tmp["count"] * tmp["is_spatial_morph_response"]
        tmp["distribution_count"] = tmp["count"] * tmp["is_distribution"]
        tmp["ambiguity_count"] = tmp["count"] * tmp["is_ambiguity"]
        group_counts = tmp.groupby(["case_id", "row_index"]).agg(
            n_group_features=("feature_slug", "nunique"),
            n_group_present=("present", "sum"),
            n_spatial_morph_response_phrases=("spatial_count", "sum"),
            n_distribution_phrases=("distribution_count", "sum"),
            n_ambiguity_phrases=("ambiguity_count", "sum"),
            n_uncertain_group_mentions=("uncertain_count", "sum"),
            n_negated_group_mentions=("negated_count", "sum"),
        ).reset_index()

    rows = []
    for _, row in outer_rows.iterrows():
        report_text = str(row.get("preop_MRI_text", "") or "")
        word_count = len([w for w in re.split(r"\s+", report_text.strip()) if w])
        char_count = len(report_text)
        case_id = str(row["case_id"])
        row_index = int(row["row_index"])
        base = {
            "split_id": split_id,
            "case_id": case_id,
            "row_index": row_index,
            "dispersion_true": pd.to_numeric(row["dispersion_invasive_DCIS_geographic"], errors="coerce"),
            "dispersion_true_high_low": _true_dispersion_high_low(row["dispersion_invasive_DCIS_geographic"]),
            "relapse_true": pd.to_numeric(row.get("relapse", np.nan), errors="coerce"),
            "mri_report_missing": int(_is_missing_text(report_text)),
            "mri_report_chars": char_count,
            "mri_report_words": word_count,
        }
        base.update(_report_section_flags(report_text))
        rows.append(base)

    audit = pd.DataFrame(rows)
    for extra in [phrase_counts, group_counts]:
        if len(extra):
            audit = audit.merge(extra, on=["case_id", "row_index"], how="left")
    count_cols = [c for c in audit.columns if c.startswith("n_") or c.startswith("has_")]
    audit[count_cols] = audit[count_cols].fillna(0.0)
    denom = audit["mri_report_words"].replace(0, np.nan)
    audit["mri_extraction_density"] = audit.get("n_spatial_morph_response_phrases", 0.0) / denom
    audit["distribution_density"] = audit.get("n_distribution_phrases", 0.0) / denom
    audit["uncertainty_density"] = audit.get("n_uncertain_group_mentions", 0.0) / denom
    audit["negation_density"] = audit.get("n_negated_group_mentions", 0.0) / denom
    audit = audit.fillna({
        "mri_extraction_density": 0.0,
        "distribution_density": 0.0,
        "uncertainty_density": 0.0,
        "negation_density": 0.0,
    })
    return audit


def summarize_audit_by_groups(audit_df: pd.DataFrame) -> pd.DataFrame:
    if len(audit_df) == 0:
        return pd.DataFrame()
    metrics = [
        "mri_report_words",
        "n_phrase_present",
        "n_group_present",
        "n_spatial_morph_response_phrases",
        "n_distribution_phrases",
        "n_uncertain_group_mentions",
        "n_negated_group_mentions",
        "mri_extraction_density",
        "distribution_density",
        "uncertainty_density",
    ]
    rows = []
    for label_col in ["dispersion_true_high_low", "relapse_true"]:
        if label_col not in audit_df.columns:
            continue
        sub_df = audit_df[audit_df[label_col].notna()].copy()
        if len(sub_df) == 0 or sub_df[label_col].nunique() < 2:
            continue
        for label_value, sub in sub_df.groupby(label_col):
            rec = {"comparison": label_col, "group_value": label_value, "n_cases": len(sub)}
            for m in metrics:
                if m in sub.columns:
                    rec[f"{m}_mean"] = float(pd.to_numeric(sub[m], errors="coerce").mean())
                    rec[f"{m}_median"] = float(pd.to_numeric(sub[m], errors="coerce").median())
            rows.append(rec)
    return pd.DataFrame(rows)


# -----------------------------
# Feature matrices
# -----------------------------

def _base_case_table(outer_case_df: pd.DataFrame, split_id: str, report_mode: str) -> pd.DataFrame:
    out = outer_case_df[["case_id", "row_index", "dispersion_true", "dispersion_true_high_low", "relapse_true"]].copy()
    out["split_id"] = split_id
    out["report_mode"] = report_mode
    return out.drop_duplicates().reset_index(drop=True)


def build_group_feature_matrix(group_prov_df: pd.DataFrame, outer_case_df: pd.DataFrame, split_id: str, report_mode: str) -> pd.DataFrame:
    base = _base_case_table(outer_case_df, split_id, report_mode)

    if len(group_prov_df) == 0:
        return base

    wide_parts = []
    for value_col, suffix in [
        ("present", "present"),
        ("count", "count"),
        ("negated_count", "negated_count"),
        ("uncertain_count", "uncertain_count"),
    ]:
        piv = (
            group_prov_df.pivot_table(
                index=["case_id", "row_index"],
                columns="feature_slug",
                values=value_col,
                aggfunc="sum",
                fill_value=0.0,
            )
            .rename(columns=lambda c: f"grp__{c}__{suffix}")
            .reset_index()
        )
        wide_parts.append(piv)

    out = base.copy()
    for part in wide_parts:
        out = out.merge(part, on=["case_id", "row_index"], how="left")

    out = out.fillna(0.0)
    return out


def build_phrase_feature_matrix(phrase_prov_df: pd.DataFrame, outer_case_df: pd.DataFrame, split_id: str, report_mode: str) -> pd.DataFrame:
    base = _base_case_table(outer_case_df, split_id, report_mode)

    if len(phrase_prov_df) == 0:
        return base

    piv = (
        phrase_prov_df.pivot_table(
            index=["case_id", "row_index"],
            columns="feature_slug",
            values="present",
            aggfunc="max",
            fill_value=0.0,
        )
        .rename(columns=lambda c: f"phr__{c}__present")
        .reset_index()
    )
    out = base.merge(piv, on=["case_id", "row_index"], how="left")
    out = out.fillna(0.0)
    return out


def get_representation_matrix(df: pd.DataFrame, representation: str) -> Tuple[pd.DataFrame, List[str]]:
    df = df.copy()
    if representation == "group_binary":
        cols = [c for c in df.columns if c.startswith("grp__") and c.endswith("__present")]
    elif representation == "group_count":
        cols = [c for c in df.columns if c.startswith("grp__") and c.endswith("__count") and "__negated_" not in c and "__uncertain_" not in c]
    elif representation == "group_status":
        cols = [c for c in df.columns if c.startswith("grp__") and (
            c.endswith("__present") or c.endswith("__negated_count") or c.endswith("__uncertain_count")
        )]
    elif representation == "phrase_binary":
        cols = [c for c in df.columns if c.startswith("phr__") and c.endswith("__present")]
    elif representation in {"weighted_concept_score", "weighted_plus_group_status"}:
        cols = [c for c in df.columns if c.startswith("wgrp__")]
        if representation == "weighted_plus_group_status":
            cols += [c for c in df.columns if c.startswith("grp__") and (
                c.endswith("__present") or c.endswith("__negated_count") or c.endswith("__uncertain_count")
            )]
    else:
        raise ValueError(f"Unsupported representation: {representation}")

    cols = sorted(set(cols))
    X = df[cols].copy() if cols else pd.DataFrame(index=df.index)
    return X, cols


def merge_modalities_early_fusion(
    mri_df: pd.DataFrame,
    path_df: pd.DataFrame,
    representation: str,
) -> Tuple[pd.DataFrame, List[str]]:
    mri_X, mri_cols = get_representation_matrix(mri_df, representation)
    path_X, path_cols = get_representation_matrix(path_df, representation)

    mri_meta = mri_df[["case_id", "row_index", "dispersion_true", "dispersion_true_high_low", "relapse_true"]].copy()
    path_meta = path_df[["case_id", "row_index"]].copy()

    out = mri_meta.merge(path_meta, on=["case_id", "row_index"], how="outer")

    if len(mri_cols):
        tmp = pd.concat([mri_df[["case_id", "row_index"]], mri_X], axis=1)
        tmp = tmp.rename(columns={c: f"mri__{c}" for c in mri_cols})
        out = out.merge(tmp, on=["case_id", "row_index"], how="left")

    if len(path_cols):
        tmp = pd.concat([path_df[["case_id", "row_index"]], path_X], axis=1)
        tmp = tmp.rename(columns={c: f"path__{c}" for c in path_cols})
        out = out.merge(tmp, on=["case_id", "row_index"], how="left")

    out = out.fillna(0.0)
    feature_cols = [c for c in out.columns if c not in META_COLS]
    return out, feature_cols


# -----------------------------
# Cross-modal calibration and weighted concept features
# -----------------------------

def _group_presence_cols(df: pd.DataFrame) -> Dict[str, str]:
    out = {}
    for c in df.columns:
        if c.startswith("grp__") and c.endswith("__present"):
            concept = c[len("grp__"):-len("__present")]
            out[concept] = c
    return out


def _group_status_value(df: pd.DataFrame, concept: str, uncertain_value: float, negated_value: float) -> pd.Series:
    present = df.get(f"grp__{concept}__present", pd.Series(0.0, index=df.index)).astype(float)
    uncertain = df.get(f"grp__{concept}__uncertain_count", pd.Series(0.0, index=df.index)).astype(float)
    negated = df.get(f"grp__{concept}__negated_count", pd.Series(0.0, index=df.index)).astype(float)
    return present + uncertain_value * (uncertain > 0).astype(float) + negated_value * (negated > 0).astype(float)


def compute_cross_modal_reliability(
    mri_group_matrix_df: pd.DataFrame,
    path_group_matrix_df: pd.DataFrame,
    train_case_ids: Sequence[str],
    split_id: str,
    smoothing: float = 0.5,
) -> pd.DataFrame:
    train_case_ids = set(map(str, train_case_ids))
    mri = mri_group_matrix_df[mri_group_matrix_df["case_id"].astype(str).isin(train_case_ids)].copy()
    path = path_group_matrix_df[path_group_matrix_df["case_id"].astype(str).isin(train_case_ids)].copy()
    merged = mri[["case_id", "row_index"] + list(_group_presence_cols(mri).values())].merge(
        path[["case_id", "row_index"] + list(_group_presence_cols(path).values())],
        on=["case_id", "row_index"],
        suffixes=("__mri", "__path"),
        how="inner",
    )
    rows: List[Dict[str, Any]] = []
    if len(merged) == 0:
        return pd.DataFrame(rows)

    mri_cols = [c for c in merged.columns if c.startswith("grp__") and c.endswith("__present__mri")]
    path_cols = [c for c in merged.columns if c.startswith("grp__") and c.endswith("__present__path")]

    for m_col in mri_cols:
        m = m_col[len("grp__"):-len("__present__mri")]
        m_vec = (pd.to_numeric(merged[m_col], errors="coerce").fillna(0.0).values > 0).astype(int)
        for g_col in path_cols:
            g = g_col[len("grp__"):-len("__present__path")]
            g_vec = (pd.to_numeric(merged[g_col], errors="coerce").fillna(0.0).values > 0).astype(int)
            n = len(m_vec)
            n_m = int(m_vec.sum())
            n_not_m = int((1 - m_vec).sum())
            n_g = int(g_vec.sum())
            n_mg = int(((m_vec == 1) & (g_vec == 1)).sum())
            n_notm_g = int(((m_vec == 0) & (g_vec == 1)).sum())
            n_m_notg = int(((m_vec == 1) & (g_vec == 0)).sum())
            n_notm_notg = int(((m_vec == 0) & (g_vec == 0)).sum())

            p_g = (n_g + smoothing) / (n + 2 * smoothing)
            p_g_given_m = (n_mg + smoothing) / (n_m + 2 * smoothing) if n_m > 0 else np.nan
            p_m_given_g = (n_mg + smoothing) / (n_g + 2 * smoothing) if n_g > 0 else np.nan
            p_g_given_not_m = (n_notm_g + smoothing) / (n_not_m + 2 * smoothing) if n_not_m > 0 else np.nan
            delta = p_g_given_m - p_g_given_not_m if np.isfinite(p_g_given_m) and np.isfinite(p_g_given_not_m) else np.nan
            lift = p_g_given_m / max(p_g, EPS) if np.isfinite(p_g_given_m) else np.nan
            odds_ratio = ((n_mg + smoothing) * (n_notm_notg + smoothing)) / max((n_m_notg + smoothing) * (n_notm_g + smoothing), EPS)
            mi = mutual_info_score(m_vec, g_vec) if len(np.unique(m_vec)) > 1 and len(np.unique(g_vec)) > 1 else 0.0

            rows.append({
                "split_id": split_id,
                "mri_concept": m,
                "path_concept": g,
                "n_train_pairs": n,
                "n_mri_present": n_m,
                "n_path_present": n_g,
                "n_both_present": n_mg,
                "p_path": p_g,
                "p_path_given_mri": p_g_given_m,
                "p_mri_given_path": p_m_given_g,
                "p_path_given_not_mri": p_g_given_not_m,
                "delta_p_path_given_mri": delta,
                "lift": lift,
                "odds_ratio": odds_ratio,
                "mutual_information": float(mi),
            })
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(["delta_p_path_given_mri", "lift", "n_both_present"], ascending=[False, False, False]).reset_index(drop=True)
    return out


def compute_weighted_mri_lexicon(
    reliability_df: pd.DataFrame,
    mri_group_freq_df: pd.DataFrame,
    mri_group_matrix_df: pd.DataFrame,
    train_case_ids: Sequence[str],
    y_train_continuous: pd.Series,
    split_id: str,
    min_selection_frequency: float,
    reliability_power: float,
    stability_power: float,
    association_power: float,
) -> pd.DataFrame:
    train_case_ids = set(map(str, train_case_ids))
    mri_train = mri_group_matrix_df[mri_group_matrix_df["case_id"].astype(str).isin(train_case_ids)].copy()
    y_map = pd.Series(y_train_continuous.values, index=list(map(str, y_train_continuous.index)))

    stability = defaultdict(float)
    if len(mri_group_freq_df):
        for _, r in mri_group_freq_df.iterrows():
            stability[str(r["canonical_group"])] = float(r.get("selection_frequency", 0.0))

    presence_cols = _group_presence_cols(mri_train)
    rows = []
    for concept, col in presence_cols.items():
        if concept == "other_candidate_feature":
            continue
        x = pd.to_numeric(mri_train[col], errors="coerce").fillna(0.0).values
        prevalence = float(np.mean(x > 0)) if len(x) else 0.0
        if prevalence <= 0:
            continue
        stab = max(stability.get(concept, 0.0), min_selection_frequency)
        if stab < min_selection_frequency:
            continue

        sub_rel = reliability_df[reliability_df["mri_concept"] == concept].copy() if len(reliability_df) else pd.DataFrame()
        if len(sub_rel):
            best = sub_rel.sort_values(["delta_p_path_given_mri", "lift", "n_both_present"], ascending=[False, False, False]).iloc[0]
            best_path_concept = str(best["path_concept"])
            rel_score = float(max(0.0, best.get("delta_p_path_given_mri", 0.0)))
            lift_score = float(max(0.0, min(best.get("lift", 1.0), 5.0) / 5.0))
            mi_score = float(max(0.0, best.get("mutual_information", 0.0)))
            concordance_score = 0.70 * rel_score + 0.20 * lift_score + 0.10 * mi_score
        else:
            best_path_concept = "none"
            rel_score = 0.0
            lift_score = 0.0
            mi_score = 0.0
            concordance_score = 0.0

        # Training-only univariate association with continuous dispersion.
        try:
            y = pd.to_numeric(mri_train["dispersion_true"], errors="coerce").values
            rho, _ = safe_spearman(x.astype(float), y.astype(float))
            assoc = abs(rho) if np.isfinite(rho) else 0.0
        except Exception:
            assoc = 0.0

        ambiguity_penalty = 0.55 if concept in AMBIGUITY_GROUPS else 1.0
        prevalence_penalty = min(1.0, prevalence / 0.05) if prevalence < 0.05 else 1.0
        raw_weight = (
            (max(stab, EPS) ** stability_power)
            * (max(concordance_score, EPS) ** reliability_power)
            * ((1.0 + assoc) ** association_power)
            * ambiguity_penalty
            * prevalence_penalty
        )
        rows.append({
            "split_id": split_id,
            "mri_concept": concept,
            "best_path_concept": best_path_concept,
            "weight": float(raw_weight),
            "selection_frequency": float(stab),
            "mri_prevalence_train": prevalence,
            "path_concordance_score": float(concordance_score),
            "delta_component": rel_score,
            "lift_component": lift_score,
            "mutual_information_component": mi_score,
            "abs_spearman_with_dispersion": float(assoc),
            "ambiguity_penalty": ambiguity_penalty,
            "weight_formula": "stability^stability_power * concordance^reliability_power * (1+association)^association_power * ambiguity_penalty * prevalence_penalty",
        })
    out = pd.DataFrame(rows)
    if len(out):
        max_w = float(out["weight"].max())
        if max_w > 0:
            out["weight_normalized"] = out["weight"] / max_w
        else:
            out["weight_normalized"] = 0.0
        out = out.sort_values(["weight_normalized", "path_concordance_score", "mri_prevalence_train"], ascending=[False, False, False]).reset_index(drop=True)
    return out


def build_weighted_mri_concept_score_matrix(
    mri_group_matrix_df: pd.DataFrame,
    weighted_lexicon_df: pd.DataFrame,
    uncertain_value: float,
    negated_value: float,
    split_id: str,
) -> pd.DataFrame:
    base = mri_group_matrix_df[["case_id", "row_index", "dispersion_true", "dispersion_true_high_low", "relapse_true", "split_id", "report_mode"]].copy()
    out = base.copy()
    if len(weighted_lexicon_df) == 0:
        return out
    for _, r in weighted_lexicon_df.iterrows():
        concept = str(r["mri_concept"])
        w = float(r.get("weight_normalized", r.get("weight", 0.0)))
        status = _group_status_value(mri_group_matrix_df, concept, uncertain_value, negated_value)
        out[f"wgrp__{concept}__pathcal_score"] = status.values * w
        out[f"wgrp__{concept}__status_unweighted"] = status.values
    out = out.fillna(0.0)
    return out


def randomized_or_mismatched_path_matrix(path_group_matrix_df: pd.DataFrame, train_case_ids: Sequence[str], mode: str, random_seed: int) -> pd.DataFrame:
    rng = np.random.RandomState(random_seed)
    out = path_group_matrix_df.copy()
    train_mask = out["case_id"].astype(str).isin(set(map(str, train_case_ids)))
    feature_cols = [c for c in out.columns if c.startswith("grp__")]
    if not feature_cols or train_mask.sum() <= 1:
        return out
    if mode == "randomized_labels":
        for c in feature_cols:
            vals = out.loc[train_mask, c].values.copy()
            rng.shuffle(vals)
            out.loc[train_mask, c] = vals
    elif mode == "mismatched_pairing":
        vals_df = out.loc[train_mask, feature_cols].sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
        out.loc[train_mask, feature_cols] = vals_df.values
    else:
        raise ValueError(mode)
    return out


# -----------------------------
# Model builders
# -----------------------------

def make_inner_cv_classification(y_train: np.ndarray, random_seed: int) -> StratifiedKFold:
    class_counts = pd.Series(y_train).value_counts()
    max_splits = int(class_counts.min()) if len(class_counts) else 2
    n_splits = max(2, min(INNER_CV_MAX_SPLITS, max_splits))
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)


def make_inner_cv_regression(n_train: int, random_seed: int) -> KFold:
    n_splits = max(2, min(INNER_CV_MAX_SPLITS, n_train))
    return KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)


def build_pipeline_and_grid(spec: ModelSpec, n_features: int, n_train: int) -> Tuple[Pipeline, List[Dict[str, Any]]]:
    base_steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("lowinfo", LowInfoFeatureFilter()),
        ("scaler", StandardScaler()),
    ]

    if spec.key == "ridge_logistic":
        model = LogisticRegression(
            penalty="l2",
            solver="liblinear",
            class_weight="balanced",
            max_iter=5000,
        )
        grid = [{"model__C": [0.01, 0.1, 1.0, 10.0, 100.0]}]
    elif spec.key == "elasticnet_logistic":
        model = LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            class_weight="balanced",
            max_iter=5000,
        )
        grid = [{"model__C": [0.01, 0.1, 1.0, 10.0], "model__l1_ratio": [0.1, 0.5, 0.9]}]
    elif spec.key == "linear_svm":
        model = SVC(
            kernel="linear",
            probability=True,
            class_weight="balanced",
        )
        grid = [{"model__C": [0.01, 0.1, 1.0, 10.0, 100.0]}]
    elif spec.key == "ridge_regression":
        model = Ridge()
        grid = [{"model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]}]
    elif spec.key == "elasticnet_regression":
        model = ElasticNet(max_iter=5000)
        grid = [{"model__alpha": [0.001, 0.01, 0.1, 1.0], "model__l1_ratio": [0.1, 0.5, 0.9]}]
    elif spec.key == "pls_regression":
        max_comp = max(1, min(10, n_features, max(1, n_train - 1)))
        model = PLSRegression()
        grid = [{"model__n_components": list(range(1, max_comp + 1))}]
    elif spec.key == "linear_svr":
        model = LinearSVR(max_iter=10000)
        grid = [{"model__C": [0.01, 0.1, 1.0, 10.0], "model__epsilon": [0.01, 0.1, 1.0]}]
    elif spec.key == "huber_regression":
        model = HuberRegressor(max_iter=1000)
        grid = [{"model__alpha": [1e-5, 1e-4, 1e-3], "model__epsilon": [1.1, 1.35, 1.6]}]
    else:
        raise ValueError(f"Unsupported model spec: {spec.key}")

    pipe = Pipeline(base_steps + [("model", model)])
    return pipe, grid


def get_model_specs() -> List[ModelSpec]:
    return [
        ModelSpec("ridge_logistic", "classification", "ridge_logistic", "LogisticRegression(L2)", "roc_auc", True, "Primary classification baseline."),
        ModelSpec("elasticnet_logistic", "classification", "elasticnet_logistic", "LogisticRegression(ElasticNet)", "roc_auc", True, "Secondary sparse classification model."),
        ModelSpec("linear_svm", "classification", "linear_svm", "SVC(kernel=linear)", "roc_auc", True, "Linear SVM sensitivity analysis."),
        ModelSpec("ridge_regression", "regression", "ridge_regression", "Ridge", "neg_mean_absolute_error", False, "Primary regression baseline."),
        ModelSpec("elasticnet_regression", "regression", "elasticnet_regression", "ElasticNet", "neg_mean_absolute_error", False, "Sparse linear regression sensitivity."),
        ModelSpec("pls_regression", "regression", "pls_regression", "PLSRegression", "neg_mean_absolute_error", False, "Low-rank correlated-feature regression."),
        ModelSpec("linear_svr", "regression", "linear_svr", "LinearSVR", "neg_mean_absolute_error", False, "Linear SVR sensitivity analysis."),
        ModelSpec("huber_regression", "regression", "huber_regression", "HuberRegressor", "neg_mean_absolute_error", False, "Robust regression for heavy tails/outliers."),
    ]


# -----------------------------
# Metrics and coefficients
# -----------------------------

def calibration_intercept_slope(y_true: np.ndarray, prob: np.ndarray) -> Tuple[float, float]:
    prob = np.clip(np.asarray(prob, dtype=float), 1e-6, 1 - 1e-6)
    logits = np.log(prob / (1.0 - prob)).reshape(-1, 1)
    y_true = np.asarray(y_true, dtype=int)
    if len(np.unique(y_true)) < 2:
        return np.nan, np.nan
    try:
        model = LogisticRegression(C=1e6, solver="lbfgs")
        model.fit(logits, y_true)
        return float(model.intercept_[0]), float(model.coef_[0][0])
    except Exception:
        return np.nan, np.nan


def classification_metrics(pred_df: pd.DataFrame) -> Dict[str, Any]:
    df = pred_df.copy()
    df = df[df["y_true"].notna() & df["y_prob"].notna() & df["y_pred"].notna()].copy()

    out = {
        "n": len(df),
        "prevalence": np.nan,
        "auprc_no_skill_baseline": np.nan,
        "positive_prediction_rate": np.nan,
        "auroc": np.nan,
        "auprc": np.nan,
        "brier": np.nan,
        "accuracy": np.nan,
        "balanced_accuracy": np.nan,
        "f1": np.nan,
        "precision": np.nan,
        "precision_ppv": np.nan,
        "recall_sensitivity": np.nan,
        "specificity": np.nan,
        "npv": np.nan,
        "tn": np.nan,
        "fp": np.nan,
        "fn": np.nan,
        "tp": np.nan,
        "calibration_intercept": np.nan,
        "calibration_slope": np.nan,
    }
    if len(df) == 0:
        return out

    y_true = df["y_true"].astype(int).values
    prob = np.clip(df["y_prob"].astype(float).values, 0.0, 1.0)
    y_pred = df["y_pred"].astype(int).values

    out["prevalence"] = float(np.mean(y_true))
    out["auprc_no_skill_baseline"] = out["prevalence"]
    out["positive_prediction_rate"] = float(np.mean(y_pred))
    out["brier"] = float(brier_score_loss(y_true, prob))
    out["accuracy"] = float(accuracy_score(y_true, y_pred))
    out["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    out["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    out["precision_ppv"] = out["precision"]
    out["recall_sensitivity"] = float(recall_score(y_true, y_pred, zero_division=0))

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out.update({"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})
    out["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else np.nan
    out["npv"] = float(tn / (tn + fn)) if (tn + fn) > 0 else np.nan
    if pd.notna(out["recall_sensitivity"]) and pd.notna(out["specificity"]):
        out["balanced_accuracy"] = float((out["recall_sensitivity"] + out["specificity"]) / 2.0)

    if len(np.unique(y_true)) >= 2:
        out["auroc"] = float(roc_auc_score(y_true, prob))
        out["auprc"] = float(average_precision_score(y_true, prob))
        ci, cs = calibration_intercept_slope(y_true, prob)
        out["calibration_intercept"] = ci
        out["calibration_slope"] = cs
    return out


def regression_metrics(pred_df: pd.DataFrame) -> Dict[str, Any]:
    y_true = pred_df["y_true"].astype(float).values
    y_pred = pred_df["y_pred_value"].astype(float).values
    spearman_rho, _ = safe_spearman(y_true, y_pred)
    pearson_r, _ = safe_pearson(y_true, y_pred)
    return {
        "n": len(pred_df),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse(y_true, y_pred),
        "r2": float(r2_score(y_true, y_pred)),
        "spearman_rho": spearman_rho,
        "pearson_r": pearson_r,
    }




def model_target_specs_for_model(spec: ModelSpec) -> List[Dict[str, str]]:
    if spec.task_type == "regression":
        return [{
            "target_name": TARGET_NAME_DISPERSION_SCORE,
            "target_col": "dispersion_true",
            "task_type": "regression",
        }]
    if spec.task_type == "classification":
        return [
            {
                "target_name": TARGET_NAME_DISPERSION_HIGH_LOW,
                "target_col": "dispersion_true_high_low",
                "task_type": "classification",
            },
            {
                "target_name": TARGET_NAME_RELAPSE_STATUS,
                "target_col": "relapse_true",
                "task_type": "classification",
            },
        ]
    raise ValueError(f"Unsupported task_type: {spec.task_type}")


def prepare_task_frames(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str,
    task_type: str,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    train = train_df.copy()
    test = test_df.copy()
    train = train[train[target_col].notna()].copy()
    test = test[test[target_col].notna()].copy()

    X_train = train[list(feature_cols)].copy()
    X_test = test[list(feature_cols)].copy()
    if task_type == "classification":
        y_train = train[target_col].astype(int)
        y_test = test[target_col].astype(int)
    else:
        y_train = train[target_col].astype(float)
        y_test = test[target_col].astype(float)
    return X_train, y_train, X_test, y_test


def annotate_coefficient_table(
    coef_df: pd.DataFrame,
    X_train: pd.DataFrame,
    X_test: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Add lightweight interpretation metadata to fitted coefficient tables."""
    if coef_df is None or len(coef_df) == 0 or "feature" not in coef_df.columns:
        return coef_df if coef_df is not None else pd.DataFrame()

    out = coef_df.copy()
    X_train_df = pd.DataFrame(X_train).copy()
    X_test_df = pd.DataFrame(X_test).copy() if X_test is not None else pd.DataFrame(columns=X_train_df.columns)

    train_prev = {}
    test_prev = {}
    train_mean = {}
    for col in X_train_df.columns:
        vals = pd.to_numeric(X_train_df[col], errors="coerce").fillna(0.0)
        train_prev[str(col)] = float((vals != 0).mean()) if len(vals) else np.nan
        train_mean[str(col)] = float(vals.mean()) if len(vals) else np.nan
    for col in X_test_df.columns:
        vals = pd.to_numeric(X_test_df[col], errors="coerce").fillna(0.0)
        test_prev[str(col)] = float((vals != 0).mean()) if len(vals) else np.nan

    def _feature_modality(feature: str) -> str:
        feature = str(feature)
        if feature.startswith("mri__") or feature.startswith("weighted__") or feature.startswith("mri_"):
            return "mri"
        if feature.startswith("path__") or feature.startswith("path_"):
            return "pathology"
        return "unknown"

    def _ontology_concept(feature: str) -> str:
        f = str(feature)
        for prefix in ["mri__group__", "path__group__", "weighted__", "group__", "group_status__", "group_count__", "group_binary__", "phrase__", "mri__", "path__"]:
            if f.startswith(prefix):
                f = f[len(prefix):]
        f = re.sub(r"^(present|count|negated_count|uncertain_count)__", "", f)
        for concept in SHARED_CONCEPT_ONTOLOGY:
            if concept in f:
                return concept
        return f

    out["feature_prevalence_train"] = out["feature"].astype(str).map(train_prev)
    out["feature_prevalence_test"] = out["feature"].astype(str).map(test_prev)
    out["feature_mean_train"] = out["feature"].astype(str).map(train_mean)
    out["feature_modality"] = out["feature"].astype(str).map(_feature_modality)
    out["ontology_concept"] = out["feature"].astype(str).map(_ontology_concept)
    out["coef_sign"] = np.where(out["coef"].astype(float) > COEF_ZERO_TOL, "positive", np.where(out["coef"].astype(float) < -COEF_ZERO_TOL, "negative", "zero"))
    return out


def should_skip_model_fit(
    y_train: pd.Series,
    y_test: pd.Series,
    task_type: str,
    split_id: str,
    dataset_key: str,
    representation: str,
    model_key: str,
    target_name: str,
) -> bool:
    if len(y_train) == 0 or len(y_test) == 0:
        print(
            f"[SKIP] split={split_id} dataset={dataset_key} representation={representation} "
            f"model={model_key} target={target_name}: empty train/test after target and MRI-missing filtering."
        )
        return True
    if task_type == "classification":
        counts = pd.Series(y_train).value_counts()
        if len(counts) < 2:
            print(
                f"[SKIP] split={split_id} dataset={dataset_key} representation={representation} "
                f"model={model_key} target={target_name}: training labels contain only one class."
            )
            return True
        if int(counts.min()) < 2:
            print(
                f"[SKIP] split={split_id} dataset={dataset_key} representation={representation} "
                f"model={model_key} target={target_name}: minority class has <2 training cases, "
                f"so inner StratifiedKFold would be invalid. class_counts={counts.to_dict()}"
            )
            return True
    else:
        if len(y_train) < 3 or len(np.unique(y_train.astype(float).values)) < 2:
            print(
                f"[SKIP] split={split_id} dataset={dataset_key} representation={representation} "
                f"model={model_key} target={target_name}: insufficient continuous target variation."
            )
            return True
    return False


def add_bootstrap_metric_cis(
    base_metrics: Dict[str, Any],
    pred_df: pd.DataFrame,
    task_type: str,
    n_bootstrap: int,
    random_seed: int,
) -> Dict[str, Any]:
    """Add simple case-level bootstrap CIs for aggregate held-out metrics."""
    if n_bootstrap <= 0 or len(pred_df) < 2:
        return base_metrics

    rng = np.random.default_rng(int(random_seed))
    metric_names = [
        "auroc", "auprc", "brier", "accuracy", "balanced_accuracy", "f1", "precision",
        "precision_ppv", "recall_sensitivity", "specificity", "npv",
    ] if task_type == "classification" else ["mae", "rmse", "r2", "spearman_rho", "pearson_r"]
    samples: Dict[str, List[float]] = {m: [] for m in metric_names}
    df = pred_df.reset_index(drop=True).copy()

    for _ in range(int(n_bootstrap)):
        idx = rng.integers(0, len(df), len(df))
        boot = df.iloc[idx].copy()
        try:
            m = classification_metrics(boot) if task_type == "classification" else regression_metrics(boot)
        except Exception:
            continue
        for name in metric_names:
            val = m.get(name, np.nan)
            if pd.notna(val):
                samples[name].append(float(val))

    out = dict(base_metrics)
    for name, vals in samples.items():
        if vals:
            out[f"{name}_ci_low"] = float(np.percentile(vals, 2.5))
            out[f"{name}_ci_high"] = float(np.percentile(vals, 97.5))
        else:
            out[f"{name}_ci_low"] = np.nan
            out[f"{name}_ci_high"] = np.nan
    out["bootstrap_n"] = int(n_bootstrap)
    return out

def extract_fitted_feature_coefficients(best_estimator: Pipeline, original_feature_names: List[str]) -> pd.DataFrame:
    try:
        selected_features = list(
            best_estimator.named_steps["lowinfo"].get_feature_names_out()
        )
    except Exception:
        selected_features = list(original_feature_names)

    model = best_estimator.named_steps["model"]
    coef = None

    if hasattr(model, "coef_"):
        coef = np.asarray(model.coef_).reshape(-1)
    elif hasattr(model, "feature_importances_"):
        coef = np.asarray(model.feature_importances_).reshape(-1)

    if coef is None:
        return pd.DataFrame(columns=["feature", "coef", "abs_coef"])

    n = min(len(selected_features), len(coef))
    out = pd.DataFrame({
        "feature": selected_features[:n],
        "coef": coef[:n],
    })
    out["abs_coef"] = out["coef"].abs()
    return out


# -----------------------------
# Outer-loop model fitting
# -----------------------------

def fit_one_outer_model(
    spec: ModelSpec,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    split_random_seed: int,
    ml_n_jobs: int,
) -> Tuple[pd.DataFrame, Dict[str, Any], pd.DataFrame]:
    X_train = pd.DataFrame(X_train).copy()
    X_test = pd.DataFrame(X_test).copy()
    feature_names = list(X_train.columns)

    pipe, param_grid = build_pipeline_and_grid(spec, n_features=X_train.shape[1], n_train=len(X_train))
    if spec.task_type == "classification":
        inner_cv = make_inner_cv_classification(y_train.values, split_random_seed)
    else:
        inner_cv = make_inner_cv_regression(len(X_train), split_random_seed)

    from joblib import parallel_backend
    search = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring=spec.scoring,
        cv=inner_cv,
        refit=True,
        n_jobs=ml_n_jobs,
        error_score=np.nan,
    )
    try:
        with parallel_backend("threading", n_jobs=ml_n_jobs):
            search.fit(X_train, y_train)
    except Exception as e:
        print(
            f"[WARN] Threaded GridSearchCV failed for model; "
            f"falling back to serial fit. Error={e}"
        )
        search = GridSearchCV(
            estimator=pipe,
            param_grid=param_grid,
            scoring=spec.scoring,
            cv=inner_cv,
            refit=True,
            n_jobs=1,
            error_score=np.nan,
        )
        search.fit(X_train, y_train)

    best_estimator = search.best_estimator_
    best_params = dict(search.best_params_)
    best_score = float(search.best_score_) if search.best_score_ is not None else np.nan
    coef_df = extract_fitted_feature_coefficients(best_estimator, feature_names)

    if spec.task_type == "classification":
        y_prob = best_estimator.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        pred_df = pd.DataFrame({
            "y_prob": y_prob,
            "y_pred": y_pred,
        })
    else:
        y_pred_value = best_estimator.predict(X_test)
        y_pred_value = np.asarray(y_pred_value).reshape(-1)
        pred_df = pd.DataFrame({
            "y_pred_value": y_pred_value,
        })

    hyper = {
        "best_score_inner_cv": best_score,
        "best_params_json": json.dumps(best_params, sort_keys=True),
        "n_features_input": X_train.shape[1],
        "n_train": len(X_train),
        "n_test": len(X_test),
    }
    return pred_df, hyper, coef_df


# -----------------------------
# Teacher-student MRI model
# -----------------------------

def _fixed_ridge_pipeline(alpha: float = 10.0) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("lowinfo", LowInfoFeatureFilter()),
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=alpha)),
    ])


def _make_oof_teacher_score(X_path: pd.DataFrame, y: pd.Series, random_seed: int, alpha: float) -> np.ndarray:
    if len(X_path) < 5:
        model = _fixed_ridge_pipeline(alpha=alpha)
        model.fit(X_path, y)
        return np.asarray(model.predict(X_path)).reshape(-1)
    cv = make_inner_cv_regression(len(X_path), random_seed)
    model = _fixed_ridge_pipeline(alpha=alpha)
    try:
        pred = cross_val_predict(model, X_path, y, cv=cv, n_jobs=1)
    except Exception as e:
        print(f"[WARN] cross_val_predict teacher failed; using in-sample teacher scores. Error={e}")
        model.fit(X_path, y)
        pred = model.predict(X_path)
    return np.asarray(pred).reshape(-1)


def fit_teacher_student_mri_model(
    X_mri_train: pd.DataFrame,
    X_mri_test: pd.DataFrame,
    X_path_train: pd.DataFrame,
    y_train: pd.Series,
    y_test_cont: pd.Series,
    y_test_binary: pd.Series,
    path_concept_train: pd.DataFrame,
    split_random_seed: int,
    alpha: float,
    lambda_dispersion: float,
    lambda_teacher_score: float,
    lambda_path_concepts: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Simple multi-output ridge student.

    Inputs are MRI-only. Training targets include the true continuous dispersion,
    a pathology-teacher OOF continuous score, and pathology concept presences.
    The returned primary prediction is the first output, i.e. continuous dispersion.
    """
    teacher_score_train = _make_oof_teacher_score(X_path_train, y_train, split_random_seed, alpha=alpha)

    y_scaler = StandardScaler()
    y_primary_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1)).reshape(-1)
    teacher_scaler = StandardScaler()
    teacher_scaled = teacher_scaler.fit_transform(teacher_score_train.reshape(-1, 1)).reshape(-1)

    concept_cols = [c for c in path_concept_train.columns if c not in META_COLS]
    if concept_cols:
        path_concepts = path_concept_train[concept_cols].copy().fillna(0.0)
        # Keep small-sample outputs bounded and comparable to the scaled y targets.
        path_concepts_scaled = StandardScaler(with_mean=True, with_std=True).fit_transform(path_concepts)
    else:
        path_concepts_scaled = np.zeros((len(y_train), 0))

    targets = [lambda_dispersion * y_primary_scaled.reshape(-1, 1)]
    if lambda_teacher_score > 0:
        targets.append(lambda_teacher_score * teacher_scaled.reshape(-1, 1))
    if lambda_path_concepts > 0 and path_concepts_scaled.shape[1] > 0:
        targets.append(lambda_path_concepts * path_concepts_scaled)
    Y_multi = np.hstack(targets)

    pipe = _fixed_ridge_pipeline(alpha=alpha)
    pipe.fit(X_mri_train, Y_multi)
    pred_multi = np.asarray(pipe.predict(X_mri_test))
    if pred_multi.ndim == 1:
        pred_multi = pred_multi.reshape(-1, 1)
    y_pred_scaled = pred_multi[:, 0] / max(lambda_dispersion, EPS)
    y_pred_value = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).reshape(-1)
    y_prob = np.clip((y_pred_value - DISPERSION_TRUE_HIGH_THRESHOLD + 20.0) / 40.0, 0.0, 1.0)
    y_pred_binary = (y_pred_value >= DISPERSION_TRUE_HIGH_THRESHOLD).astype(int)

    reg_pred = pd.DataFrame({
        "y_true": y_test_cont.values,
        "y_pred_value": y_pred_value,
    })
    cls_pred = pd.DataFrame({
        "y_true": y_test_binary.values,
        "y_prob": y_prob,
        "y_pred": y_pred_binary,
    })
    hyper = {
        "teacher_student_alpha": alpha,
        "lambda_dispersion": lambda_dispersion,
        "lambda_teacher_score": lambda_teacher_score,
        "lambda_path_concepts": lambda_path_concepts,
        "n_mri_features": X_mri_train.shape[1],
        "n_path_concept_targets": len(concept_cols),
        "n_train": len(X_mri_train),
        "n_test": len(X_mri_test),
    }
    coef_df = extract_fitted_feature_coefficients(pipe, list(X_mri_train.columns))
    return reg_pred, cls_pred, {"hyper": hyper, "coef_df": coef_df}


# -----------------------------
# Ranking, stability, plotting
# -----------------------------

def rank_features_across_models(stability_tables: List[pd.DataFrame], task_type: str) -> pd.DataFrame:
    tables = [df for df in stability_tables if len(df) and df["task_type"].iloc[0] == task_type]
    if not tables:
        return pd.DataFrame(columns=["target_name", "feature", "n_nonzero", "mean_abs_coef", "models"])

    all_df = pd.concat(tables, ignore_index=True)
    if "target_name" not in all_df.columns:
        all_df["target_name"] = "unknown"
    group_cols = ["target_name", "feature"]
    agg = (
        all_df.groupby(group_cols)
        .agg(
            n_nonzero=("abs_coef", lambda s: int((pd.Series(s) > COEF_ZERO_TOL).sum())),
            mean_abs_coef=("abs_coef", "mean"),
            median_abs_coef=("abs_coef", "median"),
            models=("model_key", lambda s: ";".join(sorted(set(map(str, s))))),
        )
        .reset_index()
        .sort_values(["target_name", "n_nonzero", "mean_abs_coef"], ascending=[True, False, False])
        .reset_index(drop=True)
    )
    return agg


def summarize_coefficient_sign_stability(coef_all: pd.DataFrame) -> pd.DataFrame:
    if len(coef_all) == 0:
        return pd.DataFrame()
    df = coef_all.copy()
    df["sign"] = np.sign(df["coef"].astype(float))
    df["nonzero"] = (df["abs_coef"].astype(float) > COEF_ZERO_TOL).astype(int)
    if "target_name" not in df.columns:
        df["target_name"] = "unknown"
    agg = df.groupby(["dataset_key", "representation", "model_key", "task_type", "target_name", "feature"]).agg(
        n_rows=("coef", "size"),
        n_outer_splits=("split_id", "nunique"),
        n_nonzero=("nonzero", "sum"),
        mean_coef=("coef", "mean"),
        median_coef=("coef", "median"),
        mean_abs_coef=("abs_coef", "mean"),
        n_positive=("sign", lambda s: int((pd.Series(s) > 0).sum())),
        n_negative=("sign", lambda s: int((pd.Series(s) < 0).sum())),
    ).reset_index()
    agg["dominant_sign"] = np.where(agg["n_positive"] >= agg["n_negative"], "positive", "negative")
    agg["sign_consistency"] = agg[["n_positive", "n_negative"]].max(axis=1) / agg["n_rows"].replace(0, np.nan)
    agg = agg.sort_values(["n_nonzero", "sign_consistency", "mean_abs_coef"], ascending=[False, False, False]).reset_index(drop=True)
    return agg


def _metric_plot_label(df: pd.DataFrame) -> pd.Series:
    target = df["target_name"].astype(str) if "target_name" in df.columns else pd.Series("target", index=df.index)
    return target + " | " + df["dataset_key"].astype(str) + " | " + df["model_key"].astype(str) + " | " + df["representation"].astype(str)


def _metric_bar_figure_size(labels: Sequence[str], horizontal: bool = False) -> Tuple[float, float]:
    n = max(len(labels), 1)
    max_len = max((len(str(x)) for x in labels), default=10)
    if horizontal:
        return (max(10.0, min(0.45 * max_len + 4.0, 18.0)), max(4.5, min(0.38 * n + 1.5, 24.0)))
    return (max(12.0, min(0.55 * n + 2.0, 28.0)), max(6.0, min(0.18 * max_len + 4.5, 16.0)))


def _save_figure(path: str, fig=None) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if fig is None:
        fig = plt.gcf()
    with contextlib.suppress(Exception):
        fig.tight_layout(pad=1.2)
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.35)
    plt.close(fig)


def _plot_metric_bars(labels: Sequence[str], values: Sequence[float], ylabel: str, title: str, out_png: str, ascending: bool = False) -> None:
    if len(labels) == 0:
        return
    order = np.argsort(values)
    if not ascending:
        order = order[::-1]
    labels = [str(labels[i]) for i in order]
    values = [float(values[i]) for i in order]
    fig_w, fig_h = _metric_bar_figure_size(labels, horizontal=True)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ypos = np.arange(len(labels))
    ax.barh(ypos, values)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel(ylabel)
    ax.set_title(title)
    _save_figure(out_png, fig)


def plot_classification_comparison(metrics_df: pd.DataFrame, out_png: str) -> None:
    df = metrics_df[metrics_df["task_type"] == "classification"].copy()
    if len(df) == 0 or "auroc" not in df.columns:
        return
    df["label"] = _metric_plot_label(df)
    _plot_metric_bars(
        df["label"].tolist(),
        df["auroc"].astype(float).tolist(),
        ylabel="AUROC",
        title="Nested held-out classification AUROC",
        out_png=out_png,
    )


def plot_relapse_metric_comparison(metrics_df: pd.DataFrame, out_png: str, metric: str) -> None:
    if "target_name" not in metrics_df.columns:
        return
    df = metrics_df[(metrics_df["task_type"] == "classification") & (metrics_df["target_name"] == TARGET_NAME_RELAPSE_STATUS)].copy()
    if len(df) == 0 or metric not in df.columns:
        return
    df = df.sort_values(metric, ascending=(metric == "brier")).copy()
    df["label"] = _metric_plot_label(df)
    _plot_metric_bars(
        df["label"].tolist(),
        df[metric].astype(float).tolist(),
        ylabel=metric.upper(),
        title=f"Nested held-out relapse prediction {metric.upper()}",
        out_png=out_png,
        ascending=(metric == "brier"),
    )


def plot_relapse_curves(pred_all: pd.DataFrame, metrics_df: pd.DataFrame, out_dir: str, top_n: int = 8) -> None:
    if len(pred_all) == 0 or "target_name" not in pred_all.columns:
        return
    rel_pred = pred_all[(pred_all["task_type"] == "classification") & (pred_all["target_name"] == TARGET_NAME_RELAPSE_STATUS)].copy()
    rel_metrics = metrics_df[(metrics_df["task_type"] == "classification") & (metrics_df["target_name"] == TARGET_NAME_RELAPSE_STATUS)].copy() if "target_name" in metrics_df.columns else pd.DataFrame()
    if len(rel_pred) == 0 or len(rel_metrics) == 0:
        return
    rel_metrics = rel_metrics.sort_values(["auroc", "auprc"], ascending=[False, False]).head(top_n)

    curve_specs = [
        ("roc", "nested_relapse_roc_curves_top_models.png", "False positive rate", "True positive rate"),
        ("pr", "nested_relapse_pr_curves_top_models.png", "Recall", "Precision"),
    ]
    for curve_type, filename, xlabel, ylabel in curve_specs:
        fig, ax = plt.subplots(figsize=(9, 7))
        any_curve = False
        for _, r in rel_metrics.iterrows():
            mask = (
                (rel_pred["dataset_key"].astype(str) == str(r["dataset_key"]))
                & (rel_pred["representation"].astype(str) == str(r["representation"]))
                & (rel_pred["model_key"].astype(str) == str(r["model_key"]))
            )
            sub = rel_pred.loc[mask].copy()
            if len(sub) < 2 or sub["y_true"].nunique() < 2:
                continue
            y_true = sub["y_true"].astype(int).values
            prob = sub["y_prob"].astype(float).values
            label = f"{r['dataset_key']} | {r['representation']} | {r['model_key']}"
            if curve_type == "roc":
                x, y, _ = roc_curve(y_true, prob)
            else:
                y, x, _ = precision_recall_curve(y_true, prob)
            ax.plot(x, y, label=label[:90])
            any_curve = True
        if any_curve:
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title("Held-out relapse prediction curves")
            ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.02, 0.5))
            _save_figure(os.path.join(out_dir, filename), fig)
        else:
            plt.close(fig)


def plot_regression_error_comparison(metrics_df: pd.DataFrame, out_png: str) -> None:
    df = metrics_df[metrics_df["task_type"] == "regression"].copy()
    if len(df) == 0 or "mae" not in df.columns:
        return
    df = df.sort_values("mae", ascending=True)
    df["label"] = _metric_plot_label(df)
    _plot_metric_bars(
        df["label"].tolist(),
        df["mae"].astype(float).tolist(),
        ylabel="MAE",
        title="Nested held-out regression MAE",
        out_png=out_png,
        ascending=True,
    )


def plot_regression_correlation_comparison(metrics_df: pd.DataFrame, out_png: str) -> None:
    df = metrics_df[metrics_df["task_type"] == "regression"].copy()
    if len(df) == 0 or "spearman_rho" not in df.columns:
        return
    df = df.sort_values("spearman_rho", ascending=False)
    df["label"] = _metric_plot_label(df)
    _plot_metric_bars(
        df["label"].tolist(),
        df["spearman_rho"].astype(float).tolist(),
        ylabel="Spearman rho",
        title="Nested held-out regression rank correlation",
        out_png=out_png,
    )


def summarize_metrics(metrics_df: pd.DataFrame) -> str:
    lines: List[str] = []
    lines.append("=== Nested Resampling Predictive Evaluation Summary ===")
    if len(metrics_df) == 0:
        lines.append("No metrics were generated.")
        return "\n".join(lines)

    cls = metrics_df[metrics_df["task_type"] == "classification"].copy()
    reg = metrics_df[metrics_df["task_type"] == "regression"].copy()

    if len(cls):
        if "target_name" not in cls.columns:
            cls["target_name"] = TARGET_NAME_DISPERSION_HIGH_LOW
        for target_name, target_df in cls.groupby("target_name"):
            target_df = target_df.copy()
            sort_cols = [c for c in ["auroc", "auprc", "f1"] if c in target_df.columns]
            if not sort_cols:
                continue
            best_cls = target_df.sort_values(sort_cols, ascending=[False] * len(sort_cols)).iloc[0]
            lines.append(
                f"Best classification setting for {target_name}: "
                f"{best_cls['dataset_key']} | {best_cls['representation']} | {best_cls['model_key']} "
                f"(AUROC={best_cls.get('auroc', np.nan):.3f}, AUPRC={best_cls.get('auprc', np.nan):.3f}, "
                f"F1={best_cls.get('f1', np.nan):.3f}, Brier={best_cls.get('brier', np.nan):.3f})"
            )

    if len(reg):
        best_reg = reg.sort_values(["mae", "spearman_rho"], ascending=[True, False]).iloc[0]
        lines.append(
            "Best regression setting: "
            f"{best_reg['dataset_key']} | {best_reg['representation']} | {best_reg['model_key']} "
            f"(MAE={best_reg['mae']:.3f}, RMSE={best_reg['rmse']:.3f}, "
            f"R2={best_reg['r2']:.3f}, Spearman={best_reg['spearman_rho']:.3f})"
        )

    lines.append("")
    lines.append("All metrics below are aggregated over outer held-out predictions only.")
    return "\n".join(lines)


def write_methodology_markdown(out_dir: str, args: argparse.Namespace) -> str:
    path = os.path.join(out_dir, "PATHOLOGY_INFORMED_MRI_REFINEMENT_SUMMARY.md")
    text = f"""# Pathology-Informed MRI Lexical Refinement: Implementation Summary

## What changed

This run extends the original nested NLP + ML pipeline with pathology-informed MRI refinement while preserving the original leakage-aware evaluation design.

### Added outputs

- `shared_biological_concept_ontology.csv`: concept definitions, examples, and mapping regexes.
- per-split `*_mri_audit_case_table.csv`: MRI report length, extraction/concept densities, uncertainty/negation densities, and section flags.
- per-split `*_mri_audit_density_summary.csv`: high/low and relapse-stratified audit summaries when labels are available.
- per-split `*_mri_pathology_reliability_matrix.csv`: training-only MRI-concept to pathology-concept concordance.
- per-split `*_weighted_mri_lexicon.csv`: pathology-informed MRI concept weights learned from outer-training cases only.
- per-split `*_weighted_mri_concept_score_matrix.csv`: MRI-only weighted concept scores applied to train/test cases using frozen weights.
- optional ablation weighted matrices for randomized pathology labels and mismatched MRI-pathology pairing.
- relapse-status predictions from the same dispersion-vector feature matrices used for high/low dispersion classification.
- process-wide fold-level concurrency when `--parallel-fold-workers > 1`; each outer fold writes to an isolated split directory with its own frozen lexicons, calibration files, logs/checkpoints, and split provenance manifest.
- process-wide API concurrency control through a global semaphore, so `--max-api-workers` caps active API calls across all folds/modalities/cases rather than multiplying silently.
- automated performance reports: `automated_results_report.md` and `automated_results_report.html`, plus interleaved plots and deduplicated one-prediction-per-case metrics.
- interpretability reports: `interpretability_report.md` and `interpretability_report.html`, coefficient annotations, feature-density summaries, reliability heatmaps, and calibration-weight plots.
- missed-case/error reports: `missed_case_error_analysis.csv` and `missed_case_error_analysis.md`.
- relapse-specific AUROC, AUPRC, no-skill AUPRC baseline, F1, Brier, precision/PPV, recall/sensitivity, specificity, NPV, balanced accuracy, confusion-matrix counts, calibration diagnostics, bootstrap confidence intervals, class-balance diagnostics, and permutation tests.
- MRI-missing cases are explicitly skipped for MRI-only, MRI+pathology combined, pathology-calibrated MRI, and teacher-student MRI evaluations. Pathology-only evaluations retain these cases.
- `nested_feature_sign_stability.csv`: sign-consistency and coefficient-stability summary across outer splits.
- `logs/run_log_feature_discovery_nested_eval.txt`: persisted stdout/stderr log for resume/debugging.

## Leakage protocol

For the primary scientific claim, the following operations are restricted to each outer-training split:

1. LLM extraction.
2. rediscovery and stable lexicon definition.
3. MRI-pathology reliability estimation.
4. weighted MRI lexicon derivation.
5. pathology-teacher score construction.
6. teacher-student training.

Outer-test pathology is not used to calibrate, weight, or train MRI features. Outer-test prediction uses MRI-derived features only.

## Main new dataset/model keys

- `mri`: original MRI-only lexical baseline.
- `path`: original pathology-only lexical baseline.
- `combined`: original MRI + pathology early-fusion baseline.
- `mri_pathcal_weighted`: MRI-only weighted concept-score model using training-only pathology calibration.
- `mri_pathcal_weighted_random_pathology`: optional negative-control ablation.
- `mri_pathcal_weighted_mismatched_pairing`: optional pairing-control ablation.
- `mri_teacher_student`: optional MRI-only multi-task ridge student trained with pathology-derived training targets.

## Ablation meaning in this code

An ablation is a deliberately altered control run that removes or corrupts one component of the proposed method while keeping the rest of the pipeline similar. Here, randomized-pathology and mismatched-pairing ablations test whether pathology-calibrated MRI weights help because of real MRI-pathology biological concordance, rather than because any extra weighting or regularization improves performance nonspecifically.

## Important critique and limitations

1. The ontology is rule-based. It improves interpretability and robustness but can miss synonyms, local report conventions, and context-dependent meanings.
2. Regex recoding is not a substitute for full clinical text understanding. It is intended as a leakage-safe, frozen representation layer after LLM discovery.
3. Pathology is a teacher, not truth. Concordance with pathology may downweight MRI features that are radiographically meaningful but not sampled or described pathologically.
4. Weighted lexicon formulas are heuristic and should be ablated. Improved performance under randomized or mismatched pathology ablations would indicate overfitting or non-specific regularization rather than true biological supervision.
5. The teacher-student model is intentionally simple. With ~100 cases, complex neural multi-task learning would likely overfit unless externally validated.
6. Internal nested performance is still small-sample and high-variance. External validation remains necessary before clinical claims.

## Run configuration

```text
pathology_calibration_enabled = {args.enable_pathology_calibration}
teacher_student_enabled = {args.enable_teacher_student}
ontology_groups_mode = {args.ontology_groups_mode}
weighted_uncertain_value = {args.weighted_uncertain_value}
weighted_negated_value = {args.weighted_negated_value}
calibration_smoothing = {args.calibration_smoothing}
run_calibration_ablations = {args.run_calibration_ablations}
parallel_fold_workers = {getattr(args, "parallel_fold_workers", 1)}
parallel_modality_workers = {getattr(args, "parallel_modality_workers", 1)}
max_api_workers_global = {getattr(args, "max_api_workers", 1)}
ml_n_jobs = {getattr(args, "ml_n_jobs", 1)}
bootstrap_n = {getattr(args, "bootstrap_n", 0)}
relapse_permutation_n = {getattr(args, "relapse_permutation_n", 0)}
```
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path


# -----------------------------
# One modality / one outer split
# -----------------------------

def _atomic_write_df(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def _safe_read_csv_if_exists(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"[RESUME] Failed to read checkpoint table {path}: {e}")
        return pd.DataFrame()


def _split_resume_dir(split_dir: str) -> str:
    path = os.path.join(split_dir, "_split_resume_checkpoint")
    os.makedirs(path, exist_ok=True)
    return path


def _split_resume_marker(split_dir: str) -> str:
    return os.path.join(_split_resume_dir(split_dir), "COMPLETED.json")


def _split_resume_paths(split_dir: str) -> Dict[str, str]:
    root = _split_resume_dir(split_dir)
    return {
        "phrase_freq_df": os.path.join(root, "phrase_freq.csv"),
        "group_freq_df": os.path.join(root, "group_freq.csv"),
        "stable_phrase_df": os.path.join(root, "stable_phrase.csv"),
        "stable_group_df": os.path.join(root, "stable_group.csv"),
        "mri_audit_df": os.path.join(root, "mri_audit.csv"),
        "mri_audit_summary_df": os.path.join(root, "mri_audit_summary.csv"),
        "reliability_df": os.path.join(root, "reliability.csv"),
        "weighted_lexicon_df": os.path.join(root, "weighted_lexicon.csv"),
        "predictions_df": os.path.join(root, "predictions.csv"),
        "fold_metrics_df": os.path.join(root, "fold_metrics.csv"),
        "hyperparameters_df": os.path.join(root, "hyperparameters.csv"),
        "coefficients_df": os.path.join(root, "coefficients.csv"),
    }


def _concat_tables_for_split(tables: List[pd.DataFrame], split_id: str) -> pd.DataFrame:
    if not tables:
        return pd.DataFrame()
    usable = [t for t in tables if isinstance(t, pd.DataFrame) and len(t)]
    if not usable:
        return pd.DataFrame()
    df = pd.concat(usable, ignore_index=True)
    if "split_id" in df.columns:
        df = df[df["split_id"].astype(str) == str(split_id)].copy()
    return df.reset_index(drop=True)


def save_completed_split_checkpoint(
    split_dir: str,
    split_id: str,
    args: argparse.Namespace,
    all_phrase_freq_tables: List[pd.DataFrame],
    all_group_freq_tables: List[pd.DataFrame],
    all_stable_phrase_tables: List[pd.DataFrame],
    all_stable_group_tables: List[pd.DataFrame],
    all_mri_audit_tables: List[pd.DataFrame],
    all_mri_audit_summary_tables: List[pd.DataFrame],
    all_reliability_tables: List[pd.DataFrame],
    all_weighted_lexicon_tables: List[pd.DataFrame],
    prediction_tables: List[pd.DataFrame],
    fold_result_tables: List[pd.DataFrame],
    hyper_tables: List[pd.DataFrame],
    coef_tables: List[pd.DataFrame],
) -> None:
    """Write enough per-split state to skip this split on rerun."""
    paths = _split_resume_paths(split_dir)
    payloads = {
        "phrase_freq_df": _concat_tables_for_split(all_phrase_freq_tables, split_id),
        "group_freq_df": _concat_tables_for_split(all_group_freq_tables, split_id),
        "stable_phrase_df": _concat_tables_for_split(all_stable_phrase_tables, split_id),
        "stable_group_df": _concat_tables_for_split(all_stable_group_tables, split_id),
        "mri_audit_df": _concat_tables_for_split(all_mri_audit_tables, split_id),
        "mri_audit_summary_df": _concat_tables_for_split(all_mri_audit_summary_tables, split_id),
        "reliability_df": _concat_tables_for_split(all_reliability_tables, split_id),
        "weighted_lexicon_df": _concat_tables_for_split(all_weighted_lexicon_tables, split_id),
        "predictions_df": _concat_tables_for_split(prediction_tables, split_id),
        "fold_metrics_df": _concat_tables_for_split(fold_result_tables, split_id),
        "hyperparameters_df": _concat_tables_for_split(hyper_tables, split_id),
        "coefficients_df": _concat_tables_for_split(coef_tables, split_id),
    }
    for key, df in payloads.items():
        _atomic_write_df(df, paths[key])

    marker = {
        "split_id": split_id,
        "completed_at": datetime.now().isoformat(timespec="seconds"),
        "csv_path": getattr(args, "csv_path", ""),
        "outer_scheme": getattr(args, "outer_scheme", ""),
        "outer_repeats": getattr(args, "outer_repeats", None),
        "outer_test_frac": getattr(args, "outer_test_frac", None),
        "outer_folds": getattr(args, "outer_folds", None),
        "random_seed": getattr(args, "random_seed", None),
        "modalities": getattr(args, "modalities", []),
        "representations": getattr(args, "representations", []),
    }
    marker_path = _split_resume_marker(split_dir)
    tmp_marker = f"{marker_path}.tmp.{os.getpid()}"
    with open(tmp_marker, "w", encoding="utf-8") as f:
        json.dump(marker, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp_marker, marker_path)
    print(f"[RESUME] Wrote completed split checkpoint marker: {marker_path}")


def load_completed_split_checkpoint(split_dir: str, split_id: str) -> Optional[Dict[str, pd.DataFrame]]:
    marker_path = _split_resume_marker(split_dir)
    if not os.path.exists(marker_path):
        return None
    try:
        with open(marker_path, "r", encoding="utf-8") as f:
            marker = json.load(f)
        if str(marker.get("split_id")) != str(split_id):
            print(f"[RESUME] Marker split_id mismatch in {marker_path}; recomputing split.")
            return None
    except Exception as e:
        print(f"[RESUME] Could not read split marker {marker_path}: {e}; recomputing split.")
        return None

    paths = _split_resume_paths(split_dir)
    loaded = {key: _safe_read_csv_if_exists(path) for key, path in paths.items()}
    print(f"[RESUME] Loaded completed split checkpoint for {split_id}: {marker_path}")
    return loaded

def run_outer_split_for_modality(
    raw_df: pd.DataFrame,
    outer_train_case_df: pd.DataFrame,
    outer_test_case_df: pd.DataFrame,
    report_mode: str,
    split_id: str,
    split_dir: str,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    mode_dir = os.path.join(split_dir, report_mode)
    os.makedirs(mode_dir, exist_ok=True)

    print(
        f"[OUTER] split={split_id} report_mode={report_mode} "
        f"n_train={len(outer_train_case_df)} n_test={len(outer_test_case_df)}"
    )

    train_indices = outer_train_case_df["row_index"].astype(int).tolist()

    t0 = time.time()
    train_records = extract_subset_records(
        df=raw_df,
        row_indices=train_indices,
        report_mode=report_mode,
        split_id=split_id,
        split_role="train",
        sleep_between_calls_s=args.rate_limit_sleep_s,
        max_workers=args.max_api_workers,
        checkpoint_dir=mode_dir,
        resume=args.resume,
        force_reextract=args.force_reextract,
    )
    train_paths = write_extractions(
        extractions=train_records,
        out_dir=mode_dir,
        report_mode=report_mode,
        filename_prefix=f"{split_id}_train",
        split_id=split_id,
    )
    print(f"[OUTER] First-pass train-only extraction completed in {(time.time() - t0)/60:.2f} min.")

    train_extractions_df = load_extractions_csv(train_paths["csv"])
    train_phrase_df = explode_phrase_rows(train_extractions_df)
    train_phrase_csv = os.path.join(mode_dir, f"{split_id}_normalized_phrase_table_train_{report_mode}.csv")
    train_phrase_df.to_csv(train_phrase_csv, index=False)

    phrase_freq_df, group_freq_df, stable_phrase_df, stable_group_df = build_stable_lexicon_from_training_extractions(
        train_extractions_df=train_extractions_df,
        train_phrase_df=train_phrase_df,
        rediscovery_scheme=args.rediscovery_scheme,
        rediscovery_repeats=args.rediscovery_repeats,
        rediscovery_test_frac=args.rediscovery_test_frac,
        rediscovery_folds=args.rediscovery_folds,
        stability_threshold=args.stability_threshold,
        min_phrase_cases=args.min_phrase_cases,
        min_group_cases=args.min_group_cases,
        random_seed=args.random_seed + int(split_id.split("_")[-1]),
    )

    phrase_freq_csv = os.path.join(mode_dir, f"{split_id}_phrase_rediscovery_frequency_{report_mode}.csv")
    group_freq_csv = os.path.join(mode_dir, f"{split_id}_group_rediscovery_frequency_{report_mode}.csv")
    stable_phrase_csv = os.path.join(mode_dir, f"{split_id}_stable_phrase_lexicon_{report_mode}.csv")
    stable_group_csv = os.path.join(mode_dir, f"{split_id}_stable_group_lexicon_{report_mode}.csv")

    phrase_freq_df.to_csv(phrase_freq_csv, index=False)
    group_freq_df.to_csv(group_freq_csv, index=False)
    stable_phrase_df.to_csv(stable_phrase_csv, index=False)
    stable_group_df.to_csv(stable_group_csv, index=False)

    extra_groups: List[str] = []
    if args.ontology_groups_mode == "stable_plus_ontology":
        extra_groups = list(SHARED_CONCEPT_ONTOLOGY.keys())

    outer_case_df = pd.concat([outer_train_case_df, outer_test_case_df], ignore_index=True).copy()
    phrase_prov_df, group_prov_df = recode_cases_with_frozen_lexicon(
        raw_df=raw_df,
        outer_case_df=outer_case_df,
        report_mode=report_mode,
        stable_phrase_df=stable_phrase_df,
        stable_group_df=stable_group_df,
        split_id=split_id,
        train_case_ids=outer_train_case_df["case_id"].astype(str).tolist(),
        extra_group_names=extra_groups,
    )
    phrase_prov_csv = os.path.join(mode_dir, f"{split_id}_phrase_recode_provenance_{report_mode}.csv")
    group_prov_csv = os.path.join(mode_dir, f"{split_id}_group_recode_provenance_{report_mode}.csv")
    phrase_prov_df.to_csv(phrase_prov_csv, index=False)
    group_prov_df.to_csv(group_prov_csv, index=False)

    group_matrix_df = build_group_feature_matrix(group_prov_df, outer_case_df, split_id, report_mode)
    phrase_matrix_df = build_phrase_feature_matrix(phrase_prov_df, outer_case_df, split_id, report_mode)

    group_matrix_csv = os.path.join(mode_dir, f"{split_id}_group_feature_matrix_{report_mode}.csv")
    phrase_matrix_csv = os.path.join(mode_dir, f"{split_id}_phrase_feature_matrix_{report_mode}.csv")
    group_matrix_df.to_csv(group_matrix_csv, index=False)
    phrase_matrix_df.to_csv(phrase_matrix_csv, index=False)

    audit_df = pd.DataFrame()
    audit_summary_df = pd.DataFrame()
    if report_mode == "mri":
        audit_df = compute_mri_audit_table(raw_df, outer_case_df, phrase_prov_df, group_prov_df, split_id)
        audit_summary_df = summarize_audit_by_groups(audit_df)
        audit_csv = os.path.join(mode_dir, f"{split_id}_mri_audit_case_table.csv")
        audit_summary_csv = os.path.join(mode_dir, f"{split_id}_mri_audit_density_summary.csv")
        audit_df.to_csv(audit_csv, index=False)
        audit_summary_df.to_csv(audit_summary_csv, index=False)
        print(f"[AUDIT] Wrote MRI audit table: {audit_csv}")
        print(f"[AUDIT] Wrote MRI audit summary: {audit_summary_csv}")

    return {
        "train_extractions_df": train_extractions_df,
        "train_phrase_df": train_phrase_df,
        "phrase_freq_df": phrase_freq_df.assign(split_id=split_id, report_mode=report_mode),
        "group_freq_df": group_freq_df.assign(split_id=split_id, report_mode=report_mode),
        "stable_phrase_df": stable_phrase_df.assign(split_id=split_id, report_mode=report_mode),
        "stable_group_df": stable_group_df.assign(split_id=split_id, report_mode=report_mode),
        "phrase_prov_df": phrase_prov_df,
        "group_prov_df": group_prov_df,
        "group_matrix_df": group_matrix_df,
        "phrase_matrix_df": phrase_matrix_df,
        "mri_audit_df": audit_df,
        "mri_audit_summary_df": audit_summary_df,
    }



# -----------------------------
# Fold-level orchestration, reports, and diagnostics
# -----------------------------

def _empty_split_result(split_id: str) -> Dict[str, List[pd.DataFrame]]:
    return {
        "split_id": split_id,
        "all_phrase_freq_tables": [],
        "all_group_freq_tables": [],
        "all_stable_phrase_tables": [],
        "all_stable_group_tables": [],
        "all_mri_audit_tables": [],
        "all_mri_audit_summary_tables": [],
        "all_reliability_tables": [],
        "all_weighted_lexicon_tables": [],
        "prediction_tables": [],
        "fold_result_tables": [],
        "hyper_tables": [],
        "coef_tables": [],
        "error_tables": [],
    }


def _result_from_loaded_checkpoint(split_id: str, loaded_split: Dict[str, pd.DataFrame]) -> Dict[str, List[pd.DataFrame]]:
    result = _empty_split_result(split_id)
    mapping = {
        "phrase_freq_df": "all_phrase_freq_tables",
        "group_freq_df": "all_group_freq_tables",
        "stable_phrase_df": "all_stable_phrase_tables",
        "stable_group_df": "all_stable_group_tables",
        "mri_audit_df": "all_mri_audit_tables",
        "mri_audit_summary_df": "all_mri_audit_summary_tables",
        "reliability_df": "all_reliability_tables",
        "weighted_lexicon_df": "all_weighted_lexicon_tables",
        "predictions_df": "prediction_tables",
        "fold_metrics_df": "fold_result_tables",
        "hyperparameters_df": "hyper_tables",
        "coefficients_df": "coef_tables",
    }
    for source_key, result_key in mapping.items():
        df = loaded_split.get(source_key, pd.DataFrame())
        if isinstance(df, pd.DataFrame) and len(df):
            result[result_key].append(df)
    return result


def _extend_aggregate_tables(result: Dict[str, Any], aggregate_lists: Dict[str, List[pd.DataFrame]]) -> None:
    for key, dest in aggregate_lists.items():
        for df in result.get(key, []):
            if isinstance(df, pd.DataFrame) and len(df):
                dest.append(df)


def _case_hash(values: Sequence[Any]) -> str:
    payload = "\n".join(map(str, values))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def write_split_provenance(
    split_dir: str,
    split_id: str,
    outer_train_case_df: pd.DataFrame,
    outer_test_case_df: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    """Persist immutable train/test membership and hashes for audit/resume."""
    os.makedirs(split_dir, exist_ok=True)
    train_ids = outer_train_case_df["case_id"].astype(str).tolist()
    test_ids = outer_test_case_df["case_id"].astype(str).tolist()
    prov = pd.concat([
        outer_train_case_df[["case_id", "row_index", "dispersion_true", "dispersion_true_high_low", "relapse_true"]].assign(split_id=split_id, split_role="train"),
        outer_test_case_df[["case_id", "row_index", "dispersion_true", "dispersion_true_high_low", "relapse_true"]].assign(split_id=split_id, split_role="test"),
    ], ignore_index=True)
    prov.to_csv(os.path.join(split_dir, f"{split_id}_split_provenance.csv"), index=False)
    manifest = {
        "split_id": split_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "random_seed": int(getattr(args, "random_seed", RANDOM_SEED)),
        "outer_scheme": getattr(args, "outer_scheme", ""),
        "outer_repeats": getattr(args, "outer_repeats", None),
        "outer_test_frac": getattr(args, "outer_test_frac", None),
        "outer_folds": getattr(args, "outer_folds", None),
        "n_train": len(train_ids),
        "n_test": len(test_ids),
        "train_case_hash": _case_hash(train_ids),
        "test_case_hash": _case_hash(test_ids),
        "train_case_ids": train_ids,
        "test_case_ids": test_ids,
    }
    with open(os.path.join(split_dir, f"{split_id}_split_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")


def write_failed_split_marker(split_dir: str, split_id: str, exc: BaseException) -> pd.DataFrame:
    os.makedirs(split_dir, exist_ok=True)
    tb = traceback.format_exc()
    payload = {
        "split_id": split_id,
        "failed_at": datetime.now().isoformat(timespec="seconds"),
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "traceback": tb,
    }
    path = os.path.join(_split_resume_dir(split_dir), "FAILED.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"[ERROR] Fold {split_id} failed; wrote failure marker: {path}")
    return pd.DataFrame([payload])


def run_one_outer_split(
    split_num: int,
    train_pos: np.ndarray,
    test_pos: np.ndarray,
    raw_df: pd.DataFrame,
    target_df: pd.DataFrame,
    model_specs: Sequence[ModelSpec],
    standard_representations: Sequence[str],
    weighted_representations: Sequence[str],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """Run one outer split with only split-local mutable state.

    This function is the unit of fold-level parallelism. All artifacts are
    written under this split's own output directory and returned as local
    tables, avoiding concurrent mutation of aggregate lists in the main thread.
    """
    split_id = f"outer_split_{split_num:03d}"
    split_dir = os.path.join(args.out_dir, "outer_splits", split_id)
    os.makedirs(split_dir, exist_ok=True)
    result = _empty_split_result(split_id)

    try:
        if args.resume and args.skip_completed_splits:
            loaded_split = load_completed_split_checkpoint(split_dir, split_id)
            if loaded_split is not None:
                print(f"[RESUME] Skipping completed {split_id}; loading per-split outputs into aggregate tables.")
                return _result_from_loaded_checkpoint(split_id, loaded_split)

        outer_train_case_df = target_df.iloc[train_pos].copy().reset_index(drop=True)
        outer_test_case_df = target_df.iloc[test_pos].copy().reset_index(drop=True)
        outer_train_case_df["split_role"] = "train"
        outer_test_case_df["split_role"] = "test"
        write_split_provenance(split_dir, split_id, outer_train_case_df, outer_test_case_df, args)

        print("=" * 100)
        print(
            f"[OUTER] {split_id}: "
            f"n_train={len(outer_train_case_df)} n_test={len(outer_test_case_df)} "
            f"train_pos={int(outer_train_case_df['dispersion_true_high_low'].sum())}/"
            f"{len(outer_train_case_df)} "
            f"test_pos={int(outer_test_case_df['dispersion_true_high_low'].sum())}/"
            f"{len(outer_test_case_df)}"
        )
        print("=" * 100)

        modality_results: Dict[str, Dict[str, Any]] = {}
        modes_to_run = [
            mode for mode in ["mri", "path"]
            if mode in args.modalities or "combined" in args.modalities or args.enable_pathology_calibration or args.enable_teacher_student
        ]

        if len(modes_to_run) > 1 and args.parallel_modality_workers > 1:
            mode_workers = min(args.parallel_modality_workers, len(modes_to_run))
            print(
                f"[OUTER] {split_id} running modalities in parallel with workers={mode_workers}: "
                f"{','.join(modes_to_run)}"
            )
            with ThreadPoolExecutor(max_workers=mode_workers) as executor:
                future_to_mode = {
                    executor.submit(
                        run_outer_split_for_modality,
                        raw_df=raw_df,
                        outer_train_case_df=outer_train_case_df,
                        outer_test_case_df=outer_test_case_df,
                        report_mode=mode,
                        split_id=split_id,
                        split_dir=split_dir,
                        args=args,
                    ): mode
                    for mode in modes_to_run
                }
                for future in as_completed(future_to_mode):
                    mode = future_to_mode[future]
                    modality_results[mode] = future.result()
        else:
            for mode in modes_to_run:
                modality_results[mode] = run_outer_split_for_modality(
                    raw_df=raw_df,
                    outer_train_case_df=outer_train_case_df,
                    outer_test_case_df=outer_test_case_df,
                    report_mode=mode,
                    split_id=split_id,
                    split_dir=split_dir,
                    args=args,
                )

        for mode in modes_to_run:
            result["all_phrase_freq_tables"].append(modality_results[mode]["phrase_freq_df"])
            result["all_group_freq_tables"].append(modality_results[mode]["group_freq_df"])
            result["all_stable_phrase_tables"].append(modality_results[mode]["stable_phrase_df"])
            result["all_stable_group_tables"].append(modality_results[mode]["stable_group_df"])
        if "mri" in modality_results and len(modality_results["mri"].get("mri_audit_df", pd.DataFrame())):
            result["all_mri_audit_tables"].append(modality_results["mri"]["mri_audit_df"])
            result["all_mri_audit_summary_tables"].append(modality_results["mri"]["mri_audit_summary_df"])

        dataset_matrices: Dict[str, Dict[str, pd.DataFrame]] = {}
        if "mri" in args.modalities:
            dataset_matrices["mri"] = {"group": modality_results["mri"]["group_matrix_df"], "phrase": modality_results["mri"]["phrase_matrix_df"]}
        if "path" in args.modalities:
            dataset_matrices["path"] = {"group": modality_results["path"]["group_matrix_df"], "phrase": modality_results["path"]["phrase_matrix_df"]}
        if "combined" in args.modalities:
            dataset_matrices["combined"] = {"group": None, "phrase": None}

        if args.enable_pathology_calibration:
            print(f"[CALIBRATION] {split_id}: computing MRI↔pathology reliability on outer-training cases only.")
            mri_group_matrix = filter_missing_mri_for_dataset(
                modality_results["mri"]["group_matrix_df"], raw_df, "mri_pathcal_weighted", split_id
            )
            path_group_matrix = modality_results["path"]["group_matrix_df"]
            mri_available_train_df = outer_train_case_df[
                ~outer_train_case_df["row_index"].astype(int).isin(_mri_missing_row_indices(raw_df))
            ].copy()
            train_ids = mri_available_train_df["case_id"].astype(str).tolist()
            if len(train_ids) == 0:
                print(f"[CALIBRATION] {split_id}: no MRI-available training cases; skipping pathology-informed MRI calibration.")
                reliability_df = pd.DataFrame()
            else:
                reliability_df = compute_cross_modal_reliability(
                    mri_group_matrix_df=mri_group_matrix,
                    path_group_matrix_df=path_group_matrix,
                    train_case_ids=train_ids,
                    split_id=split_id,
                    smoothing=args.calibration_smoothing,
                )
            rel_csv = os.path.join(split_dir, f"{split_id}_mri_pathology_reliability_matrix.csv")
            reliability_df.to_csv(rel_csv, index=False)
            result["all_reliability_tables"].append(reliability_df)
            print(f"[CALIBRATION] Wrote reliability matrix: {rel_csv}")

            weighted_lexicon_df = compute_weighted_mri_lexicon(
                reliability_df=reliability_df,
                mri_group_freq_df=modality_results["mri"]["group_freq_df"],
                mri_group_matrix_df=mri_group_matrix,
                train_case_ids=train_ids,
                y_train_continuous=mri_available_train_df.set_index("case_id")["dispersion_true"],
                split_id=split_id,
                min_selection_frequency=args.weighted_lexicon_min_selection_frequency,
                reliability_power=args.weight_reliability_power,
                stability_power=args.weight_stability_power,
                association_power=args.weight_association_power,
            )
            weighted_lexicon_csv = os.path.join(split_dir, f"{split_id}_weighted_mri_lexicon.csv")
            weighted_lexicon_df.to_csv(weighted_lexicon_csv, index=False)
            result["all_weighted_lexicon_tables"].append(weighted_lexicon_df)
            print(f"[CALIBRATION] Wrote weighted MRI lexicon: {weighted_lexicon_csv}")

            weighted_matrix_df = build_weighted_mri_concept_score_matrix(
                mri_group_matrix_df=mri_group_matrix,
                weighted_lexicon_df=weighted_lexicon_df,
                uncertain_value=args.weighted_uncertain_value,
                negated_value=args.weighted_negated_value,
                split_id=split_id,
            )
            weighted_matrix_csv = os.path.join(split_dir, f"{split_id}_weighted_mri_concept_score_matrix.csv")
            weighted_matrix_df.to_csv(weighted_matrix_csv, index=False)
            dataset_matrices["mri_pathcal_weighted"] = {"weighted": weighted_matrix_df, "group": weighted_matrix_df, "phrase": weighted_matrix_df}
            print(f"[CALIBRATION] Wrote weighted MRI concept score matrix: {weighted_matrix_csv}")

            if args.run_calibration_ablations:
                for ablation_key, mode in [
                    ("mri_pathcal_weighted_random_pathology", "randomized_labels"),
                    ("mri_pathcal_weighted_mismatched_pairing", "mismatched_pairing"),
                ]:
                    path_ablate = randomized_or_mismatched_path_matrix(path_group_matrix, train_ids, mode, args.random_seed + split_num)
                    rel_ablate = compute_cross_modal_reliability(mri_group_matrix, path_ablate, train_ids, split_id, args.calibration_smoothing)
                    w_ablate = compute_weighted_mri_lexicon(rel_ablate, modality_results["mri"]["group_freq_df"], mri_group_matrix, train_ids, mri_available_train_df.set_index("case_id")["dispersion_true"], split_id, args.weighted_lexicon_min_selection_frequency, args.weight_reliability_power, args.weight_stability_power, args.weight_association_power)
                    mat_ablate = build_weighted_mri_concept_score_matrix(mri_group_matrix, w_ablate, args.weighted_uncertain_value, args.weighted_negated_value, split_id)
                    mat_ablate.to_csv(os.path.join(split_dir, f"{split_id}_{ablation_key}_matrix.csv"), index=False)
                    dataset_matrices[ablation_key] = {"weighted": mat_ablate, "group": mat_ablate, "phrase": mat_ablate}
                    print(f"[ABLATION] Built {ablation_key} dataset.")

        for dataset_key in list(dataset_matrices.keys()):
            reps = weighted_representations if "weighted" in dataset_key else standard_representations
            for representation in reps:
                if dataset_key == "combined":
                    if representation.startswith("group_"):
                        dataset_df, feature_cols = merge_modalities_early_fusion(
                            modality_results["mri"]["group_matrix_df"],
                            modality_results["path"]["group_matrix_df"],
                            representation=representation,
                        )
                    elif representation == "phrase_binary":
                        dataset_df, feature_cols = merge_modalities_early_fusion(
                            modality_results["mri"]["phrase_matrix_df"],
                            modality_results["path"]["phrase_matrix_df"],
                            representation=representation,
                        )
                    else:
                        continue
                elif "weighted" in dataset_key:
                    dataset_df = dataset_matrices[dataset_key]["weighted"].copy()
                    X_tmp, feature_cols = get_representation_matrix(dataset_df, representation)
                    dataset_df = pd.concat([
                        dataset_df[["case_id", "row_index", "dispersion_true", "dispersion_true_high_low", "relapse_true"]],
                        X_tmp,
                    ], axis=1)
                else:
                    source_key = "group" if representation.startswith("group_") else "phrase"
                    dataset_df = dataset_matrices[dataset_key][source_key].copy()
                    X_tmp, feature_cols = get_representation_matrix(dataset_df, representation)
                    dataset_df = pd.concat([
                        dataset_df[["case_id", "row_index", "dispersion_true", "dispersion_true_high_low", "relapse_true"]],
                        X_tmp,
                    ], axis=1)

                dataset_df = dataset_df.drop_duplicates(subset=["case_id", "row_index"]).reset_index(drop=True)
                dataset_df = filter_missing_mri_for_dataset(dataset_df, raw_df, dataset_key, split_id)
                dataset_df["split_id"] = split_id

                if len(feature_cols) == 0:
                    print(f"[SKIP] split={split_id} dataset={dataset_key} representation={representation} has no features after recoding.")
                    continue

                train_mask = dataset_df["case_id"].isin(outer_train_case_df["case_id"])
                test_mask = dataset_df["case_id"].isin(outer_test_case_df["case_id"])
                train_df = dataset_df.loc[train_mask].copy().reset_index(drop=True)
                test_df = dataset_df.loc[test_mask].copy().reset_index(drop=True)

                for spec in model_specs:
                    for target_spec in model_target_specs_for_model(spec):
                        target_name = target_spec["target_name"]
                        target_col = target_spec["target_col"]
                        X_train, y_train, X_test, y_test = prepare_task_frames(
                            train_df=train_df,
                            test_df=test_df,
                            feature_cols=feature_cols,
                            target_col=target_col,
                            task_type=spec.task_type,
                        )
                        if should_skip_model_fit(
                            y_train=y_train,
                            y_test=y_test,
                            task_type=spec.task_type,
                            split_id=split_id,
                            dataset_key=dataset_key,
                            representation=representation,
                            model_key=spec.key,
                            target_name=target_name,
                        ):
                            continue

                        print(
                            f"[MODEL] split={split_id} dataset={dataset_key} "
                            f"representation={representation} model={spec.key} "
                            f"target={target_name} n_features={len(feature_cols)} "
                            f"n_train={len(y_train)} n_test={len(y_test)}"
                        )
                        pred_df, hyper, coef_df = fit_one_outer_model(
                            spec=spec,
                            X_train=X_train,
                            y_train=y_train,
                            X_test=X_test,
                            split_random_seed=args.random_seed + split_num,
                            ml_n_jobs=args.ml_n_jobs,
                        )
                        test_meta = test_df[test_df[target_col].notna()].copy().reset_index(drop=True)
                        meta_pred = test_meta[["case_id", "row_index"]].copy()
                        meta_pred["split_id"] = split_id
                        meta_pred["dataset_key"] = dataset_key
                        meta_pred["representation"] = representation
                        meta_pred["model_key"] = spec.key
                        meta_pred["task_type"] = spec.task_type
                        meta_pred["target_name"] = target_name
                        meta_pred["target_col"] = target_col
                        meta_pred["y_true"] = y_test.values
                        meta_pred = pd.concat([meta_pred.reset_index(drop=True), pred_df.reset_index(drop=True)], axis=1)
                        result["prediction_tables"].append(meta_pred)

                        metrics = classification_metrics(meta_pred) if spec.task_type == "classification" else regression_metrics(meta_pred)
                        metrics.update({
                            "split_id": split_id,
                            "dataset_key": dataset_key,
                            "representation": representation,
                            "model_key": spec.key,
                            "task_type": spec.task_type,
                            "target_name": target_name,
                            "target_col": target_col,
                            "family": spec.family,
                            "estimator_name": spec.estimator_name,
                            "notes": spec.notes,
                        })
                        result["fold_result_tables"].append(pd.DataFrame([metrics]))

                        hyper.update({
                            "split_id": split_id,
                            "dataset_key": dataset_key,
                            "representation": representation,
                            "model_key": spec.key,
                            "task_type": spec.task_type,
                            "target_name": target_name,
                            "target_col": target_col,
                        })
                        result["hyper_tables"].append(pd.DataFrame([hyper]))

                        if len(coef_df):
                            coef_df = annotate_coefficient_table(coef_df, X_train, X_test)
                            coef_df = coef_df.copy()
                            coef_df["split_id"] = split_id
                            coef_df["dataset_key"] = dataset_key
                            coef_df["representation"] = representation
                            coef_df["model_key"] = spec.key
                            coef_df["task_type"] = spec.task_type
                            coef_df["target_name"] = target_name
                            coef_df["target_col"] = target_col
                            result["coef_tables"].append(coef_df)

        if args.enable_teacher_student:
            try:
                representation = "group_status"
                mri_df = modality_results["mri"]["group_matrix_df"].drop_duplicates(subset=["case_id", "row_index"]).reset_index(drop=True)
                path_df = modality_results["path"]["group_matrix_df"].drop_duplicates(subset=["case_id", "row_index"]).reset_index(drop=True)
                X_mri_all, mri_cols = get_representation_matrix(mri_df, representation)
                X_path_all, path_cols = get_representation_matrix(path_df, representation)
                mri_model_df = pd.concat([mri_df[["case_id", "row_index", "dispersion_true", "dispersion_true_high_low", "relapse_true"]], X_mri_all], axis=1)
                mri_model_df = filter_missing_mri_for_dataset(mri_model_df, raw_df, "mri_teacher_student", split_id)
                path_model_df = pd.concat([path_df[["case_id", "row_index", "dispersion_true", "dispersion_true_high_low", "relapse_true"]], X_path_all], axis=1)
                train_mask = mri_model_df["case_id"].isin(outer_train_case_df["case_id"])
                test_mask = mri_model_df["case_id"].isin(outer_test_case_df["case_id"])

                if len(mri_cols) > 0 and len(path_cols) > 0:
                    train_mri = mri_model_df.loc[train_mask].reset_index(drop=True)
                    test_mri = mri_model_df.loc[test_mask].reset_index(drop=True)
                    train_pair = train_mri[["case_id", "row_index"] + mri_cols].merge(
                        path_model_df[["case_id", "row_index"] + path_cols],
                        on=["case_id", "row_index"],
                        how="inner",
                    )
                    train_mri = train_mri.merge(
                        train_pair[["case_id", "row_index"]],
                        on=["case_id", "row_index"],
                        how="inner",
                    ).reset_index(drop=True)
                    train_pair = train_mri[["case_id", "row_index"] + mri_cols].merge(
                        path_model_df[["case_id", "row_index"] + path_cols],
                        on=["case_id", "row_index"],
                        how="inner",
                    )
                    X_mri_train = train_pair[mri_cols]
                    X_mri_test = test_mri[mri_cols]
                    X_path_train = train_pair[path_cols]
                    path_concept_train = train_pair[["case_id", "row_index"] + path_cols].copy()
                    reg_pred, cls_pred, ts_extra = fit_teacher_student_mri_model(
                        X_mri_train=X_mri_train,
                        X_mri_test=X_mri_test,
                        X_path_train=X_path_train,
                        y_train=train_mri["dispersion_true"].astype(float),
                        y_test_cont=test_mri["dispersion_true"].astype(float),
                        y_test_binary=test_mri["dispersion_true_high_low"].astype(int),
                        path_concept_train=path_concept_train,
                        split_random_seed=args.random_seed + split_num,
                        alpha=args.teacher_student_alpha,
                        lambda_dispersion=args.teacher_student_lambda_dispersion,
                        lambda_teacher_score=args.teacher_student_lambda_teacher_score,
                        lambda_path_concepts=args.teacher_student_lambda_path_concepts,
                    )
                    for task_type, pred_df, target_name, target_col in [
                        ("regression", reg_pred, TARGET_NAME_DISPERSION_SCORE, "dispersion_true"),
                        ("classification", cls_pred, TARGET_NAME_DISPERSION_HIGH_LOW, "dispersion_true_high_low"),
                    ]:
                        meta_pred = test_mri[["case_id", "row_index"]].copy()
                        meta_pred["split_id"] = split_id
                        meta_pred["dataset_key"] = "mri_teacher_student"
                        meta_pred["representation"] = representation
                        meta_pred["model_key"] = "multitask_ridge_student"
                        meta_pred["task_type"] = task_type
                        meta_pred["target_name"] = target_name
                        meta_pred["target_col"] = target_col
                        meta_pred = pd.concat([meta_pred.reset_index(drop=True), pred_df.reset_index(drop=True)], axis=1)
                        result["prediction_tables"].append(meta_pred)
                        metrics = regression_metrics(meta_pred) if task_type == "regression" else classification_metrics(meta_pred)
                        metrics.update({
                            "split_id": split_id,
                            "dataset_key": "mri_teacher_student",
                            "representation": representation,
                            "model_key": "multitask_ridge_student",
                            "task_type": task_type,
                            "target_name": target_name,
                            "target_col": target_col,
                            "family": "teacher_student_multitask_ridge",
                            "estimator_name": "Multi-output Ridge student",
                            "notes": "MRI-only inputs trained with true dispersion, pathology teacher score, and pathology concept outputs from outer-train only.",
                        })
                        result["fold_result_tables"].append(pd.DataFrame([metrics]))
                    ts_hyper = ts_extra["hyper"]
                    ts_hyper.update({
                        "split_id": split_id,
                        "dataset_key": "mri_teacher_student",
                        "representation": representation,
                        "model_key": "multitask_ridge_student",
                        "task_type": "regression_and_thresholded_classification",
                        "target_name": TARGET_NAME_DISPERSION_SCORE,
                        "target_col": "dispersion_true",
                    })
                    result["hyper_tables"].append(pd.DataFrame([ts_hyper]))
                    coef_df = ts_extra["coef_df"]
                    if len(coef_df):
                        coef_df = annotate_coefficient_table(coef_df, X_mri_train, X_mri_test)
                        coef_df["split_id"] = split_id
                        coef_df["dataset_key"] = "mri_teacher_student"
                        coef_df["representation"] = representation
                        coef_df["model_key"] = "multitask_ridge_student"
                        coef_df["task_type"] = "regression"
                        coef_df["target_name"] = TARGET_NAME_DISPERSION_SCORE
                        coef_df["target_col"] = "dispersion_true"
                        result["coef_tables"].append(coef_df)
                    print(f"[TEACHER_STUDENT] Completed {split_id} MRI-only multitask student.")
            except Exception as e:
                print(f"[WARN] Teacher-student model failed for {split_id}: {e}")

        if args.resume:
            save_completed_split_checkpoint(
                split_dir=split_dir,
                split_id=split_id,
                args=args,
                all_phrase_freq_tables=result["all_phrase_freq_tables"],
                all_group_freq_tables=result["all_group_freq_tables"],
                all_stable_phrase_tables=result["all_stable_phrase_tables"],
                all_stable_group_tables=result["all_stable_group_tables"],
                all_mri_audit_tables=result["all_mri_audit_tables"],
                all_mri_audit_summary_tables=result["all_mri_audit_summary_tables"],
                all_reliability_tables=result["all_reliability_tables"],
                all_weighted_lexicon_tables=result["all_weighted_lexicon_tables"],
                prediction_tables=result["prediction_tables"],
                fold_result_tables=result["fold_result_tables"],
                hyper_tables=result["hyper_tables"],
                coef_tables=result["coef_tables"],
            )
        return result
    except Exception as e:
        print(f"[ERROR] {split_id} failed with {type(e).__name__}: {e}")
        print(traceback.format_exc())
        result["error_tables"].append(write_failed_split_marker(split_dir, split_id, e))
        return result


def coordinate_parallelism(args: argparse.Namespace) -> argparse.Namespace:
    """Resolve worker controls and prevent obvious CPU/API oversubscription."""
    cpu_count = os.cpu_count() or 1
    args.parallel_fold_workers = max(1, int(getattr(args, "parallel_fold_workers", 1) or 1))
    args.max_api_workers = max(1, int(args.max_api_workers))
    args.parallel_modality_workers = max(1, int(args.parallel_modality_workers))
    args.ml_n_jobs = max(1, int(args.ml_n_jobs))

    # max_api_workers is treated as a process-wide cap enforced by a semaphore in
    # the auxiliary API client, not as a per-fold quota.
    configure_global_api_concurrency(args.max_api_workers)

    requested_cpu_slots = args.parallel_fold_workers * max(1, args.parallel_modality_workers) * max(1, args.ml_n_jobs)
    if requested_cpu_slots > max(1, cpu_count):
        new_ml_n_jobs = max(1, cpu_count // max(1, args.parallel_fold_workers * args.parallel_modality_workers))
        if new_ml_n_jobs < args.ml_n_jobs:
            print(
                f"[PARALLEL] Reducing ml_n_jobs from {args.ml_n_jobs} to {new_ml_n_jobs} "
                f"to avoid oversubscription: fold_workers={args.parallel_fold_workers}, "
                f"modality_workers={args.parallel_modality_workers}, cpus={cpu_count}."
            )
            args.ml_n_jobs = new_ml_n_jobs
    return args


def deduplicate_outer_predictions(pred_all: pd.DataFrame) -> pd.DataFrame:
    """Create one held-out prediction per case/model by averaging repeats.

    Stratified 5-fold CV already yields one held-out prediction per case. For
    repeated Monte Carlo splits, a case can appear in multiple outer-test sets;
    this aggregation keeps the final metric layer aligned with the one-case-one-
    prediction rule while preserving the raw per-split predictions separately.
    """
    if pred_all is None or len(pred_all) == 0:
        return pd.DataFrame()
    df = pred_all.copy()
    group_cols = ["dataset_key", "representation", "model_key", "task_type", "target_name", "target_col", "case_id", "row_index"]
    for col in group_cols:
        if col not in df.columns:
            df[col] = "unknown"
    rows: List[Dict[str, Any]] = []
    for keys, sub in df.groupby(group_cols, dropna=False):
        rec = dict(zip(group_cols, keys))
        rec["n_outer_test_predictions_aggregated"] = int(len(sub))
        rec["split_ids"] = ";".join(sorted(set(sub["split_id"].astype(str)))) if "split_id" in sub.columns else ""
        rec["y_true"] = sub["y_true"].dropna().iloc[0] if "y_true" in sub.columns and sub["y_true"].notna().any() else np.nan
        if str(rec["task_type"]) == "classification":
            rec["y_prob"] = float(pd.to_numeric(sub.get("y_prob", pd.Series(dtype=float)), errors="coerce").mean())
            rec["y_pred"] = int(rec["y_prob"] >= 0.5) if pd.notna(rec["y_prob"]) else np.nan
        else:
            rec["y_pred_value"] = float(pd.to_numeric(sub.get("y_pred_value", pd.Series(dtype=float)), errors="coerce").mean())
        rows.append(rec)
    return pd.DataFrame(rows)


def compute_metrics_from_predictions(pred_df: pd.DataFrame, args: argparse.Namespace, seed_offset: int = 0) -> pd.DataFrame:
    metrics_rows: List[Dict[str, Any]] = []
    if pred_df is None or len(pred_df) == 0:
        return pd.DataFrame()
    group_cols = ["dataset_key", "representation", "model_key", "task_type", "target_name"]
    for group_values, sub in pred_df.groupby(group_cols):
        dataset_key, representation, model_key, task_type, target_name = group_values
        metrics = classification_metrics(sub) if task_type == "classification" else regression_metrics(sub)
        metrics = add_bootstrap_metric_cis(
            base_metrics=metrics,
            pred_df=sub,
            task_type=task_type,
            n_bootstrap=args.bootstrap_n,
            random_seed=args.random_seed + seed_offset,
        )
        metrics.update({
            "dataset_key": dataset_key,
            "representation": representation,
            "model_key": model_key,
            "task_type": task_type,
            "target_name": target_name,
        })
        metrics_rows.append(metrics)
    return pd.DataFrame(metrics_rows)


def _best_row(metrics_df: pd.DataFrame, task_type: str, target_name: Optional[str] = None, dataset_key: Optional[str] = None) -> Optional[pd.Series]:
    if metrics_df is None or len(metrics_df) == 0:
        return None
    df = metrics_df[metrics_df["task_type"].astype(str) == task_type].copy()
    if target_name is not None and "target_name" in df.columns:
        df = df[df["target_name"].astype(str) == str(target_name)].copy()
    if dataset_key is not None:
        df = df[df["dataset_key"].astype(str) == str(dataset_key)].copy()
    if len(df) == 0:
        return None
    if task_type == "regression":
        return df.sort_values(["mae", "spearman_rho"], ascending=[True, False]).iloc[0]
    if target_name == TARGET_NAME_RELAPSE_STATUS:
        return df.sort_values(["auprc", "auroc", "brier"], ascending=[False, False, True]).iloc[0]
    return df.sort_values(["auroc", "auprc", "f1"], ascending=[False, False, False]).iloc[0]


def _df_to_markdown(df: pd.DataFrame, max_rows: int = 25, float_digits: int = 3) -> str:
    if df is None or len(df) == 0:
        return "_No rows available._"
    tmp = df.head(max_rows).copy()
    for col in tmp.columns:
        if pd.api.types.is_float_dtype(tmp[col]):
            tmp[col] = tmp[col].map(lambda x: "" if pd.isna(x) else f"{x:.{float_digits}f}")
    try:
        return tmp.to_markdown(index=False)
    except Exception:
        return tmp.to_csv(index=False)


def _safe_savefig(path: str) -> None:
    _save_figure(path)


def generate_performance_plots(pred_case_df: pd.DataFrame, metrics_df: pd.DataFrame, out_dir: str) -> List[str]:
    plot_dir = os.path.join(out_dir, "report_plots")
    os.makedirs(plot_dir, exist_ok=True)
    paths: List[str] = []
    if pred_case_df is None or len(pred_case_df) == 0 or metrics_df is None or len(metrics_df) == 0:
        return paths

    best_reg = _best_row(metrics_df, "regression", TARGET_NAME_DISPERSION_SCORE)
    if best_reg is not None:
        mask = (
            (pred_case_df["dataset_key"].astype(str) == str(best_reg["dataset_key"]))
            & (pred_case_df["representation"].astype(str) == str(best_reg["representation"]))
            & (pred_case_df["model_key"].astype(str) == str(best_reg["model_key"]))
            & (pred_case_df["task_type"].astype(str) == "regression")
        )
        sub = pred_case_df.loc[mask].copy()
        if len(sub):
            y_true = pd.to_numeric(sub["y_true"], errors="coerce")
            y_pred = pd.to_numeric(sub["y_pred_value"], errors="coerce")
            plt.figure(figsize=(6, 6))
            plt.scatter(y_true, y_pred)
            lo = float(np.nanmin([y_true.min(), y_pred.min()]))
            hi = float(np.nanmax([y_true.max(), y_pred.max()]))
            plt.plot([lo, hi], [lo, hi], linestyle="--")
            plt.xlabel("True dispersion score")
            plt.ylabel("Predicted dispersion score")
            plt.title("Top regression model: predicted vs true")
            path = os.path.join(plot_dir, "top_regression_predicted_vs_true.png")
            _safe_savefig(path); paths.append(path)

            residual = y_pred - y_true
            plt.figure(figsize=(7, 5))
            plt.scatter(y_pred, residual)
            plt.axhline(0, linestyle="--")
            plt.xlabel("Predicted dispersion score")
            plt.ylabel("Residual: predicted - true")
            plt.title("Top regression model residuals")
            path = os.path.join(plot_dir, "top_regression_residuals.png")
            _safe_savefig(path); paths.append(path)

    for target_name in [TARGET_NAME_DISPERSION_HIGH_LOW, TARGET_NAME_RELAPSE_STATUS]:
        best_cls = _best_row(metrics_df, "classification", target_name)
        if best_cls is None:
            continue
        mask = (
            (pred_case_df["dataset_key"].astype(str) == str(best_cls["dataset_key"]))
            & (pred_case_df["representation"].astype(str) == str(best_cls["representation"]))
            & (pred_case_df["model_key"].astype(str) == str(best_cls["model_key"]))
            & (pred_case_df["target_name"].astype(str) == str(target_name))
        )
        sub = pred_case_df.loc[mask].copy()
        if len(sub) == 0:
            continue
        y_true = sub["y_true"].astype(int).values
        prob = sub["y_prob"].astype(float).values
        y_pred = sub["y_pred"].astype(int).values
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        plt.figure(figsize=(5, 4))
        plt.imshow(cm)
        plt.xticks([0, 1], ["Pred 0", "Pred 1"])
        plt.yticks([0, 1], ["True 0", "True 1"])
        for i in range(2):
            for j in range(2):
                plt.text(j, i, str(cm[i, j]), ha="center", va="center")
        plt.title(f"Confusion matrix: {target_name}")
        path = os.path.join(plot_dir, f"top_{target_name}_confusion_matrix.png")
        _safe_savefig(path); paths.append(path)

        if len(np.unique(y_true)) >= 2:
            fpr, tpr, _ = roc_curve(y_true, prob)
            plt.figure(figsize=(6, 5))
            plt.plot(fpr, tpr)
            plt.plot([0, 1], [0, 1], linestyle="--")
            plt.xlabel("False positive rate")
            plt.ylabel("True positive rate")
            plt.title(f"ROC curve: {target_name}")
            path = os.path.join(plot_dir, f"top_{target_name}_roc.png")
            _safe_savefig(path); paths.append(path)

            precision, recall, _ = precision_recall_curve(y_true, prob)
            plt.figure(figsize=(6, 5))
            plt.plot(recall, precision)
            plt.axhline(float(np.mean(y_true)), linestyle="--")
            plt.xlabel("Recall / sensitivity")
            plt.ylabel("Precision / PPV")
            plt.title(f"Precision-recall curve: {target_name}")
            path = os.path.join(plot_dir, f"top_{target_name}_pr.png")
            _safe_savefig(path); paths.append(path)

            bins = np.linspace(0, 1, 6)
            bin_id = np.digitize(prob, bins, right=True)
            cal_rows = []
            for b in sorted(set(bin_id)):
                idx = bin_id == b
                if idx.sum() >= 1:
                    cal_rows.append((float(np.mean(prob[idx])), float(np.mean(y_true[idx])), int(idx.sum())))
            if cal_rows:
                cal = pd.DataFrame(cal_rows, columns=["mean_predicted_risk", "observed_rate", "n"])
                plt.figure(figsize=(6, 5))
                plt.scatter(cal["mean_predicted_risk"], cal["observed_rate"])
                plt.plot([0, 1], [0, 1], linestyle="--")
                for _, r in cal.iterrows():
                    plt.text(r["mean_predicted_risk"], r["observed_rate"], str(int(r["n"])))
                plt.xlabel("Mean predicted risk")
                plt.ylabel("Observed event rate")
                plt.title(f"Calibration plot: {target_name}")
                path = os.path.join(plot_dir, f"top_{target_name}_calibration.png")
                _safe_savefig(path); paths.append(path)

    if len(metrics_df):
        rank_df = metrics_df.copy()
        rank_df["label"] = _metric_plot_label(rank_df)
        for metric, title in [("mae", "Regression MAE"), ("auprc", "Classification AUPRC"), ("auroc", "Classification AUROC")]:
            sub = rank_df[rank_df[metric].notna()].copy() if metric in rank_df.columns else pd.DataFrame()
            if len(sub) == 0:
                continue
            sub = sub.sort_values(metric, ascending=(metric == "mae")).head(25)
            path = os.path.join(plot_dir, f"ranked_model_{metric}.png")
            _plot_metric_bars(
                sub["label"].tolist(),
                sub[metric].astype(float).tolist(),
                ylabel=metric,
                title=title,
                out_png=path,
                ascending=(metric == "mae"),
            )
            paths.append(path)

        ci_rows = []
        for _, r in metrics_df.iterrows():
            primary = "mae" if r.get("task_type") == "regression" else ("auprc" if r.get("target_name") == TARGET_NAME_RELAPSE_STATUS else "auroc")
            if primary in r and f"{primary}_ci_low" in r and pd.notna(r.get(primary)):
                ci_rows.append({
                    "label": f"{r.get('target_name')}|{r.get('dataset_key')}|{r.get('model_key')}"[:80],
                    "metric": primary,
                    "value": r.get(primary),
                    "ci_low": r.get(f"{primary}_ci_low"),
                    "ci_high": r.get(f"{primary}_ci_high"),
                })
        if ci_rows:
            ci = pd.DataFrame(ci_rows).head(30)
            fig_w, fig_h = _metric_bar_figure_size(ci["label"].astype(str).tolist(), horizontal=False)
            fig_h = max(fig_h, 7.0)
            fig, ax = plt.subplots(figsize=(fig_w, fig_h))
            xpos = np.arange(len(ci))
            yerr_low = np.maximum(0.0, ci["value"].astype(float) - ci["ci_low"].astype(float))
            yerr_high = np.maximum(0.0, ci["ci_high"].astype(float) - ci["value"].astype(float))
            ax.errorbar(xpos, ci["value"], yerr=[yerr_low, yerr_high], fmt="o")
            ax.set_xticks(xpos)
            ax.set_xticklabels(ci["label"], rotation=75, ha="right", fontsize=7)
            ax.set_ylabel("Primary metric with 95% bootstrap CI")
            ax.set_title("Bootstrap confidence intervals")
            path = os.path.join(plot_dir, "bootstrap_ci_primary_metrics.png")
            _save_figure(path, fig)
            paths.append(path)
    return paths


def relapse_split_diagnostics(target_df: pd.DataFrame, outer_splits: Sequence[Tuple[np.ndarray, np.ndarray]], out_dir: str) -> pd.DataFrame:
    rows = []
    y_all = pd.to_numeric(target_df.get("relapse_true"), errors="coerce")
    rows.append({"split_id": "overall", "partition": "all", "n": int(y_all.notna().sum()), "relapse_positive": int((y_all == 1).sum()), "relapse_negative": int((y_all == 0).sum()), "prevalence": float((y_all == 1).mean()) if y_all.notna().any() else np.nan, "warning": ""})
    for split_num, (train_pos, test_pos) in enumerate(outer_splits, 1):
        split_id = f"outer_split_{split_num:03d}"
        for partition, pos in [("train", train_pos), ("test", test_pos)]:
            yy = pd.to_numeric(target_df.iloc[pos].get("relapse_true"), errors="coerce")
            n_pos = int((yy == 1).sum())
            n_neg = int((yy == 0).sum())
            warnings = []
            if n_pos == 0 or n_neg == 0:
                warnings.append("single_class_partition")
            if n_pos < 2:
                warnings.append("too_few_relapse_positive_for_stable_auc_or_calibration")
            rows.append({"split_id": split_id, "partition": partition, "n": int(yy.notna().sum()), "relapse_positive": n_pos, "relapse_negative": n_neg, "prevalence": float((yy == 1).mean()) if yy.notna().any() else np.nan, "warning": ";".join(warnings)})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(out_dir, "relapse_class_balance_by_split.csv"), index=False)
    return out


def permutation_test_relapse_metrics(pred_case_df: pd.DataFrame, metrics_df: pd.DataFrame, args: argparse.Namespace, out_dir: str) -> pd.DataFrame:
    n_perm = int(getattr(args, "relapse_permutation_n", 0) or 0)
    if n_perm <= 0 or pred_case_df is None or len(pred_case_df) == 0:
        return pd.DataFrame()
    rows = []
    rng = np.random.default_rng(int(args.random_seed) + 991)
    rel_metrics = metrics_df[(metrics_df["task_type"] == "classification") & (metrics_df["target_name"] == TARGET_NAME_RELAPSE_STATUS)].copy() if len(metrics_df) else pd.DataFrame()
    if len(rel_metrics) == 0:
        return pd.DataFrame()
    for _, m in rel_metrics.iterrows():
        mask = (
            (pred_case_df["dataset_key"].astype(str) == str(m["dataset_key"]))
            & (pred_case_df["representation"].astype(str) == str(m["representation"]))
            & (pred_case_df["model_key"].astype(str) == str(m["model_key"]))
            & (pred_case_df["target_name"].astype(str) == TARGET_NAME_RELAPSE_STATUS)
        )
        sub = pred_case_df.loc[mask].copy()
        if len(sub) < 4 or sub["y_true"].nunique() < 2:
            continue
        y = sub["y_true"].astype(int).values
        prob = sub["y_prob"].astype(float).values
        obs_auroc = roc_auc_score(y, prob)
        obs_auprc = average_precision_score(y, prob)
        null_auroc = []
        null_auprc = []
        for _ in range(n_perm):
            yp = rng.permutation(y)
            if len(np.unique(yp)) < 2:
                continue
            null_auroc.append(float(roc_auc_score(yp, prob)))
            null_auprc.append(float(average_precision_score(yp, prob)))
        rows.append({
            "dataset_key": m["dataset_key"],
            "representation": m["representation"],
            "model_key": m["model_key"],
            "n": len(sub),
            "n_permutations": len(null_auroc),
            "observed_auroc": obs_auroc,
            "observed_auprc": obs_auprc,
            "auroc_empirical_p": float((1 + np.sum(np.asarray(null_auroc) >= obs_auroc)) / (1 + len(null_auroc))) if null_auroc else np.nan,
            "auprc_empirical_p": float((1 + np.sum(np.asarray(null_auprc) >= obs_auprc)) / (1 + len(null_auprc))) if null_auprc else np.nan,
            "null_auroc_mean": float(np.mean(null_auroc)) if null_auroc else np.nan,
            "null_auprc_mean": float(np.mean(null_auprc)) if null_auprc else np.nan,
        })
    out = pd.DataFrame(rows)
    if len(out):
        out.to_csv(os.path.join(out_dir, "relapse_permutation_tests.csv"), index=False)
    return out


def generate_error_analysis(pred_case_df: pd.DataFrame, metrics_df: pd.DataFrame, raw_df: pd.DataFrame, out_dir: str) -> Tuple[pd.DataFrame, str]:
    rows: List[Dict[str, Any]] = []
    if pred_case_df is None or len(pred_case_df) == 0 or metrics_df is None or len(metrics_df) == 0:
        return pd.DataFrame(), ""
    raw = _raw_df_with_row_index(raw_df)
    raw_flags = raw[["case_id", "row_index", "preop_MRI_text", "path_report_text"]].copy()
    raw_flags["mri_report_missing"] = raw_flags["preop_MRI_text"].apply(_is_missing_text).astype(int)
    raw_flags["path_report_missing"] = raw_flags["path_report_text"].apply(_is_missing_text).astype(int)
    raw_flags["mri_report_chars"] = raw_flags["preop_MRI_text"].fillna("").astype(str).str.len()
    raw_flags = raw_flags.drop(columns=["preop_MRI_text", "path_report_text"])

    best_reg = _best_row(metrics_df, "regression", TARGET_NAME_DISPERSION_SCORE)
    if best_reg is not None:
        sub = pred_case_df[
            (pred_case_df["dataset_key"].astype(str) == str(best_reg["dataset_key"]))
            & (pred_case_df["representation"].astype(str) == str(best_reg["representation"]))
            & (pred_case_df["model_key"].astype(str) == str(best_reg["model_key"]))
            & (pred_case_df["task_type"].astype(str) == "regression")
        ].copy()
        if len(sub):
            sub["residual"] = pd.to_numeric(sub["y_pred_value"], errors="coerce") - pd.to_numeric(sub["y_true"], errors="coerce")
            sd = float(sub["residual"].std(ddof=0) or 1.0)
            sub["abs_residual"] = sub["residual"].abs()
            sub["standardized_residual"] = sub["residual"] / max(sd, EPS)
            for _, r in sub.sort_values("abs_residual", ascending=False).head(20).iterrows():
                rows.append({**r.to_dict(), "error_task": "dispersion_regression", "error_type": "strong_overprediction" if r["residual"] > 0 else "strong_underprediction", "model_summary": f"{best_reg['dataset_key']}|{best_reg['representation']}|{best_reg['model_key']}"})

    for target_name in [TARGET_NAME_DISPERSION_HIGH_LOW, TARGET_NAME_RELAPSE_STATUS]:
        best_cls = _best_row(metrics_df, "classification", target_name)
        if best_cls is None:
            continue
        sub = pred_case_df[
            (pred_case_df["dataset_key"].astype(str) == str(best_cls["dataset_key"]))
            & (pred_case_df["representation"].astype(str) == str(best_cls["representation"]))
            & (pred_case_df["model_key"].astype(str) == str(best_cls["model_key"]))
            & (pred_case_df["target_name"].astype(str) == str(target_name))
        ].copy()
        if len(sub) == 0:
            continue
        sub["confidence"] = (pd.to_numeric(sub["y_prob"], errors="coerce") - 0.5).abs() * 2.0
        sub["correct"] = sub["y_true"].astype(int) == sub["y_pred"].astype(int)
        for _, r in sub[(sub["y_true"] == 0) & (sub["y_pred"] == 1)].sort_values("confidence", ascending=False).head(10).iterrows():
            rows.append({**r.to_dict(), "error_task": target_name, "error_type": "false_positive", "model_summary": f"{best_cls['dataset_key']}|{best_cls['representation']}|{best_cls['model_key']}"})
        for _, r in sub[(sub["y_true"] == 1) & (sub["y_pred"] == 0)].sort_values("confidence", ascending=False).head(10).iterrows():
            rows.append({**r.to_dict(), "error_task": target_name, "error_type": "false_negative", "model_summary": f"{best_cls['dataset_key']}|{best_cls['representation']}|{best_cls['model_key']}"})
        for _, r in sub[sub["correct"]].sort_values("confidence", ascending=True).head(10).iterrows():
            rows.append({**r.to_dict(), "error_task": target_name, "error_type": "low_confidence_correct", "model_summary": f"{best_cls['dataset_key']}|{best_cls['representation']}|{best_cls['model_key']}"})
        for _, r in sub[~sub["correct"]].sort_values("confidence", ascending=False).head(10).iterrows():
            rows.append({**r.to_dict(), "error_task": target_name, "error_type": "high_confidence_incorrect", "model_summary": f"{best_cls['dataset_key']}|{best_cls['representation']}|{best_cls['model_key']}"})

    out = pd.DataFrame(rows)
    if len(out):
        out = out.merge(raw_flags, on=["case_id", "row_index"], how="left")
        out["likely_failure_modes"] = out.apply(lambda r: ";".join([
            "missing_mri_report" if int(r.get("mri_report_missing", 0) or 0) == 1 else "",
            "sparse_mri_language" if int(r.get("mri_report_chars", 9999) or 9999) < 800 and str(r.get("dataset_key", "")).startswith("mri") else "",
            "near_dispersion_threshold" if abs(float(r.get("y_true", np.nan)) - DISPERSION_TRUE_HIGH_THRESHOLD) <= 10 and r.get("error_task") == "dispersion_regression" else "",
            "low_confidence_boundary_case" if str(r.get("error_type", "")).startswith("low_confidence") else "",
        ]).strip(";"), axis=1)
        out["likely_failure_modes"] = out["likely_failure_modes"].replace("", "requires_manual_review")
        out_path = os.path.join(out_dir, "missed_case_error_analysis.csv")
        out.to_csv(out_path, index=False)
    else:
        out_path = ""
    md_path = os.path.join(out_dir, "missed_case_error_analysis.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Missed-case and Error Analysis\n\n")
        f.write("Cases are highlighted from the top aggregate held-out models after one-prediction-per-case deduplication.\n\n")
        f.write(_df_to_markdown(out.head(50) if len(out) else out, max_rows=50))
        f.write("\n")
    return out, md_path


def generate_interpretability_report(
    out_dir: str,
    coef_all: pd.DataFrame,
    sign_stability_df: pd.DataFrame,
    phrase_freq_all: pd.DataFrame,
    group_freq_all: pd.DataFrame,
    stable_phrase_summary: pd.DataFrame,
    stable_group_summary: pd.DataFrame,
    reliability_all: pd.DataFrame,
    weighted_lexicon_all: pd.DataFrame,
) -> Tuple[str, str]:
    md_path = os.path.join(out_dir, "interpretability_report.md")
    html_path = os.path.join(out_dir, "interpretability_report.html")
    report_dir = os.path.join(out_dir, "interpretability_plots")
    os.makedirs(report_dir, exist_ok=True)

    density_rows = []
    if len(phrase_freq_all):
        for mode, sub in phrase_freq_all.groupby("report_mode"):
            density_rows.append({"report_mode": mode, "n_candidate_phrases": sub["phrase_slug"].nunique() if "phrase_slug" in sub.columns else len(sub), "n_stable_phrases": int((sub.get("stable", pd.Series(dtype=int)) == 1).sum()) if "stable" in sub.columns else np.nan})
    if len(group_freq_all):
        for mode, sub in group_freq_all.groupby("report_mode"):
            match = next((r for r in density_rows if r["report_mode"] == mode), None)
            if match is None:
                match = {"report_mode": mode}; density_rows.append(match)
            match["n_candidate_groups"] = sub["canonical_group"].nunique() if "canonical_group" in sub.columns else len(sub)
            match["n_stable_groups"] = int((sub.get("stable", pd.Series(dtype=int)) == 1).sum()) if "stable" in sub.columns else np.nan
    density_df = pd.DataFrame(density_rows)
    if len(density_df):
        density_df.to_csv(os.path.join(out_dir, "feature_density_summary_by_modality.csv"), index=False)

    top_coef = pd.DataFrame()
    if len(coef_all):
        top_coef = coef_all.copy()
        top_coef = top_coef.sort_values("abs_coef", ascending=False).head(200)
        top_coef.to_csv(os.path.join(out_dir, "top_model_coefficients_interpretability.csv"), index=False)

    if len(sign_stability_df):
        top_stab = sign_stability_df.sort_values(["n_nonzero", "sign_consistency", "mean_abs_coef"], ascending=[False, False, False]).head(40)
        fig_h = max(5.0, min(0.35 * len(top_stab) + 2.0, 18.0))
        fig, ax = plt.subplots(figsize=(10, fig_h))
        ax.barh(top_stab["feature"].astype(str), top_stab["sign_consistency"].astype(float))
        ax.set_xlabel("Fold-level sign consistency")
        ax.set_ylabel("Feature")
        ax.set_title("Coefficient sign stability across folds")
        ax.invert_yaxis()
        _save_figure(os.path.join(report_dir, "coefficient_sign_stability.png"), fig)

    if len(weighted_lexicon_all):
        weight_col = "final_weight" if "final_weight" in weighted_lexicon_all.columns else (
            "weight_normalized" if "weight_normalized" in weighted_lexicon_all.columns else (
                "weight" if "weight" in weighted_lexicon_all.columns else None
            )
        )
        concept_col = "mri_concept" if "mri_concept" in weighted_lexicon_all.columns else weighted_lexicon_all.columns[0]
        if weight_col is not None:
            top_w = weighted_lexicon_all.groupby(concept_col, as_index=False)[weight_col].mean()
            top_w = top_w.sort_values(weight_col, ascending=False).head(30)
            fig_w, fig_h = _metric_bar_figure_size(top_w[concept_col].astype(str).tolist(), horizontal=True)
            fig, ax = plt.subplots(figsize=(fig_w, fig_h))
            ypos = np.arange(len(top_w))
            ax.barh(ypos, top_w[weight_col].astype(float))
            ax.set_yticks(ypos)
            ax.set_yticklabels(top_w[concept_col].astype(str), fontsize=8)
            ax.invert_yaxis()
            ax.set_xlabel("Calibration-derived MRI concept weight")
            ax.set_title("Top pathology-calibrated MRI concept weights")
            _save_figure(os.path.join(report_dir, "weighted_mri_concepts.png"), fig)

    if len(reliability_all):
        val_col = "delta_p_path_given_mri" if "delta_p_path_given_mri" in reliability_all.columns else ("lift" if "lift" in reliability_all.columns else None)
        if val_col and {"mri_concept", "path_concept"}.issubset(reliability_all.columns):
            piv = reliability_all.pivot_table(index="mri_concept", columns="path_concept", values=val_col, aggfunc="mean")
            if len(piv):
                fig_w = max(10.0, min(0.45 * len(piv.columns) + 4.0, 20.0))
                fig_h = max(8.0, min(0.45 * len(piv.index) + 3.0, 16.0))
                fig, ax = plt.subplots(figsize=(fig_w, fig_h))
                im = ax.imshow(piv.fillna(0).values, aspect="auto")
                ax.set_xticks(range(len(piv.columns)))
                ax.set_xticklabels(piv.columns, rotation=75, ha="right", fontsize=8)
                ax.set_yticks(range(len(piv.index)))
                ax.set_yticklabels(piv.index, fontsize=8)
                ax.set_title("MRI-pathology concept reliability matrix")
                fig.colorbar(im, ax=ax, label=val_col, fraction=0.046, pad=0.04)
                _save_figure(os.path.join(report_dir, "mri_pathology_reliability_heatmap.png"), fig)

    if len(coef_all):
        comp = coef_all.copy()
        comp = comp.groupby(["feature", "target_name"]).agg(mean_abs_coef=("abs_coef", "mean")).reset_index()
        piv = comp.pivot_table(index="feature", columns="target_name", values="mean_abs_coef", aggfunc="mean").fillna(0)
        if TARGET_NAME_RELAPSE_STATUS in piv.columns and TARGET_NAME_DISPERSION_SCORE in piv.columns:
            piv["relapse_minus_dispersion_abscoef"] = piv[TARGET_NAME_RELAPSE_STATUS] - piv[TARGET_NAME_DISPERSION_SCORE]
            piv.sort_values("relapse_minus_dispersion_abscoef", ascending=False).to_csv(os.path.join(out_dir, "feature_association_dispersion_vs_relapse.csv"))

    md = []
    md.append("# Interpretability Report\n")
    md.append("This report summarizes extracted lexical feature density, stable lexicons, fitted model coefficients, coefficient stability, pathology-MRI reliability, and pathology-calibrated MRI weights. MRI and pathology are not forced to have equal feature counts; any matched-budget comparison should be interpreted as a sensitivity analysis, not the primary extraction.\n")
    md.append("## Feature-density summary\n")
    md.append(_df_to_markdown(density_df, max_rows=50))
    md.append("\n## Stable phrase summary\n")
    md.append(_df_to_markdown(stable_phrase_summary, max_rows=25))
    md.append("\n## Stable group summary\n")
    md.append(_df_to_markdown(stable_group_summary, max_rows=25))
    md.append("\n## Top coefficients\n")
    cols = [c for c in ["dataset_key", "representation", "model_key", "target_name", "feature", "coef", "abs_coef", "coef_sign", "feature_prevalence_train", "feature_modality", "ontology_concept"] if c in top_coef.columns]
    md.append(_df_to_markdown(top_coef[cols] if len(top_coef) and cols else top_coef, max_rows=50))
    md.append("\n## Sign stability\n")
    md.append(_df_to_markdown(sign_stability_df.head(50) if len(sign_stability_df) else sign_stability_df, max_rows=50))
    md.append("\n## Pathology-calibrated MRI weights\n")
    md.append(_df_to_markdown(weighted_lexicon_all.head(50) if len(weighted_lexicon_all) else weighted_lexicon_all, max_rows=50))
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    interp_plot_map = {
        "coefficient_sign_stability.png": os.path.join(report_dir, "coefficient_sign_stability.png"),
        "mri_pathology_reliability_heatmap.png": os.path.join(report_dir, "mri_pathology_reliability_heatmap.png"),
        "weighted_mri_concepts.png": os.path.join(report_dir, "weighted_mri_concepts.png"),
    }

    def _interp_plot_html(filename: str, title: str) -> str:
        plot_path = interp_plot_map.get(filename, "")
        if plot_path and os.path.exists(plot_path):
            return html_plot_block(plot_path, f"interpretability_plots/{filename}", title=title)
        return ""

    html_sections = [
        html_section("Feature-density summary", [df_to_html_table(density_df, max_rows=50)]),
        html_section("Stable phrase summary", [df_to_html_table(stable_phrase_summary, max_rows=25)]),
        html_section("Stable group summary", [df_to_html_table(stable_group_summary, max_rows=25)]),
        html_section(
            "Top coefficients",
            [df_to_html_table(top_coef[cols] if len(top_coef) and cols else top_coef, max_rows=50)],
        ),
        html_section(
            "Sign stability",
            [
                df_to_html_table(sign_stability_df.head(50) if len(sign_stability_df) else sign_stability_df, max_rows=50),
                _interp_plot_html("coefficient_sign_stability.png", "Coefficient sign stability"),
            ],
        ),
        html_section(
            "MRI-pathology reliability",
            [
                _interp_plot_html("mri_pathology_reliability_heatmap.png", "MRI-pathology concept reliability")
                or html_paragraph("No reliability heatmap was generated for this run."),
            ],
        ),
        html_section(
            "Pathology-calibrated MRI weights",
            [
                df_to_html_table(weighted_lexicon_all.head(50) if len(weighted_lexicon_all) else weighted_lexicon_all, max_rows=50),
                _interp_plot_html("weighted_mri_concepts.png", "Top pathology-calibrated MRI concept weights"),
            ],
        ),
    ]
    intro = (
        "This report summarizes extracted lexical feature density, stable lexicons, fitted model coefficients, "
        "coefficient stability, pathology-MRI reliability, and pathology-calibrated MRI weights. MRI and pathology "
        "are not forced to have equal feature counts; any matched-budget comparison should be interpreted as a "
        "sensitivity analysis, not the primary extraction."
    )
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(build_html_report("Interpretability Report", intro, html_sections))
    return md_path, html_path


def generate_results_report(
    out_dir: str,
    metrics_df: pd.DataFrame,
    pred_case_df: pd.DataFrame,
    fold_results_all: pd.DataFrame,
    relapse_balance_df: pd.DataFrame,
    permutation_df: pd.DataFrame,
    plot_paths: Sequence[str],
    path_mri_subset_metrics_df: pd.DataFrame,
) -> Tuple[str, str]:
    md_path = os.path.join(out_dir, "automated_results_report.md")
    html_path = os.path.join(out_dir, "automated_results_report.html")
    lines = []
    lines.append("# Automated Model Performance Report\n")
    lines.append("All aggregate metrics in this report are computed from held-out outer-test predictions after case-level deduplication, so each case contributes at most one prediction per dataset / representation / model / target. Raw per-split predictions are saved separately.\n")
    lines.append("## Top model summary\n")
    best_specs = [
        ("Top MRI-only regression", _best_row(metrics_df, "regression", TARGET_NAME_DISPERSION_SCORE, "mri")),
        ("Top pathology-only regression", _best_row(metrics_df, "regression", TARGET_NAME_DISPERSION_SCORE, "path")),
        ("Top combined regression", _best_row(metrics_df, "regression", TARGET_NAME_DISPERSION_SCORE, "combined")),
        ("Top overall regression", _best_row(metrics_df, "regression", TARGET_NAME_DISPERSION_SCORE)),
        ("Top high/low classification", _best_row(metrics_df, "classification", TARGET_NAME_DISPERSION_HIGH_LOW)),
        ("Top relapse classification", _best_row(metrics_df, "classification", TARGET_NAME_RELAPSE_STATUS)),
    ]
    summary_rows = []
    for label, row in best_specs:
        if row is None:
            continue
        summary_rows.append({"selection": label, "dataset_key": row.get("dataset_key"), "representation": row.get("representation"), "model_key": row.get("model_key"), "target_name": row.get("target_name"), "mae": row.get("mae", np.nan), "spearman_rho": row.get("spearman_rho", np.nan), "auroc": row.get("auroc", np.nan), "auprc": row.get("auprc", np.nan), "f1": row.get("f1", np.nan), "brier": row.get("brier", np.nan)})
    lines.append(_df_to_markdown(pd.DataFrame(summary_rows), max_rows=20))

    lines.append("\n## Aggregate held-out metrics\n")
    preferred_cols = [c for c in ["target_name", "dataset_key", "representation", "model_key", "n", "mae", "mae_ci_low", "mae_ci_high", "rmse", "pearson_r", "spearman_rho", "r2", "accuracy", "balanced_accuracy", "f1", "auroc", "auprc", "auprc_no_skill_baseline", "brier", "precision_ppv", "recall_sensitivity", "specificity", "npv", "tn", "fp", "fn", "tp", "prevalence"] if c in metrics_df.columns]
    lines.append(_df_to_markdown(metrics_df[preferred_cols].sort_values(["target_name", "dataset_key"]) if len(metrics_df) and preferred_cols else metrics_df, max_rows=100))

    lines.append("\n## Relapse imbalance diagnostics\n")
    lines.append(_df_to_markdown(relapse_balance_df, max_rows=50))
    lines.append("\nAUPRC no-skill baselines are included in the aggregate metrics table and equal the event prevalence for the evaluated prediction set.\n")

    if len(permutation_df):
        lines.append("\n## Relapse permutation tests\n")
        lines.append(_df_to_markdown(permutation_df.sort_values("auprc_empirical_p"), max_rows=50))

    if len(path_mri_subset_metrics_df):
        lines.append("\n## Pathology-only full-cohort vs MRI-complete sensitivity\n")
        lines.append("Pathology-only models can use all target-eligible cases, while MRI-derived and combined models exclude MRI-missing cases. The table below recomputes pathology-only aggregate metrics on MRI-complete cases for comparison.\n")
        lines.append(_df_to_markdown(path_mri_subset_metrics_df, max_rows=50))

    lines.append("\n## Generated plots\n")
    for path in plot_paths:
        rel = os.path.relpath(path, out_dir)
        lines.append(f"- `{rel}`")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    summary_df = pd.DataFrame(summary_rows)
    metrics_table_df = metrics_df[preferred_cols].sort_values(["target_name", "dataset_key"]) if len(metrics_df) and preferred_cols else metrics_df

    plot_groups: List[Tuple[str, List[str]]] = [
        ("Regression performance", [
            "top_regression_predicted_vs_true.png",
            "top_regression_residuals.png",
            "nested_regression_error_comparison.png",
            "nested_regression_correlation_comparison.png",
            "ranked_model_mae.png",
        ]),
        ("Dispersion high/low classification", [
            "top_dispersion_high_low_confusion_matrix.png",
            "top_dispersion_high_low_roc.png",
            "top_dispersion_high_low_pr.png",
            "top_dispersion_high_low_calibration.png",
            "nested_classification_comparison.png",
            "ranked_model_auroc.png",
        ]),
        ("Relapse classification", [
            "top_relapse_status_confusion_matrix.png",
            "top_relapse_status_roc.png",
            "top_relapse_status_pr.png",
            "top_relapse_status_calibration.png",
            "nested_relapse_auroc_comparison.png",
            "nested_relapse_auprc_comparison.png",
            "nested_relapse_f1_comparison.png",
            "nested_relapse_brier_comparison.png",
            "nested_relapse_roc_curves_top_models.png",
            "nested_relapse_pr_curves_top_models.png",
            "ranked_model_auprc.png",
        ]),
        ("Uncertainty and model ranking", [
            "bootstrap_ci_primary_metrics.png",
        ]),
    ]
    plot_path_by_name = {os.path.basename(p): p for p in plot_paths if os.path.exists(p)}
    plotted = set()
    plot_sections: List[str] = []
    for group_title, filenames in plot_groups:
        blocks = []
        for filename in filenames:
            plot_path = plot_path_by_name.get(filename)
            if plot_path is None:
                continue
            plotted.add(filename)
            rel = os.path.relpath(plot_path, out_dir)
            blocks.append(html_plot_block(plot_path, rel))
        if blocks:
            plot_sections.append(html_section(group_title, blocks))
    remaining = [
        html_plot_block(path, os.path.relpath(path, out_dir))
        for name, path in sorted(plot_path_by_name.items())
        if name not in plotted
    ]
    if remaining:
        plot_sections.append(html_section("Additional figures", remaining))

    html_sections = [
        html_section("Top model summary", [df_to_html_table(summary_df, max_rows=20)]),
        html_section("Aggregate held-out metrics", [df_to_html_table(metrics_table_df, max_rows=100)]),
        html_section("Relapse imbalance diagnostics", [
            df_to_html_table(relapse_balance_df, max_rows=50),
            html_paragraph(
                "AUPRC no-skill baselines are included in the aggregate metrics table and equal the event prevalence for the evaluated prediction set."
            ),
        ]),
    ]
    if len(permutation_df):
        html_sections.append(
            html_section(
                "Relapse permutation tests",
                [df_to_html_table(permutation_df.sort_values("auprc_empirical_p"), max_rows=50)],
            )
        )
    if len(path_mri_subset_metrics_df):
        html_sections.append(
            html_section(
                "Pathology-only full-cohort vs MRI-complete sensitivity",
                [
                    html_paragraph(
                        "Pathology-only models can use all target-eligible cases, while MRI-derived and combined models exclude MRI-missing cases. The table below recomputes pathology-only aggregate metrics on MRI-complete cases for comparison."
                    ),
                    df_to_html_table(path_mri_subset_metrics_df, max_rows=50),
                ],
            )
        )
    html_sections.extend(plot_sections)
    intro = (
        "All aggregate metrics in this report are computed from held-out outer-test predictions after case-level "
        "deduplication, so each case contributes at most one prediction per dataset / representation / model / target. "
        "Raw per-split predictions are saved separately."
    )
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(build_html_report("Automated Model Performance Report", intro, html_sections))
    return md_path, html_path


def regenerate_reports_and_plots(out_dir: str, args: argparse.Namespace) -> None:
    """Rebuild plots and HTML reports from saved nested-evaluation artifacts."""
    metrics_path = os.path.join(out_dir, "nested_outer_metrics_summary.csv")
    pred_path = os.path.join(out_dir, "nested_outer_predictions_case_deduplicated.csv")
    if not os.path.exists(metrics_path) or not os.path.exists(pred_path):
        raise FileNotFoundError(
            f"Expected {metrics_path} and {pred_path} for report regeneration."
        )

    metrics_df = pd.read_csv(metrics_path)
    pred_case_all = pd.read_csv(pred_path)
    coef_path = os.path.join(out_dir, "nested_outer_feature_coefficients_all.csv")
    coef_all = pd.read_csv(coef_path) if os.path.exists(coef_path) else pd.DataFrame()
    sign_stability_path = os.path.join(out_dir, "nested_feature_sign_stability.csv")
    sign_stability_df = pd.read_csv(sign_stability_path) if os.path.exists(sign_stability_path) else pd.DataFrame()
    phrase_freq_path = os.path.join(out_dir, "all_outer_phrase_rediscovery_frequencies.csv")
    phrase_freq_all = pd.read_csv(phrase_freq_path) if os.path.exists(phrase_freq_path) else pd.DataFrame()
    group_freq_path = os.path.join(out_dir, "all_outer_group_rediscovery_frequencies.csv")
    group_freq_all = pd.read_csv(group_freq_path) if os.path.exists(group_freq_path) else pd.DataFrame()
    stable_phrase_path = os.path.join(out_dir, "stable_phrase_lexicon_outer_summary.csv")
    stable_phrase_summary = pd.read_csv(stable_phrase_path) if os.path.exists(stable_phrase_path) else pd.DataFrame()
    stable_group_path = os.path.join(out_dir, "stable_group_lexicon_outer_summary.csv")
    stable_group_summary = pd.read_csv(stable_group_path) if os.path.exists(stable_group_path) else pd.DataFrame()
    reliability_path = os.path.join(out_dir, "all_outer_mri_pathology_reliability_matrices.csv")
    reliability_all = pd.read_csv(reliability_path) if os.path.exists(reliability_path) else pd.DataFrame()
    weighted_path = os.path.join(out_dir, "all_outer_weighted_mri_lexicons.csv")
    weighted_lexicon_all = pd.read_csv(weighted_path) if os.path.exists(weighted_path) else pd.DataFrame()
    relapse_balance_path = os.path.join(out_dir, "relapse_class_balance_by_split.csv")
    relapse_balance_df = pd.read_csv(relapse_balance_path) if os.path.exists(relapse_balance_path) else pd.DataFrame()
    permutation_path = os.path.join(out_dir, "relapse_permutation_tests.csv")
    permutation_df = pd.read_csv(permutation_path) if os.path.exists(permutation_path) else pd.DataFrame()
    path_subset_path = os.path.join(out_dir, "pathology_only_mri_complete_subset_metrics.csv")
    path_mri_subset_metrics_df = pd.read_csv(path_subset_path) if os.path.exists(path_subset_path) else pd.DataFrame()

    interp_md, interp_html = generate_interpretability_report(
        out_dir=out_dir,
        coef_all=coef_all,
        sign_stability_df=sign_stability_df,
        phrase_freq_all=phrase_freq_all,
        group_freq_all=group_freq_all,
        stable_phrase_summary=stable_phrase_summary,
        stable_group_summary=stable_group_summary,
        reliability_all=reliability_all,
        weighted_lexicon_all=weighted_lexicon_all,
    )
    print(f"[SAVE] Wrote interpretability reports: {interp_md}, {interp_html}")

    cls_plot_png = os.path.join(out_dir, "nested_classification_comparison.png")
    relapse_auroc_plot_png = os.path.join(out_dir, "nested_relapse_auroc_comparison.png")
    relapse_auprc_plot_png = os.path.join(out_dir, "nested_relapse_auprc_comparison.png")
    relapse_f1_plot_png = os.path.join(out_dir, "nested_relapse_f1_comparison.png")
    relapse_brier_plot_png = os.path.join(out_dir, "nested_relapse_brier_comparison.png")
    reg_err_plot_png = os.path.join(out_dir, "nested_regression_error_comparison.png")
    reg_corr_plot_png = os.path.join(out_dir, "nested_regression_correlation_comparison.png")
    plot_classification_comparison(metrics_df, cls_plot_png)
    plot_relapse_metric_comparison(metrics_df, relapse_auroc_plot_png, "auroc")
    plot_relapse_metric_comparison(metrics_df, relapse_auprc_plot_png, "auprc")
    plot_relapse_metric_comparison(metrics_df, relapse_f1_plot_png, "f1")
    plot_relapse_metric_comparison(metrics_df, relapse_brier_plot_png, "brier")
    plot_relapse_curves(pred_case_all, metrics_df, out_dir)
    plot_regression_error_comparison(metrics_df, reg_err_plot_png)
    plot_regression_correlation_comparison(metrics_df, reg_corr_plot_png)
    report_plot_paths = generate_performance_plots(pred_case_all, metrics_df, out_dir)
    for pth in [
        cls_plot_png,
        relapse_auroc_plot_png,
        relapse_auprc_plot_png,
        relapse_f1_plot_png,
        relapse_brier_plot_png,
        reg_err_plot_png,
        reg_corr_plot_png,
        os.path.join(out_dir, "nested_relapse_roc_curves_top_models.png"),
        os.path.join(out_dir, "nested_relapse_pr_curves_top_models.png"),
    ]:
        if os.path.exists(pth):
            report_plot_paths.append(pth)
    fold_results_path = os.path.join(out_dir, "nested_outer_fold_metrics_all.csv")
    fold_results_all = pd.read_csv(fold_results_path) if os.path.exists(fold_results_path) else pd.DataFrame()
    results_md, results_html = generate_results_report(
        out_dir=out_dir,
        metrics_df=metrics_df,
        pred_case_df=pred_case_all,
        fold_results_all=fold_results_all,
        relapse_balance_df=relapse_balance_df,
        permutation_df=permutation_df,
        plot_paths=report_plot_paths,
        path_mri_subset_metrics_df=path_mri_subset_metrics_df,
    )
    print(f"[SAVE] Wrote automated results reports: {results_md}, {results_html}")


def pathology_metrics_on_mri_complete(pred_case_df: pd.DataFrame, raw_df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if pred_case_df is None or len(pred_case_df) == 0:
        return pd.DataFrame()
    missing_rows = _mri_missing_row_indices(raw_df)
    df = pred_case_df[(pred_case_df["dataset_key"].astype(str) == "path") & (~pred_case_df["row_index"].astype(int).isin(missing_rows))].copy()
    if len(df) == 0:
        return pd.DataFrame()
    out = compute_metrics_from_predictions(df, args, seed_offset=404)
    if len(out):
        out["comparison_subset"] = "pathology_only_on_mri_complete_cases"
    return out


# -----------------------------
# Main evaluation driver
# -----------------------------

def _add_bool_arg(parser: argparse.ArgumentParser, name: str, default: bool, help_text: str) -> None:
    dest = name.replace("-", "_")
    # For names such as enable-pathology-calibration, expose the natural pair
    # --enable-pathology-calibration / --disable-pathology-calibration.
    disable_name = name[len("enable-"):] if name.startswith("enable-") else name
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(f"--{name}", dest=dest, action="store_true", help=help_text)
    group.add_argument(f"--disable-{disable_name}", dest=dest, action="store_false", help=f"Disable: {help_text}")
    parser.set_defaults(**{dest: default})


def planned_llm_extraction_modes(args: argparse.Namespace) -> List[str]:
    """Return report modes requiring LLM extraction for this run."""
    return [
        mode for mode in ["mri", "path"]
        if (
            mode in args.modalities
            or "combined" in args.modalities
            or args.enable_pathology_calibration
            or args.enable_teacher_student
        )
    ]


def estimate_nested_pipeline_llm_cost(
    raw_df: pd.DataFrame,
    target_df: pd.DataFrame,
    outer_splits: Sequence[Tuple[np.ndarray, np.ndarray]],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """Build the exact prompts scheduled by nested extraction and estimate cost."""
    prompt_counts: List[int] = []
    prompt_modes: List[str] = []
    skipped_completed_splits = 0
    modes_to_run = planned_llm_extraction_modes(args)

    for split_num, (train_pos, _test_pos) in enumerate(outer_splits, 1):
        split_id = f"outer_split_{split_num:03d}"
        split_dir = os.path.join(args.out_dir, "outer_splits", split_id)
        if args.resume and args.skip_completed_splits and os.path.exists(_split_resume_marker(split_dir)):
            skipped_completed_splits += 1
            continue

        outer_train_case_df = target_df.iloc[train_pos].copy().reset_index(drop=True)
        for mode in modes_to_run:
            for row_index in outer_train_case_df["row_index"].astype(int).tolist():
                case = make_case_from_row(raw_df, int(row_index))
                if _is_missing_text(_selected_report_text(case, mode)):
                    continue
                prompt = build_user_prompt(case, mode)
                prompt_counts.append(estimate_prompt_tokens_from_messages(build_chat_messages(prompt)))
                prompt_modes.append(mode)

    estimate = summarize_apriori_cost_estimate(
        prompt_token_counts=prompt_counts,
        report_modes=prompt_modes,
        max_completion_tokens=MAX_TOKENS,
    )
    estimate["n_outer_splits"] = len(outer_splits)
    estimate["n_completed_splits_skipped_in_estimate"] = skipped_completed_splits
    estimate["planned_report_modes"] = modes_to_run
    return estimate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-path", type=str, required=True, help="Path to the raw CSV with MRI/pathology reports and true outcomes.")
    parser.add_argument("--out_dir", type=str, required=True, help="Directory to write nested-resampling outputs.")
    parser.add_argument("--outer-scheme", type=str, default="repeated_mc", choices=["repeated_mc", "stratified_kfold"], help="Outer resampling design.")
    parser.add_argument("--outer-repeats", type=int, default=5, help="Number of repeated 80/20 Monte Carlo outer splits when --outer-scheme repeated_mc.")
    parser.add_argument("--outer-test-frac", type=float, default=0.20, help="Outer test fraction for repeated Monte Carlo splitting.")
    parser.add_argument("--outer-folds", type=int, default=5, help="Number of folds when --outer-scheme stratified_kfold.")
    parser.add_argument("--rediscovery-scheme", type=str, default="repeated_mc", choices=["repeated_mc", "stratified_kfold"], help="Training-only rediscovery design used to estimate lexicon selection frequency.")
    parser.add_argument("--rediscovery-repeats", type=int, default=25, help="Number of rediscovery Monte Carlo resamples when --rediscovery-scheme repeated_mc.")
    parser.add_argument("--rediscovery-test-frac", type=float, default=0.20, help="Test fraction for rediscovery Monte Carlo splits.")
    parser.add_argument("--rediscovery-folds", type=int, default=5, help="Number of folds when --rediscovery-scheme stratified_kfold.")
    parser.add_argument("--stability-threshold", type=float, default=0.60, help="Selection-frequency threshold used to freeze the outer-split stable lexicon.")
    parser.add_argument("--min-phrase-cases", type=int, default=2, help="Minimum number of training cases in a rediscovery subset required for a phrase to count as rediscovered.")
    parser.add_argument("--min-group-cases", type=int, default=2, help="Minimum number of training cases in a rediscovery subset required for a group to count as rediscovered.")
    parser.add_argument("--modalities", nargs="+", default=["mri", "path", "combined"], choices=["mri", "path", "combined"], help="Modalities to evaluate.")
    parser.add_argument("--representations", nargs="+", default=["group_binary", "group_count", "group_status", "phrase_binary"], choices=["group_binary", "group_count", "group_status", "phrase_binary", "weighted_concept_score", "weighted_plus_group_status"], help="Feature representations to evaluate.")
    parser.add_argument("--rate-limit-sleep-s", type=float, default=0.25, help="Sleep interval between SecureGPT extraction calls.")
    parser.add_argument("--parallel-fold-workers", type=int, default=1, help="Maximum number of outer CV folds to run concurrently. Use 5 to run 5-fold CV folds simultaneously when resources/API limits allow.")
    parser.add_argument("--max-api-workers", type=int, default=2, help="Global cap on simultaneous Stanford AI Sandbox HTTP requests across all folds/modalities/cases.")
    parser.add_argument("--parallel-modality-workers", type=int, default=2, help="Maximum number of modalities to process concurrently within each outer split.")
    parser.add_argument("--ml-n-jobs", type=int, default=2, help="Number of local CPU workers for GridSearchCV in the classical ML stage after fold/modality coordination.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED, help="Base random seed.")
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        default=True,
        help="Disable resume behavior. By default, case checkpoints and completed split checkpoints are reused.",
    )
    parser.add_argument(
        "--force-reextract",
        action="store_true",
        default=False,
        help="Ignore per-case extraction checkpoints and call SecureGPT again. Completed split checkpoints are still skipped unless --no-skip-completed-splits is also used.",
    )
    parser.add_argument(
        "--no-skip-completed-splits",
        dest="skip_completed_splits",
        action="store_false",
        default=True,
        help="When resuming, recompute outer splits even if their completed split checkpoint exists.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        default=False,
        help="Accept the printed a-priori LLM cost estimate and skip the interactive confirmation prompt.",
    )
    parser.add_argument(
        "--regenerate-reports-only",
        action="store_true",
        default=False,
        help="Rebuild plots and HTML reports from saved nested_outer_* artifacts without rerunning extraction or model fitting.",
    )
    parser.add_argument("--bootstrap-n", type=int, default=DEFAULT_BOOTSTRAP_N, help="Number of case-level bootstrap resamples for final aggregate metric confidence intervals. Use 0 to disable.")
    parser.add_argument("--relapse-permutation-n", type=int, default=1000, help="Number of label permutations for relapse AUROC/AUPRC empirical p-values. Use 0 to disable.")

    _add_bool_arg(parser, "enable-pathology-calibration", True, "Enable leakage-aware pathology-informed MRI weighting.")
    _add_bool_arg(parser, "enable-teacher-student", True, "Enable MRI-only teacher-student multi-task ridge model.")
    parser.add_argument("--ontology-groups-mode", type=str, default="stable_plus_ontology", choices=["stable_only", "stable_plus_ontology"], help="Whether recoding should include all ontology groups in addition to stable groups.")
    parser.add_argument("--weighted-lexicon-min-selection-frequency", type=float, default=0.0, help="Minimum rediscovery frequency floor for weighted MRI concept lexicon.")
    parser.add_argument("--weighted-uncertain-value", type=float, default=0.5, help="Numeric value assigned to uncertain concept evidence in weighted scores.")
    parser.add_argument("--weighted-negated-value", type=float, default=0.0, help="Numeric value assigned to negated concept evidence in weighted scores.")
    parser.add_argument("--calibration-smoothing", type=float, default=0.5, help="Additive smoothing for cross-modal reliability estimates.")
    parser.add_argument("--weight-reliability-power", type=float, default=1.0, help="Exponent for pathology concordance in MRI concept weight formula.")
    parser.add_argument("--weight-stability-power", type=float, default=0.5, help="Exponent for rediscovery stability in MRI concept weight formula.")
    parser.add_argument("--weight-association-power", type=float, default=0.5, help="Exponent for univariate dispersion association in MRI concept weight formula.")
    _add_bool_arg(parser, "run-calibration-ablations", False, "Run randomized-pathology and mismatched-pairing calibration ablations.")
    parser.add_argument("--teacher-student-alpha", type=float, default=10.0, help="Fixed ridge alpha for teacher-student model.")
    parser.add_argument("--teacher-student-lambda-dispersion", type=float, default=1.0, help="Loss weight for true continuous dispersion target.")
    parser.add_argument("--teacher-student-lambda-teacher-score", type=float, default=0.5, help="Loss weight for pathology-teacher score target.")
    parser.add_argument("--teacher-student-lambda-path-concepts", type=float, default=0.25, help="Loss weight for pathology concept targets.")

    args = parser.parse_args()

    args.max_api_workers = resolve_default_api_workers(args.max_api_workers)
    args.parallel_modality_workers = resolve_default_parallel_modality_workers(args.parallel_modality_workers)
    args.ml_n_jobs = resolve_default_ml_n_jobs(args.ml_n_jobs)
    args = coordinate_parallelism(args)

    os.makedirs(args.out_dir, exist_ok=True)
    if args.regenerate_reports_only:
        regenerate_reports_and_plots(args.out_dir, args)
        print("[DONE] Report and plot regeneration complete.")
        return

    log_dir = os.path.join(args.out_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "run_log_feature_discovery_nested_eval.txt")

    log_mode = "a" if args.resume and os.path.exists(log_path) else "w"
    with open(log_path, log_mode, encoding="utf-8") as log_f:
        tee_out = Tee(sys.__stdout__, log_f)
        tee_err = Tee(sys.__stderr__, log_f)

        with contextlib.redirect_stdout(tee_out), contextlib.redirect_stderr(tee_err):
            print("\n" + "#" * 100)
            print(f"[RUN_START] {datetime.now().isoformat(timespec='seconds')}")
            print(f"[RUN_START] resume={args.resume} force_reextract={args.force_reextract} skip_completed_splits={args.skip_completed_splits}")
            print("#" * 100)

            print(f"[LOG] Logging stdout/stderr to: {log_path}")
            print(
                f"[PARALLEL] parallel_fold_workers={args.parallel_fold_workers} "
                f"parallel_modality_workers={args.parallel_modality_workers} "
                f"max_api_workers_global={args.max_api_workers} ml_n_jobs={args.ml_n_jobs}"
            )
            print(f"[LOAD] Reading raw CSV: {args.csv_path}")
            raw_df = load_cases(args.csv_path)
            raw_df = ensure_case_id(raw_df)
            target_df = get_target_frame(raw_df)
            print(f"[LOAD] Loaded target-eligible cases={len(target_df)}")

            outer_splits = build_outer_splits(
                y_binary=target_df["dispersion_true_high_low"].astype(int).values,
                scheme=args.outer_scheme,
                random_seed=args.random_seed,
                n_repeats=args.outer_repeats,
                test_frac=args.outer_test_frac,
                n_folds=args.outer_folds,
            )

            estimate = estimate_nested_pipeline_llm_cost(raw_df, target_df, outer_splits, args)
            print_apriori_cost_estimate_report(estimate, label="nested outer-training extraction pipeline")
            print(f"[A-PRIORI] n_outer_splits={estimate['n_outer_splits']} completed_splits_skipped={estimate['n_completed_splits_skipped_in_estimate']} planned_report_modes={estimate['planned_report_modes']}")
            confirm_cost_estimate_or_exit(estimate, assume_yes=args.yes)

            preflight_check()

            ontology_csv = os.path.join(args.out_dir, "shared_biological_concept_ontology.csv")
            ontology_table().to_csv(ontology_csv, index=False)
            print(f"[SAVE] Wrote ontology table: {ontology_csv}")

            methodology_md = write_methodology_markdown(args.out_dir, args)
            print(f"[SAVE] Wrote methodology summary: {methodology_md}")

            print(f"[OUTER] Generated n_outer_splits={len(outer_splits)}")

            all_phrase_freq_tables: List[pd.DataFrame] = []
            all_group_freq_tables: List[pd.DataFrame] = []
            all_stable_phrase_tables: List[pd.DataFrame] = []
            all_stable_group_tables: List[pd.DataFrame] = []
            all_mri_audit_tables: List[pd.DataFrame] = []
            all_mri_audit_summary_tables: List[pd.DataFrame] = []
            all_reliability_tables: List[pd.DataFrame] = []
            all_weighted_lexicon_tables: List[pd.DataFrame] = []

            prediction_tables: List[pd.DataFrame] = []
            fold_result_tables: List[pd.DataFrame] = []
            hyper_tables: List[pd.DataFrame] = []
            coef_tables: List[pd.DataFrame] = []
            error_tables: List[pd.DataFrame] = []

            model_specs = get_model_specs()
            standard_representations = [r for r in args.representations if not r.startswith("weighted")]
            weighted_representations = [r for r in args.representations if r.startswith("weighted")]
            if args.enable_pathology_calibration and not weighted_representations:
                weighted_representations = ["weighted_concept_score"]

            aggregate_lists = {
                "all_phrase_freq_tables": all_phrase_freq_tables,
                "all_group_freq_tables": all_group_freq_tables,
                "all_stable_phrase_tables": all_stable_phrase_tables,
                "all_stable_group_tables": all_stable_group_tables,
                "all_mri_audit_tables": all_mri_audit_tables,
                "all_mri_audit_summary_tables": all_mri_audit_summary_tables,
                "all_reliability_tables": all_reliability_tables,
                "all_weighted_lexicon_tables": all_weighted_lexicon_tables,
                "prediction_tables": prediction_tables,
                "fold_result_tables": fold_result_tables,
                "hyper_tables": hyper_tables,
                "coef_tables": coef_tables,
                "error_tables": error_tables,
            }

            split_jobs = [
                (split_num, train_pos, test_pos)
                for split_num, (train_pos, test_pos) in enumerate(outer_splits, 1)
            ]

            if args.parallel_fold_workers > 1 and len(split_jobs) > 1:
                fold_workers = min(args.parallel_fold_workers, len(split_jobs))
                print(
                    f"[OUTER_PARALLEL] Running outer folds concurrently with workers={fold_workers}. "
                    f"Each fold writes only to its own outer_splits/<split_id>/ directory."
                )
                with ThreadPoolExecutor(max_workers=fold_workers) as executor:
                    future_to_split = {
                        executor.submit(
                            run_one_outer_split,
                            split_num=split_num,
                            train_pos=train_pos,
                            test_pos=test_pos,
                            raw_df=raw_df,
                            target_df=target_df,
                            model_specs=model_specs,
                            standard_representations=standard_representations,
                            weighted_representations=weighted_representations,
                            args=args,
                        ): f"outer_split_{split_num:03d}"
                        for split_num, train_pos, test_pos in split_jobs
                    }
                    for future in as_completed(future_to_split):
                        split_id = future_to_split[future]
                        try:
                            split_result = future.result()
                        except Exception as e:
                            split_dir = os.path.join(args.out_dir, "outer_splits", split_id)
                            split_result = _empty_split_result(split_id)
                            split_result["error_tables"].append(write_failed_split_marker(split_dir, split_id, e))
                        _extend_aggregate_tables(split_result, aggregate_lists)
                        print(f"[OUTER_PARALLEL] Collected results for {split_id}.")
            else:
                print("[OUTER_PARALLEL] Fold-level parallelism disabled; running outer folds sequentially.")
                for split_num, train_pos, test_pos in split_jobs:
                    split_result = run_one_outer_split(
                        split_num=split_num,
                        train_pos=train_pos,
                        test_pos=test_pos,
                        raw_df=raw_df,
                        target_df=target_df,
                        model_specs=model_specs,
                        standard_representations=standard_representations,
                        weighted_representations=weighted_representations,
                        args=args,
                    )
                    _extend_aggregate_tables(split_result, aggregate_lists)

            # Save aggregate lexicon/calibration/audit summaries.
            if all_phrase_freq_tables:
                phrase_freq_all = pd.concat(all_phrase_freq_tables, ignore_index=True)
                phrase_freq_all.to_csv(os.path.join(args.out_dir, "all_outer_phrase_rediscovery_frequencies.csv"), index=False)
            else:
                phrase_freq_all = pd.DataFrame()

            if all_group_freq_tables:
                group_freq_all = pd.concat(all_group_freq_tables, ignore_index=True)
                group_freq_all.to_csv(os.path.join(args.out_dir, "all_outer_group_rediscovery_frequencies.csv"), index=False)
            else:
                group_freq_all = pd.DataFrame()

            if all_stable_phrase_tables:
                stable_phrase_all = pd.concat(all_stable_phrase_tables, ignore_index=True)
                stable_phrase_summary = stable_phrase_all.groupby(["report_mode", "phrase_slug", "quote_norm", "canonical_group"]).size().reset_index(name="n_outer_splits_stable").sort_values(["report_mode", "n_outer_splits_stable"], ascending=[True, False])
                stable_phrase_summary.to_csv(os.path.join(args.out_dir, "stable_phrase_lexicon_outer_summary.csv"), index=False)
            else:
                stable_phrase_summary = pd.DataFrame()

            if all_stable_group_tables:
                stable_group_all = pd.concat(all_stable_group_tables, ignore_index=True)
                stable_group_summary = stable_group_all.groupby(["report_mode", "canonical_group"]).size().reset_index(name="n_outer_splits_stable").sort_values(["report_mode", "n_outer_splits_stable"], ascending=[True, False])
                stable_group_summary.to_csv(os.path.join(args.out_dir, "stable_group_lexicon_outer_summary.csv"), index=False)
            else:
                stable_group_summary = pd.DataFrame()

            if all_mri_audit_tables:
                pd.concat(all_mri_audit_tables, ignore_index=True).to_csv(os.path.join(args.out_dir, "mri_audit_case_table_all_outer_splits.csv"), index=False)
            if all_mri_audit_summary_tables:
                pd.concat(all_mri_audit_summary_tables, ignore_index=True).to_csv(os.path.join(args.out_dir, "mri_audit_density_summary_all_outer_splits.csv"), index=False)
            if all_reliability_tables:
                reliability_all = pd.concat(all_reliability_tables, ignore_index=True)
                reliability_all.to_csv(os.path.join(args.out_dir, "all_outer_mri_pathology_reliability_matrices.csv"), index=False)
            else:
                reliability_all = pd.DataFrame()
            if all_weighted_lexicon_tables:
                weighted_lexicon_all = pd.concat(all_weighted_lexicon_tables, ignore_index=True)
                weighted_lexicon_all.to_csv(os.path.join(args.out_dir, "all_outer_weighted_mri_lexicons.csv"), index=False)
            else:
                weighted_lexicon_all = pd.DataFrame()

            pred_all = pd.concat(prediction_tables, ignore_index=True) if prediction_tables else pd.DataFrame()
            if len(pred_all) and "target_name" not in pred_all.columns:
                pred_all["target_name"] = np.where(
                    pred_all["task_type"].astype(str) == "regression",
                    TARGET_NAME_DISPERSION_SCORE,
                    TARGET_NAME_DISPERSION_HIGH_LOW,
                )
                pred_all["target_col"] = np.where(
                    pred_all["task_type"].astype(str) == "regression",
                    "dispersion_true",
                    "dispersion_true_high_low",
                )
            pred_case_all = deduplicate_outer_predictions(pred_all)
            fold_results_all = pd.concat(fold_result_tables, ignore_index=True) if fold_result_tables else pd.DataFrame()
            hyper_all = pd.concat(hyper_tables, ignore_index=True) if hyper_tables else pd.DataFrame()
            coef_all = pd.concat(coef_tables, ignore_index=True) if coef_tables else pd.DataFrame()
            error_all = pd.concat(error_tables, ignore_index=True) if error_tables else pd.DataFrame()

            pred_out = os.path.join(args.out_dir, "nested_outer_predictions_all.csv")
            pred_case_out = os.path.join(args.out_dir, "nested_outer_predictions_case_deduplicated.csv")
            fold_out = os.path.join(args.out_dir, "nested_outer_fold_metrics_all.csv")
            hyper_out = os.path.join(args.out_dir, "nested_outer_hyperparameters_all.csv")
            coef_out = os.path.join(args.out_dir, "nested_outer_feature_coefficients_all.csv")
            error_out = os.path.join(args.out_dir, "nested_outer_split_errors.csv")

            pred_all.to_csv(pred_out, index=False)
            pred_case_all.to_csv(pred_case_out, index=False)
            fold_results_all.to_csv(fold_out, index=False)
            hyper_all.to_csv(hyper_out, index=False)
            coef_all.to_csv(coef_out, index=False)
            error_all.to_csv(error_out, index=False)

            print(f"[SAVE] Wrote raw per-split predictions: {pred_out}")
            print(f"[SAVE] Wrote case-deduplicated predictions: {pred_case_out}")
            print(f"[SAVE] Wrote fold-level metrics: {fold_out}")
            print(f"[SAVE] Wrote hyperparameter summary: {hyper_out}")
            print(f"[SAVE] Wrote coefficient summary: {coef_out}")
            if len(error_all):
                print(f"[WARN] One or more outer folds failed. Error summary: {error_out}")

            metrics_df = compute_metrics_from_predictions(pred_case_all, args)
            metrics_csv = os.path.join(args.out_dir, "nested_outer_metrics_summary.csv")
            metrics_df.to_csv(metrics_csv, index=False)
            print(f"[SAVE] Wrote case-deduplicated metrics summary: {metrics_csv}")

            feature_rank_cls_df = rank_features_across_models([coef_all[coef_all["task_type"] == "classification"].assign(task_type="classification")] if len(coef_all) else [], task_type="classification")
            feature_rank_reg_df = rank_features_across_models([coef_all[coef_all["task_type"] == "regression"].assign(task_type="regression")] if len(coef_all) else [], task_type="regression")

            feature_rank_cls_csv = os.path.join(args.out_dir, "nested_feature_ranking_classification.csv")
            feature_rank_reg_csv = os.path.join(args.out_dir, "nested_feature_ranking_regression.csv")
            feature_rank_cls_df.to_csv(feature_rank_cls_csv, index=False)
            feature_rank_reg_df.to_csv(feature_rank_reg_csv, index=False)
            print(f"[SAVE] Wrote classification feature ranking: {feature_rank_cls_csv}")
            print(f"[SAVE] Wrote regression feature ranking: {feature_rank_reg_csv}")

            sign_stability_df = summarize_coefficient_sign_stability(coef_all)
            sign_stability_csv = os.path.join(args.out_dir, "nested_feature_sign_stability.csv")
            sign_stability_df.to_csv(sign_stability_csv, index=False)
            print(f"[SAVE] Wrote coefficient sign stability: {sign_stability_csv}")

            relapse_balance_df = relapse_split_diagnostics(target_df, outer_splits, args.out_dir)
            permutation_df = permutation_test_relapse_metrics(pred_case_all, metrics_df, args, args.out_dir)
            path_mri_subset_metrics_df = pathology_metrics_on_mri_complete(pred_case_all, raw_df, args)
            if len(path_mri_subset_metrics_df):
                path_mri_subset_metrics_df.to_csv(os.path.join(args.out_dir, "pathology_only_mri_complete_subset_metrics.csv"), index=False)

            error_case_df, error_md = generate_error_analysis(pred_case_all, metrics_df, raw_df, args.out_dir)
            if error_md:
                print(f"[SAVE] Wrote missed-case error analysis: {error_md}")

            interp_md, interp_html = generate_interpretability_report(
                out_dir=args.out_dir,
                coef_all=coef_all,
                sign_stability_df=sign_stability_df,
                phrase_freq_all=phrase_freq_all,
                group_freq_all=group_freq_all,
                stable_phrase_summary=stable_phrase_summary,
                stable_group_summary=stable_group_summary,
                reliability_all=reliability_all,
                weighted_lexicon_all=weighted_lexicon_all,
            )
            print(f"[SAVE] Wrote interpretability reports: {interp_md}, {interp_html}")

            cls_plot_png = os.path.join(args.out_dir, "nested_classification_comparison.png")
            relapse_auroc_plot_png = os.path.join(args.out_dir, "nested_relapse_auroc_comparison.png")
            relapse_auprc_plot_png = os.path.join(args.out_dir, "nested_relapse_auprc_comparison.png")
            relapse_f1_plot_png = os.path.join(args.out_dir, "nested_relapse_f1_comparison.png")
            relapse_brier_plot_png = os.path.join(args.out_dir, "nested_relapse_brier_comparison.png")
            reg_err_plot_png = os.path.join(args.out_dir, "nested_regression_error_comparison.png")
            reg_corr_plot_png = os.path.join(args.out_dir, "nested_regression_correlation_comparison.png")
            plot_classification_comparison(metrics_df, cls_plot_png)
            plot_relapse_metric_comparison(metrics_df, relapse_auroc_plot_png, "auroc")
            plot_relapse_metric_comparison(metrics_df, relapse_auprc_plot_png, "auprc")
            plot_relapse_metric_comparison(metrics_df, relapse_f1_plot_png, "f1")
            plot_relapse_metric_comparison(metrics_df, relapse_brier_plot_png, "brier")
            plot_relapse_curves(pred_case_all, metrics_df, args.out_dir)
            plot_regression_error_comparison(metrics_df, reg_err_plot_png)
            plot_regression_correlation_comparison(metrics_df, reg_corr_plot_png)
            report_plot_paths = generate_performance_plots(pred_case_all, metrics_df, args.out_dir)
            for pth in [cls_plot_png, relapse_auroc_plot_png, relapse_auprc_plot_png, relapse_f1_plot_png, relapse_brier_plot_png, reg_err_plot_png, reg_corr_plot_png]:
                if os.path.exists(pth):
                    report_plot_paths.append(pth)
            results_md, results_html = generate_results_report(
                out_dir=args.out_dir,
                metrics_df=metrics_df,
                pred_case_df=pred_case_all,
                fold_results_all=fold_results_all,
                relapse_balance_df=relapse_balance_df,
                permutation_df=permutation_df,
                plot_paths=report_plot_paths,
                path_mri_subset_metrics_df=path_mri_subset_metrics_df,
            )
            print(f"[SAVE] Wrote automated results reports: {results_md}, {results_html}")

            summary_txt = os.path.join(args.out_dir, "nested_resampling_summary.txt")
            summary = summarize_metrics(metrics_df)
            with open(summary_txt, "w", encoding="utf-8") as f:
                f.write(summary)

            print(f"[SAVE] Wrote summary text: {summary_txt}")
            print("\n" + summary)
            print_cumulative_report()
            write_cost_tracker_json(args.out_dir)
            print("[DONE] Nested resampling evaluation complete.")


if __name__ == "__main__":
    main()
