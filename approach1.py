#!/usr/bin/env python3
"""
End-to-end SecureGPT few-shot pipeline + evaluation for three report-input tiers:

  1) MRI only
  2) Pathology only
  3) MRI + pathology combined

For each exemplar/shot set, this script:
  - builds 2 high-dispersion + 2 low-dispersion few-shot prompts
  - runs held-out predictions for each modality tier
  - skips cases missing preop MRI for MRI-only and MRI+pathology tiers
  - saves JSONL/CSV predictions
  - immediately evaluates predictions and writes metrics + plots

Default exemplar sets:
  A) high rows [0, 2],   low rows [101, 102]
  B) high rows [0, 19],  low rows [82, 85]

Usage:
  python approach1-3.py \
    --csv-path /Users/lukezhao/projects/onc/PROCESSED_TRIMMED_path_cases_MRI_status.csv \
    --outdir /Users/lukezhao/projects/onc/sabcs/securegpt_dispersion_approach1_new

Notes:
  - Row indices are 0-based pandas iloc indices.
  - Every case is expected to have pathology text.
  - MRI-only and MRI+pathology tiers exclude held-out rows with missing/blank MRI text.
  - Resume support: per-case JSONL checkpoints and per-config completion markers.

Resume behavior (enabled by default):
  - Re-run the same --outdir to continue interrupted work.
  - Completed shotset/modality folders are skipped when their checkpoint matches.
  - Incomplete folders resume from valid lines in predictions_testing_cases.jsonl.
  - Use --no-resume to start fresh, --force-rerun-cases to redo API calls in-place.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from scipy.stats import spearmanr
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)


# -----------------------------
# Config defaults
# -----------------------------

CSV_PATH = "/Users/lukezhao/projects/onc/PROCESSED_TRIMMED_path_cases_MRI_status.csv"
OUT_DIR = os.path.join(os.getcwd(), "securegpt_dispersion_3tier_outputs")
ENV_PATH = os.getenv("ENV_PATH", "/Users/lukezhao/projects/onc/.env")

API_VERSION = "2024-12-01-preview"
DEPLOYMENT = "gpt-5-nano"
SECUREGPT_BASE_URL = "https://aihubapi.stanfordhealthcare.org/azure-openai"

MAX_TOKENS = 16000
REQUEST_TIMEOUT_S = 120
MAX_RETRIES = 3
BACKOFF_BASE_S = 1.5
RATE_LIMIT_SLEEP_S = 0.25

DISPERSION_HIGH_THRESHOLD = 85.0
RESUME_CHECKPOINT_SUBDIR = "_resume_checkpoint"
RESUME_SCRIPT_VERSION = "approach1-3-v1"

SHOT_SETS = [
    {
        "name": "shotset_high_0_2_low_101_102",
        "high_rows": [0, 2],
        "low_rows": [101, 102],
    },
    {
        "name": "shotset_high_0_19_low_82_85",
        "high_rows": [0, 19],
        "low_rows": [82, 85],
    },
]

MODALITY_TIERS = ["mri_only", "pathology_only", "mri_plus_pathology"]

# Units: USD per 1,000,000 tokens. Adjust if your local pricing changes.
PRICE_PER_1M_INPUT_TOKENS = 0.05
PRICE_PER_1M_CACHED_INPUT_TOKENS = 0.01
PRICE_PER_1M_OUTPUT_TOKENS = 0.40

API_KEY: Optional[str] = None
URL: Optional[str] = None
HEADERS: Dict[str, str] = {}

REASONING_EFFORT = os.getenv("REASONING_EFFORT", "minimal").strip().lower()
if REASONING_EFFORT in {"", "none", "null"}:
    REASONING_EFFORT = ""


def _empty_cost_tracker() -> Dict[str, Any]:
    return {
        "calls": 0,
        "prompt_tokens": 0,
        "cached_tokens": 0,
        "uncached_prompt_tokens": 0,
        "completion_tokens": 0,
        "reasoning_tokens": 0,
        "total_tokens": 0,
        "estimated_cost_usd": 0.0,
        "estimated_cache_savings_usd": 0.0,
    }


COST_TRACKER = _empty_cost_tracker()


# -----------------------------
# Logging tee
# -----------------------------

class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data: str) -> None:
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self) -> None:
        for s in self.streams:
            s.flush()


# -----------------------------
# Data structures
# -----------------------------

@dataclass
class Case:
    case_id: str
    preop_mri: str
    path_report: str
    index_side: Optional[str] = None
    dispersion_true: Optional[float] = None
    relapse_true: Optional[int] = None


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


# -----------------------------
# General helpers
# -----------------------------

def _safe_text(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and np.isnan(x):
        return ""
    return str(x)


def _has_report_text(x: Any) -> bool:
    s = _safe_text(x).strip()
    if not s:
        return False
    return s.lower() not in {"nan", "none", "null", "na", "n/a"}


def _shorten_for_prompt(text: str) -> str:
    # Preserved from the current pipeline. Add trimming here only if needed.
    return text


def _word_count(s: str) -> int:
    return len([w for w in re.split(r"\s+", s.strip()) if w])


def _normalize_side(side: Optional[str]) -> str:
    s = (side or "").strip().lower()
    return s if s in {"left", "right"} else "unknown"


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(math.sqrt(mean_squared_error(y_true, y_pred)))


def _case_id_for_token(case_id: str) -> str:
    # Keep the token compact and robust to spaces/symbols in case IDs.
    safe = re.sub(r"[^A-Za-z0-9_\-]+", "_", str(case_id)).strip("_")
    return safe or "case"


def _make_case_token(test_case: Case, row_index: int, modality: str) -> str:
    side = _normalize_side(test_case.index_side)
    base = f"{test_case.case_id}|row_{row_index}|side_{side}|modality_{modality}"
    digest = hashlib.sha256(base.encode("utf-8")).hexdigest()[:10].upper()
    cid = _case_id_for_token(test_case.case_id)
    return f"CTXCHK_{digest}_CASE_{cid}_ROW_{row_index}_SIDE_{side}_MODALITY_{modality}"


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


# -----------------------------
# SecureGPT API setup + cost tracking
# -----------------------------

def configure_api(env_path: str, deployment: str, api_version: str) -> None:
    global API_KEY, URL, HEADERS, DEPLOYMENT, API_VERSION
    DEPLOYMENT = deployment
    API_VERSION = api_version

    load_dotenv(env_path, override=True)
    API_KEY = os.getenv("SANDBOX_API_KEY")
    if not API_KEY:
        raise RuntimeError(f"SANDBOX_API_KEY not found in {env_path}. Check your .env file.")
    API_KEY = API_KEY.strip()

    URL = (
        SECUREGPT_BASE_URL
        + f"/deployments/{DEPLOYMENT}/chat/completions"
        + f"?api-version={API_VERSION}"
    )
    HEADERS = {
        "api-key": API_KEY,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def reset_cost_tracker() -> None:
    global COST_TRACKER
    COST_TRACKER = _empty_cost_tracker()


def estimate_cost_from_usage(usage: Dict[str, Any]) -> Dict[str, Any]:
    prompt_tokens = usage.get("prompt_tokens", 0) or 0
    completion_tokens = usage.get("completion_tokens", 0) or 0
    total_tokens = usage.get("total_tokens", 0) or 0

    prompt_details = usage.get("prompt_tokens_details", {}) or {}
    completion_details = usage.get("completion_tokens_details", {}) or {}

    cached_tokens = prompt_details.get("cached_tokens", 0) or 0
    reasoning_tokens = completion_details.get("reasoning_tokens", 0) or 0
    uncached_prompt_tokens = max(prompt_tokens - cached_tokens, 0)

    input_cost = (uncached_prompt_tokens / 1_000_000) * PRICE_PER_1M_INPUT_TOKENS
    cached_input_cost = (cached_tokens / 1_000_000) * PRICE_PER_1M_CACHED_INPUT_TOKENS
    output_cost = (completion_tokens / 1_000_000) * PRICE_PER_1M_OUTPUT_TOKENS
    estimated_cost = input_cost + cached_input_cost + output_cost

    no_cache_input_cost = (prompt_tokens / 1_000_000) * PRICE_PER_1M_INPUT_TOKENS
    actual_input_cost = input_cost + cached_input_cost
    cache_savings = max(no_cache_input_cost - actual_input_cost, 0.0)

    return {
        "prompt_tokens": prompt_tokens,
        "cached_tokens": cached_tokens,
        "uncached_prompt_tokens": uncached_prompt_tokens,
        "completion_tokens": completion_tokens,
        "reasoning_tokens": reasoning_tokens,
        "total_tokens": total_tokens,
        "estimated_input_cost_usd": input_cost,
        "estimated_cached_input_cost_usd": cached_input_cost,
        "estimated_output_cost_usd": output_cost,
        "estimated_cost_usd": estimated_cost,
        "estimated_cache_savings_usd": cache_savings,
    }


def update_cost_tracker(cost_info: Dict[str, Any]) -> None:
    COST_TRACKER["calls"] += 1
    COST_TRACKER["prompt_tokens"] += int(cost_info["prompt_tokens"])
    COST_TRACKER["cached_tokens"] += int(cost_info["cached_tokens"])
    COST_TRACKER["uncached_prompt_tokens"] += int(cost_info["uncached_prompt_tokens"])
    COST_TRACKER["completion_tokens"] += int(cost_info["completion_tokens"])
    COST_TRACKER["reasoning_tokens"] += int(cost_info["reasoning_tokens"])
    COST_TRACKER["total_tokens"] += int(cost_info["total_tokens"])
    COST_TRACKER["estimated_cost_usd"] += float(cost_info["estimated_cost_usd"])
    COST_TRACKER["estimated_cache_savings_usd"] += float(cost_info["estimated_cache_savings_usd"])


def print_cumulative_report() -> None:
    print("\n[CUMULATIVE TOKEN / COST REPORT]")
    print(f"calls:                    {COST_TRACKER['calls']}")
    print(f"prompt_tokens:            {COST_TRACKER['prompt_tokens']}")
    print(f"cached_tokens:            {COST_TRACKER['cached_tokens']}")
    print(f"uncached_prompt_tokens:   {COST_TRACKER['uncached_prompt_tokens']}")
    print(f"completion_tokens:        {COST_TRACKER['completion_tokens']}")
    print(f"reasoning_tokens:         {COST_TRACKER['reasoning_tokens']}")
    print(f"total_tokens:             {COST_TRACKER['total_tokens']}")
    print(f"estimated_total_cost_usd: ${COST_TRACKER['estimated_cost_usd']:.8f}")
    print(f"estimated_cache_savings:  ${COST_TRACKER['estimated_cache_savings_usd']:.8f}")


def _atomic_write_json(path: str, payload: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def _atomic_write_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def merge_cost_trackers(prior: Optional[Dict[str, Any]], session: Dict[str, Any]) -> Dict[str, Any]:
    if not prior:
        return dict(session)
    merged = _empty_cost_tracker()
    int_keys = {
        "calls",
        "prompt_tokens",
        "cached_tokens",
        "uncached_prompt_tokens",
        "completion_tokens",
        "reasoning_tokens",
        "total_tokens",
    }
    float_keys = {"estimated_cost_usd", "estimated_cache_savings_usd"}
    for k in int_keys:
        merged[k] = int(prior.get(k, 0)) + int(session.get(k, 0))
    for k in float_keys:
        merged[k] = float(prior.get(k, 0.0)) + float(session.get(k, 0.0))
    return merged


def load_cost_tracker_snapshot(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None
    if isinstance(data, dict) and isinstance(data.get("cumulative"), dict):
        return data["cumulative"]
    if isinstance(data, dict):
        return data
    return None


def save_cumulative_report_json(path: str, prior: Optional[Dict[str, Any]] = None) -> None:
    session = dict(COST_TRACKER)
    cumulative = merge_cost_trackers(prior, session)
    payload = {
        "resume_script_version": RESUME_SCRIPT_VERSION,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "cumulative": cumulative,
        "session": session,
        "prior_sessions": prior or _empty_cost_tracker(),
    }
    _atomic_write_json(path, payload)


# -----------------------------
# Data loading
# -----------------------------

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
    dispersion = row["dispersion_invasive_DCIS_geographic"]
    relapse = row["relapse"]
    return Case(
        case_id=str(row["case_id"]),
        preop_mri=_safe_text(row["preop_MRI_text"]),
        path_report=_safe_text(row["path_report_text"]),
        index_side=_safe_text(row["index_side"]),
        dispersion_true=float(dispersion) if pd.notna(dispersion) else None,
        relapse_true=int(relapse) if pd.notna(relapse) else None,
    )


# -----------------------------
# Prompt construction
# -----------------------------

SYSTEM_MSG = (
    "You are an advanced, careful clinical NLP model operating in a PHI-secure environment. "
    "Follow instructions exactly. Output must be valid JSON only—no extra text. "
    "Use ONLY the provided report text. If information is not present, do not invent. "
    "Do not reveal hidden chain-of-thought. Instead provide a concise, auditable reasoning summary "
    "and structured rationale fields grounded in the report."
)

DESCRIPTORS_TEXT = """
DISPERSIVENESS DESCRIPTORS (GUIDANCE; NOT EXHAUSTIVE)

Often higher dispersiveness:
- "scattered small clusters", "single cells", "cords" / infiltrative pattern
- residual carcinoma described as patchy, discontinuous, multifocal
- margin positive/close/approaches margin
- large "span/extent of residual carcinoma" relative to "largest contiguous focus"
- lymphovascular invasion (LVI) present/extensive
- extensive DCIS; "DCIS spans X cm"; extensive intraductal component; comedo-type necrosis
- skin/dermis/nipple involvement; skeletal muscle/chest wall involvement
- extranodal extension (if lymph nodes discussed)
- MRI: non-mass enhancement, clumped/segmental/linear distribution; scattered foci/satellites; multicentricity

Often lower dispersiveness:
- single dominant contiguous residual focus; well-circumscribed residual mass
- minimal/near-complete response with little/no residual invasive carcinoma described
- clear margins with comfortable distances; no LVI; limited/non-extensive DCIS
- MRI: single shrinking mass without scattered foci; localized enhancement
""".strip()


def _report_fields_for_prompt(c: Case, modality: str) -> str:
    parts: List[str] = []
    if modality in {"mri_only", "mri_plus_pathology"}:
        parts.append(f"preop_MRI_text:\n{_shorten_for_prompt(c.preop_mri)}")
    else:
        parts.append("preop_MRI_text:\n<NOT PROVIDED IN THIS TIER>")

    if modality in {"pathology_only", "mri_plus_pathology"}:
        parts.append(f"path_report_text:\n{_shorten_for_prompt(c.path_report)}")
    else:
        parts.append("path_report_text:\n<NOT PROVIDED IN THIS TIER>")
    return "\n\n".join(parts)


def build_training_block(train_cases: List[Case], modality: str) -> str:
    blocks: List[str] = []
    modality_name = modality_display_name(modality)
    for i, c in enumerate(train_cases, 1):
        blocks.append(
            f"EXAMPLE {i} (LABELED; {modality_name})\n"
            f"case_id: {c.case_id}\n"
            f"index_side: {c.index_side}\n"
            f"dispersion_score_true: {c.dispersion_true}\n"
            f"relapse_true: {c.relapse_true}\n"
            f"{_report_fields_for_prompt(c, modality)}\n"
        )
    return "\n\n".join(blocks)


def build_user_prompt(training_block: str, test_case: Case, row_index: int, modality: str) -> str:
    validation_token = _make_case_token(test_case, row_index, modality)
    modality_name = modality_display_name(modality)

    if modality == "mri_only":
        tier_instructions = (
            "This tier uses ONLY the preop MRI report. Pathology text is intentionally not provided. "
            "Base predictions only on MRI language and explicitly state that pathology was not supplied in the pathology rationale field."
        )
    elif modality == "pathology_only":
        tier_instructions = (
            "This tier uses ONLY the pathology report. MRI text is intentionally not provided. "
            "Base predictions only on pathology language and explicitly state that MRI was not supplied in the MRI rationale field."
        )
    elif modality == "mri_plus_pathology":
        tier_instructions = (
            "This tier uses BOTH the preop MRI report and the pathology report. Integrate both modalities, while still grounding every claim in the supplied text."
        )
    else:
        raise ValueError(f"Unknown modality: {modality}")

    return f"""
TASK
Given breast cancer clinical report text after neoadjuvant chemotherapy and before surgery.
Current prediction tier: {modality_name}.
{tier_instructions}

Report fields:
- preop_MRI_text: may be intentionally absent depending on tier
- path_report_text: may be intentionally absent depending on tier
- index_side: which breast ("left" or "right") the dispersion score and relapse label refer to. Some reports may mention both breasts; ALWAYS focus on the breast side indicated by index_side.

Predict ONLY:
1) dispersion_score_pred: float in [0, 450]
2) dispersion_high_low_pred: 0 or 1, where 1 = high dispersion and 0 = low dispersion, using the rule:
   - high dispersion (1) if dispersion_score_pred >= 85
   - low dispersion  (0) if dispersion_score_pred < 85
3) relapse_pred: 0 or 1 (1=relapsing, 0=non-relapsing)
4) key_evidence: up to 6 short quotes (<=25 words each) copied VERBATIM from the provided reports that most support your predictions
5) retrieval_check_token_returned: echo exactly the single validation token provided at the very end of the case prompt
6) retrieval_check_correct: boolean indicating whether the returned token exactly matches the validation token
7) reasoning_summary: one paragraph summarizing the evidence-based rationale grounded only in the reports provided for this tier
8) structured_rationale: a short stepwise evidence-grounded explanation with the required keys below

STRICT RULES
- Use ONLY the report text provided in this tier for clinical prediction. Do not assume missing information from omitted modalities.
- The validation token is NON-CLINICAL metadata. Echo it exactly; do not use it as clinical evidence.
- Do not invent measurements or findings.
- dispersion_high_low_pred MUST be consistent with dispersion_score_pred using the cutoff 85 (>= 85 = high/1; < 85 = low/0).
- For BOTH the labeled FEW-SHOT TRAINING EXAMPLES and the NEW CASE, ALWAYS focus on the breast side indicated by index_side ("left" or "right"). If reports mention both breasts, ignore findings that clearly correspond to the opposite, non-index side.
- key_evidence quotes MUST be copied exactly from the provided report text and must each be <=25 words.
- reasoning_summary and structured_rationale must be concise, auditable, and grounded in the report; do NOT provide hidden chain-of-thought or mention internal deliberation.
- Output must be valid JSON ONLY (no extra text).

{DESCRIPTORS_TEXT}

OUTPUT JSON SCHEMA (RETURN ONLY THIS OBJECT)
{{
  "case_id": "<case_id from NEW CASE>",
  "dispersion_score_pred": <float 0-450>,
  "dispersion_high_low_pred": <0 or 1>,
  "relapse_pred": <0 or 1>,
  "key_evidence": ["<verbatim quote 1>", "... up to 6 quotes total ..."],
  "retrieval_check_token_returned": "<exact validation token at the end of the case prompt>",
  "retrieval_check_correct": <true or false>,
  "reasoning_summary": "<one paragraph, concise, evidence-grounded rationale>",
  "structured_rationale": {{
    "step_1_localization": "<brief statement about side-specific localization and modality availability>",
    "step_2_pathology_pattern": "<brief statement about pathology cues, or that pathology was not supplied in this tier>",
    "step_3_mri_pattern": "<brief statement about MRI cues, or that MRI was not supplied in this tier>",
    "step_4_dispersion_synthesis": "<brief statement connecting cues to predicted dispersion score/high-low label>",
    "step_5_relapse_synthesis": "<brief statement connecting overall residual pattern to predicted relapse label>"
  }}
}}

FEW-SHOT TRAINING EXAMPLES
{training_block}

NOW PREDICT THIS NEW CASE (UNLABELED; {modality_name})
case_id: {test_case.case_id}
index_side: {test_case.index_side}
{_report_fields_for_prompt(test_case, modality)}

VALIDATION_TOKEN_FOR_THIS_CASE_DO_NOT_USE_AS_CLINICAL_EVIDENCE: {validation_token}""".strip()


# -----------------------------
# Prediction validation + API calls
# -----------------------------

def _validate_prediction_obj(
    obj: Dict[str, Any],
    expected_case_id: str,
    expected_token: str,
) -> Tuple[bool, str]:
    required = [
        "dispersion_score_pred",
        "dispersion_high_low_pred",
        "relapse_pred",
        "key_evidence",
        "retrieval_check_token_returned",
        "retrieval_check_correct",
        "reasoning_summary",
        "structured_rationale",
    ]
    for k in required:
        if k not in obj:
            return False, f"Missing key: {k}"

    if "case_id" in obj and str(obj["case_id"]) != str(expected_case_id):
        return False, f"case_id mismatch: got {obj['case_id']} expected {expected_case_id}"

    try:
        dsp = float(obj["dispersion_score_pred"])
    except Exception:
        return False, "dispersion_score_pred is not a number"
    if not (0.0 <= dsp <= 450.0):
        return False, f"dispersion_score_pred out of range [0,450]: {dsp}"

    try:
        dhl = int(obj["dispersion_high_low_pred"])
    except Exception:
        return False, "dispersion_high_low_pred is not an integer"
    if dhl not in (0, 1):
        return False, f"dispersion_high_low_pred must be 0/1, got {dhl}"

    try:
        rp = int(obj["relapse_pred"])
    except Exception:
        return False, "relapse_pred is not an integer"
    if rp not in (0, 1):
        return False, f"relapse_pred must be 0/1, got {rp}"

    ke = obj["key_evidence"]
    if not isinstance(ke, list):
        return False, "key_evidence must be a list"
    if len(ke) > 6:
        return False, f"key_evidence must have <= 6 items, got {len(ke)}"
    for i, q in enumerate(ke):
        if not isinstance(q, str):
            return False, f"key_evidence[{i}] must be a string"
        if not q.strip():
            return False, f"key_evidence[{i}] is empty"
        if _word_count(q) > 25:
            return False, f"key_evidence[{i}] exceeds 25 words"

    returned_token = obj["retrieval_check_token_returned"]
    if not isinstance(returned_token, str) or not returned_token.strip():
        return False, "retrieval_check_token_returned must be a non-empty string"

    token_correct = obj["retrieval_check_correct"]
    if token_correct not in [True, False]:
        return False, "retrieval_check_correct must be boolean"

    if not isinstance(obj["reasoning_summary"], str) or not obj["reasoning_summary"].strip():
        return False, "reasoning_summary must be a non-empty string"

    sr = obj["structured_rationale"]
    if not isinstance(sr, dict):
        return False, "structured_rationale must be an object"
    for subk in [
        "step_1_localization",
        "step_2_pathology_pattern",
        "step_3_mri_pattern",
        "step_4_dispersion_synthesis",
        "step_5_relapse_synthesis",
    ]:
        if subk not in sr:
            return False, f"structured_rationale missing key: {subk}"
        if not isinstance(sr[subk], str) or not sr[subk].strip():
            return False, f"structured_rationale[{subk}] must be a non-empty string"

    expected_correct = returned_token == expected_token
    if bool(token_correct) != expected_correct:
        return False, (
            "retrieval_check_correct inconsistent with returned token "
            f"(expected {expected_correct}, got {token_correct})"
        )

    if dhl != int(dsp >= DISPERSION_HIGH_THRESHOLD):
        return False, (
            f"dispersion_high_low_pred inconsistent with dispersion_score_pred: "
            f"score={dsp}, label={dhl}, expected={int(dsp >= DISPERSION_HIGH_THRESHOLD)}"
        )

    return True, "ok"


def _extract_json_from_text(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        raise ValueError("No JSON object found in model output.")
    return json.loads(m.group(0))


def _response_to_json(response: requests.Response) -> Dict[str, Any]:
    try:
        return response.json()
    except Exception as exc:
        raise RuntimeError(f"API returned non-JSON response: {response.text[:1000]}") from exc


def call_securegpt_chat(prompt: str, max_completion_tokens: int = MAX_TOKENS) -> str:
    if URL is None or not HEADERS:
        raise RuntimeError("API is not configured. Call configure_api(...) first.")

    print(f"[API] Sending request... prompt_chars={len(prompt)}")
    payload = {
        "model": DEPLOYMENT,
        "messages": [
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": prompt},
        ],
        "max_completion_tokens": max_completion_tokens,
    }
    # if REASONING_EFFORT:
    #     payload["reasoning_effort"] = REASONING_EFFORT

    t0 = time.time()
    response = requests.post(
        URL,
        headers=HEADERS,
        json=payload,
        timeout=REQUEST_TIMEOUT_S,
    )
    dt = time.time() - t0

    if not response.ok:
        print("[API] Error:")
        print(f"[API] Status: {response.status_code}")
        try:
            print(json.dumps(response.json(), indent=2))
        except Exception:
            print(response.text)
        response.raise_for_status()

    data = _response_to_json(response)
    usage = data.get("usage", {}) or {}
    cost_info = estimate_cost_from_usage(usage)
    update_cost_tracker(cost_info)

    choice = data["choices"][0]
    content = choice["message"].get("content") or ""

    print(
        f"[API] Response in {dt:.2f}s. response_chars={len(content)} "
        f"model={data.get('model', DEPLOYMENT)} "
        f"prompt_tokens={cost_info['prompt_tokens']} "
        f"cached_tokens={cost_info['cached_tokens']} "
        f"completion_tokens={cost_info['completion_tokens']} "
        f"estimated_cost=${cost_info['estimated_cost_usd']:.8f} "
        f"cache_savings=${cost_info['estimated_cache_savings_usd']:.8f}"
    )

    if not content:
        finish_reason = choice.get("finish_reason")
        print(
            "\n[WARNING] Empty visible output. "
            f"finish_reason={finish_reason}. "
            "Try increasing MAX_TOKENS / max_completion_tokens."
        )

    return content


def preflight_check() -> None:
    print("[PREFLIGHT] Testing SecureGPT connectivity with a small request...")
    t0 = time.time()
    reply = call_securegpt_chat("Reply with exactly: OK", max_completion_tokens=20)
    dt = time.time() - t0
    print(f"[PREFLIGHT] Success in {dt:.2f}s. Reply={reply!r}")


def predict_case(training_block: str, test_case: Case, row_index: int, modality: str) -> Dict[str, Any]:
    user_prompt = build_user_prompt(training_block, test_case, row_index=row_index, modality=modality)
    expected_token = _make_case_token(test_case, row_index, modality)
    last_err: Optional[str] = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"[PREDICT] case_id={test_case.case_id} modality={modality} attempt={attempt}/{MAX_RETRIES}")
            raw = call_securegpt_chat(user_prompt, max_completion_tokens=MAX_TOKENS)
            obj = _extract_json_from_text(raw)
            ok, msg = _validate_prediction_obj(
                obj,
                expected_case_id=test_case.case_id,
                expected_token=expected_token,
            )
            if not ok:
                raise ValueError(f"Validation failed: {msg}. Raw head: {raw[:400]}")
            print(f"[PREDICT] case_id={test_case.case_id} modality={modality} VALID JSON received.")
            return obj
        except Exception as e:
            last_err = str(e)
            print(f"[RETRY] case_id={test_case.case_id} attempt={attempt}/{MAX_RETRIES} error={last_err}")
            sleep_s = BACKOFF_BASE_S ** (attempt - 1)
            print(f"[RETRY] Sleeping for {sleep_s:.2f}s before retry...")
            time.sleep(sleep_s)

    raise RuntimeError(
        f"Failed to get valid prediction for case_id={test_case.case_id}, modality={modality}. "
        f"Last error: {last_err}"
    )


# -----------------------------
# A-priori cost estimation
# -----------------------------

def _estimate_tokens_for_text(text: str) -> int:
    try:
        import tiktoken  # type: ignore

        try:
            enc = tiktoken.encoding_for_model(DEPLOYMENT)
        except Exception:
            enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return int(math.ceil(len(text) / 4.0))


def _messages_text_for_token_estimate(user_prompt: str) -> str:
    return f"system:\n{SYSTEM_MSG}\nuser:\n{user_prompt}"


def _common_prefix_len(strings: List[str]) -> int:
    if not strings:
        return 0
    prefix = strings[0]
    for s in strings[1:]:
        max_i = min(len(prefix), len(s))
        i = 0
        while i < max_i and prefix[i] == s[i]:
            i += 1
        prefix = prefix[:i]
        if not prefix:
            break
    return len(prefix)


def estimate_apriori_pipeline_cost(
    training_block: str,
    test_cases_with_idxs: List[Tuple[int, Case]],
    modality: str,
) -> Dict[str, Any]:
    prompt_texts = [
        _messages_text_for_token_estimate(build_user_prompt(training_block, c, row_index=idx, modality=modality))
        for idx, c in test_cases_with_idxs
    ]
    prompt_tokens_by_case = [_estimate_tokens_for_text(t) for t in prompt_texts]
    total_prompt_tokens = int(sum(prompt_tokens_by_case))

    common_prefix_chars = _common_prefix_len(prompt_texts)
    common_prefix_text = prompt_texts[0][:common_prefix_chars] if prompt_texts else ""
    common_prefix_tokens = _estimate_tokens_for_text(common_prefix_text) if prompt_texts else 0

    n_calls = len(prompt_texts)
    estimated_cached_tokens = int(common_prefix_tokens * max(n_calls - 1, 0))
    estimated_cached_tokens = min(estimated_cached_tokens, total_prompt_tokens)
    estimated_uncached_prompt_tokens = max(total_prompt_tokens - estimated_cached_tokens, 0)
    estimated_completion_tokens_upper = int(n_calls * MAX_TOKENS)

    no_cache_input_cost = (total_prompt_tokens / 1_000_000) * PRICE_PER_1M_INPUT_TOKENS
    cache_adjusted_input_cost = (
        (estimated_uncached_prompt_tokens / 1_000_000) * PRICE_PER_1M_INPUT_TOKENS
        + (estimated_cached_tokens / 1_000_000) * PRICE_PER_1M_CACHED_INPUT_TOKENS
    )
    output_cost_upper = (estimated_completion_tokens_upper / 1_000_000) * PRICE_PER_1M_OUTPUT_TOKENS

    return {
        "n_calls": n_calls,
        "prompt_tokens_estimated_total": total_prompt_tokens,
        "prompt_tokens_estimated_min": min(prompt_tokens_by_case) if prompt_tokens_by_case else 0,
        "prompt_tokens_estimated_max": max(prompt_tokens_by_case) if prompt_tokens_by_case else 0,
        "prompt_tokens_estimated_mean": float(np.mean(prompt_tokens_by_case)) if prompt_tokens_by_case else 0.0,
        "common_prefix_tokens_estimated": common_prefix_tokens,
        "cached_prompt_tokens_estimated": estimated_cached_tokens,
        "uncached_prompt_tokens_estimated": estimated_uncached_prompt_tokens,
        "completion_tokens_upper_estimated": estimated_completion_tokens_upper,
        "estimated_no_cache_cost_usd_upper": no_cache_input_cost + output_cost_upper,
        "estimated_cache_adjusted_cost_usd_upper": cache_adjusted_input_cost + output_cost_upper,
        "estimated_cache_savings_usd": max(no_cache_input_cost - cache_adjusted_input_cost, 0.0),
    }


def summarize_apriori_costs(run_configs: List[RunConfig]) -> Dict[str, Any]:
    total = {
        "n_runs": len(run_configs),
        "n_calls": 0,
        "prompt_tokens_estimated_total": 0,
        "cached_prompt_tokens_estimated": 0,
        "uncached_prompt_tokens_estimated": 0,
        "completion_tokens_upper_estimated": 0,
        "estimated_no_cache_cost_usd_upper": 0.0,
        "estimated_cache_adjusted_cost_usd_upper": 0.0,
        "estimated_cache_savings_usd": 0.0,
    }
    for rc in run_configs:
        r = rc.apriori_cost
        for key in [
            "n_calls",
            "prompt_tokens_estimated_total",
            "cached_prompt_tokens_estimated",
            "uncached_prompt_tokens_estimated",
            "completion_tokens_upper_estimated",
        ]:
            total[key] += int(r[key])
        for key in [
            "estimated_no_cache_cost_usd_upper",
            "estimated_cache_adjusted_cost_usd_upper",
            "estimated_cache_savings_usd",
        ]:
            total[key] += float(r[key])
    return total


def print_apriori_cost_report(run_configs: List[RunConfig]) -> None:
    total = summarize_apriori_costs(run_configs)
    print("\n[A-PRIORI TOKEN / COST ESTIMATE ACROSS ALL REQUESTED RUNS]")
    print(f"model/deployment:                         {DEPLOYMENT}")
    print(f"planned shotsettier runs:                {total['n_runs']}")
    print(f"planned prediction calls:                 {total['n_calls']}")
    print(f"estimated prompt tokens total:            {total['prompt_tokens_estimated_total']}")
    print(f"estimated cached prompt tokens:           {total['cached_prompt_tokens_estimated']}")
    print(f"estimated uncached prompt tokens:         {total['uncached_prompt_tokens_estimated']}")
    print(f"completion-token upper bound:             {total['completion_tokens_upper_estimated']} ({MAX_TOKENS} max_completion_tokens/call)")
    print(f"estimated no-cache cost upper bound:      ${total['estimated_no_cache_cost_usd_upper']:.8f}")
    print(f"estimated cache-adjusted cost upper bound:${total['estimated_cache_adjusted_cost_usd_upper']:.8f}")
    print(f"estimated cache savings:                  ${total['estimated_cache_savings_usd']:.8f}")
    print("NOTE: This is a local estimate only. Final cost uses API response usage fields.")
    print("\n[RUN BREAKDOWN]")
    for rc in run_configs:
        r = rc.apriori_cost
        print(
            f"  {rc.shotset_name}/{rc.modality}: "
            f"calls={r['n_calls']} "
            f"prompt_mean={r['prompt_tokens_estimated_mean']:.1f} "
            f"prompt_min={r['prompt_tokens_estimated_min']} "
            f"prompt_max={r['prompt_tokens_estimated_max']} "
            f"skipped_missing_mri={len(rc.skipped_missing_mri)} "
            f"est_cost_cache_adj=${r['estimated_cache_adjusted_cost_usd_upper']:.8f}"
        )


def confirm_before_full_run(
    run_configs: List[RunConfig],
    assume_yes: bool = False,
    *,
    resume: bool = True,
    skip_completed_configs: bool = True,
    force_rerun_cases: bool = False,
) -> None:
    print_apriori_cost_report(run_configs)
    resume_summary = summarize_resume_plan(
        run_configs,
        resume=resume,
        skip_completed_configs=skip_completed_configs,
        force_rerun_cases=force_rerun_cases,
    )
    print_resume_plan(resume_summary)
    if assume_yes:
        print("[CONFIRM] --yes supplied; continuing without interactive prompt.")
        return
    answer = input("\nContinue with the full 3-tier LLM pipeline? Type 'yes' to proceed: ").strip().lower()
    if answer not in {"yes", "y"}:
        print("[ABORT] User did not confirm. No prediction calls were made.")
        raise SystemExit(0)


# -----------------------------
# Evaluation helpers
# -----------------------------

STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "for", "on", "with", "is", "are",
    "was", "were", "by", "at", "from", "as", "this", "that", "it", "be", "has", "have",
    "had", "but", "not", "no", "into", "than", "then", "there", "their", "its", "also",
    "may", "can", "which", "within", "without", "after", "before", "left", "right",
    "breast", "tumor", "carcinoma", "report", "reports", "provided", "tier",
}


def coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def coerce_int01(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    return s.where(s.isin([0, 1]), np.nan)


def parse_jsonish_list(x: Any) -> List[str]:
    if isinstance(x, list):
        return [str(v) for v in x]
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []
    s = str(x).strip()
    if not s:
        return []
    try:
        val = json.loads(s)
        if isinstance(val, list):
            return [str(v) for v in val]
    except Exception:
        pass
    try:
        val = ast.literal_eval(s)
        if isinstance(val, list):
            return [str(v) for v in val]
    except Exception:
        pass
    return [s]


def prepare_predictions_for_eval(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "dispersion_true_high_low" not in df.columns:
        df["dispersion_true_high_low"] = df["dispersion_true"].apply(
            lambda x: int(float(x) >= DISPERSION_HIGH_THRESHOLD) if pd.notna(x) else np.nan
        )
    for col in ["row_index", "dispersion_true", "dispersion_score_pred"]:
        if col in df.columns:
            df[col] = coerce_numeric(df[col])
    for col in [
        "relapse_true",
        "relapse_pred",
        "dispersion_true_high_low",
        "dispersion_high_low_pred",
        "retrieval_token_exact_match",
    ]:
        if col in df.columns:
            df[col] = coerce_int01(df[col])
    if "key_evidence" in df.columns:
        df["key_evidence_list"] = df["key_evidence"].apply(parse_jsonish_list)
    else:
        df["key_evidence_list"] = [[] for _ in range(len(df))]
    return df


def missingness_summary(df: pd.DataFrame) -> str:
    cols = [
        "dispersion_true",
        "dispersion_true_high_low",
        "dispersion_score_pred",
        "dispersion_high_low_pred",
        "relapse_true",
        "relapse_pred",
        "row_index",
        "retrieval_token_exact_match",
    ]
    lines = ["Missingness / validity summary (count of NaN):"]
    for c in cols:
        if c in df.columns:
            lines.append(f"  {c:<34} {int(df[c].isna().sum())}")
    return "\n".join(lines)


def evaluate_dispersion(df: pd.DataFrame) -> Tuple[str, pd.DataFrame, Dict[str, Any]]:
    mask = np.isfinite(df["dispersion_true"]) & np.isfinite(df["dispersion_score_pred"])
    used = df.loc[mask].copy()
    if len(used) == 0:
        return "Dispersion score (regression):\n  No valid rows.", used, {}

    y_true = used["dispersion_true"].astype(float).values
    y_pred = used["dispersion_score_pred"].astype(float).values
    mae = mean_absolute_error(y_true, y_pred)
    rmse = _rmse(y_true, y_pred)
    rho = spearmanr(y_true, y_pred).correlation
    rho_val = float(rho) if (rho is not None and not np.isnan(rho)) else np.nan

    lines = [
        "Dispersion score (regression):",
        f"  N_used = {len(used)} / {len(df)}",
        f"  MAE  = {mae:.4f}",
        f"  RMSE = {rmse:.4f}",
        f"  Spearman rho = {rho_val:.4f}" if np.isfinite(rho_val) else "  Spearman rho = nan",
    ]
    return "\n".join(lines), used, {"mae": mae, "rmse": rmse, "spearman_rho": rho_val}


def evaluate_dispersion_high_low(df: pd.DataFrame) -> Tuple[str, pd.DataFrame, Dict[str, Any]]:
    mask = df["dispersion_true_high_low"].notna() & df["dispersion_high_low_pred"].notna()
    used = df.loc[mask].copy()
    if len(used) == 0:
        return "Dispersion high/low classification:\n  No valid rows.", used, {}

    y_true = used["dispersion_true_high_low"].astype(int).values
    y_pred = used["dispersion_high_low_pred"].astype(int).values
    acc = accuracy_score(y_true, y_pred)
    f1v = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    lines = [
        f"Dispersion high/low (true high defined as dispersion_true >= {int(DISPERSION_HIGH_THRESHOLD)}):",
        f"  N_used = {len(used)} / {len(df)}",
        f"  Accuracy = {acc:.4f}",
        f"  F1       = {f1v:.4f}",
        "  Confusion matrix (rows=true [0,1], cols=pred [0,1]):",
        f"  {cm.tolist()}",
    ]
    return "\n".join(lines), used, {"accuracy": acc, "f1": f1v, "confusion_matrix": cm}


def evaluate_relapse_labels(df: pd.DataFrame) -> Tuple[str, pd.DataFrame, Dict[str, Any]]:
    mask = df["relapse_true"].notna() & df["relapse_pred"].notna()
    used = df.loc[mask].copy()
    if len(used) == 0:
        return "Relapse classification:\n  No valid rows.", used, {}

    y_true = used["relapse_true"].astype(int).values
    y_pred = used["relapse_pred"].astype(int).values
    acc = accuracy_score(y_true, y_pred)
    f1v = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    lines = [
        "Relapse (classification; label-only):",
        f"  N_used = {len(used)} / {len(df)}",
        f"  Accuracy = {acc:.4f}",
        f"  F1       = {f1v:.4f}",
        "  Confusion matrix (rows=true [0,1], cols=pred [0,1]):",
        f"  {cm.tolist()}",
    ]
    return "\n".join(lines), used, {"accuracy": acc, "f1": f1v, "confusion_matrix": cm}


def _safe_auroc_auprc(y_true: np.ndarray, scores: np.ndarray) -> Tuple[Optional[float], Optional[float], str]:
    uniq = np.unique(y_true)
    if len(uniq) < 2:
        return None, None, "Only one class present in y_true; AUROC/AUPRC undefined."
    try:
        auroc = float(roc_auc_score(y_true, scores))
    except Exception as e:
        return None, None, f"AUROC failed: {e}"
    try:
        auprc = float(average_precision_score(y_true, scores))
    except Exception as e:
        return auroc, None, f"AUPRC failed: {e}"
    return auroc, auprc, "ok"


def _best_f1_threshold(y_true: np.ndarray, scores: np.ndarray) -> Tuple[float, float]:
    vals = np.unique(scores[~np.isnan(scores)])
    if len(vals) == 0:
        return 0.5, float("nan")
    if len(vals) > 400:
        vals = np.unique(np.quantile(scores, np.linspace(0.0, 1.0, 401)))

    best_t = float(vals[0])
    best_f1 = -1.0
    for t in vals:
        pred = (scores >= t).astype(int)
        f1v = f1_score(y_true, pred, zero_division=0)
        if f1v > best_f1:
            best_f1 = float(f1v)
            best_t = float(t)
    return best_t, best_f1


def compare_relapse_predictors(df: pd.DataFrame) -> Tuple[str, Dict[str, Any]]:
    base = df[df["relapse_true"].notna()].copy()
    if len(base) == 0:
        return "Relapse predictor comparison:\n  No valid rows with relapse_true.", {}

    metrics: Dict[str, Any] = {}
    lines = [
        "Relapse predictor comparison (AUROC/AUPRC/F1):",
        f"  N_with_relapse_true = {len(base)} / {len(df)}",
        "",
    ]

    a_mask = base["relapse_pred"].notna()
    if a_mask.sum() > 0:
        y_pred_lbl = base.loc[a_mask, "relapse_pred"].astype(int).values
        y_true_a = base.loc[a_mask, "relapse_true"].astype(int).values
        acc = accuracy_score(y_true_a, y_pred_lbl)
        f1v = f1_score(y_true_a, y_pred_lbl, zero_division=0)
        auroc, auprc, note = _safe_auroc_auprc(y_true_a, y_pred_lbl.astype(float))
        metrics["llm_relapse_pred"] = {"accuracy": acc, "f1": f1v, "auroc": auroc, "auprc": auprc, "note": note}
        lines.append("  A) LLM relapse_pred (binary label):")
        lines.append(f"     Accuracy={acc:.4f}  F1={f1v:.4f}  AUROC={auroc if auroc is not None else 'NA'}  AUPRC={auprc if auprc is not None else 'NA'}")
        lines.append(f"     Note: {note}")
    else:
        lines.append("  A) LLM relapse_pred: no valid predictions.")
    lines.append("")

    b_mask = np.isfinite(base["dispersion_score_pred"])
    if b_mask.sum() > 0:
        scores = base.loc[b_mask, "dispersion_score_pred"].astype(float).values
        y_true_b = base.loc[b_mask, "relapse_true"].astype(int).values
        auroc, auprc, note = _safe_auroc_auprc(y_true_b, scores)
        t_best, f1_best = _best_f1_threshold(y_true_b, scores)
        metrics["predicted_dispersion_score"] = {"auroc": auroc, "auprc": auprc, "note": note, "best_f1": f1_best, "best_threshold": t_best}
        lines.append("  B) Predicted dispersion score (continuous risk score):")
        lines.append(f"     AUROC={auroc if auroc is not None else 'NA'}  AUPRC={auprc if auprc is not None else 'NA'}  BestF1@threshold={f1_best:.4f} (t={t_best:.4f})")
    else:
        lines.append("  B) Predicted dispersion score: no valid values.")
    lines.append("")

    c_mask = np.isfinite(base["dispersion_true"])
    if c_mask.sum() > 0:
        scores = base.loc[c_mask, "dispersion_true"].astype(float).values
        y_true_c = base.loc[c_mask, "relapse_true"].astype(int).values
        auroc, auprc, note = _safe_auroc_auprc(y_true_c, scores)
        t_best, f1_best = _best_f1_threshold(y_true_c, scores)
        metrics["true_dispersion_score"] = {"auroc": auroc, "auprc": auprc, "note": note, "best_f1": f1_best, "best_threshold": t_best}
        lines.append("  C) True dispersion score (continuous risk score; upper-bound signal check):")
        lines.append(f"     AUROC={auroc if auroc is not None else 'NA'}  AUPRC={auprc if auprc is not None else 'NA'}  BestF1@threshold={f1_best:.4f} (t={t_best:.4f})")
    else:
        lines.append("  C) True dispersion score: no valid values.")

    return "\n".join(lines), metrics


def evaluate_needle_retrieval(df: pd.DataFrame) -> Tuple[str, Dict[str, Any]]:
    if "retrieval_token_exact_match" not in df.columns or len(df) == 0:
        return "Needle retrieval:\n  No rows.", {}
    exact = df["retrieval_token_exact_match"].dropna().astype(int)
    rate = float(exact.mean()) if len(exact) else np.nan
    failures = int((df["retrieval_token_exact_match"] == 0).sum())
    lines = [
        "Needle-in-the-haystack retrieval evaluation:",
        f"  N_rows = {len(df)}",
        f"  Single-token exact retrieval rate = {rate:.4f}" if np.isfinite(rate) else "  Single-token exact retrieval rate = NA",
        f"  Single-token failures = {failures}",
    ]
    return "\n".join(lines), {"single_token_rate": rate, "single_token_failures": failures}


def normalize_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s\-\/]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def tokenize_quote(s: str) -> List[str]:
    s = normalize_text(s)
    return [t for t in s.split() if len(t) >= 3 and t not in STOPWORDS]


def extract_ngram_features(quotes: List[str], max_n: int = 2) -> List[str]:
    feats = set()
    for q in quotes:
        toks = tokenize_quote(q)
        for t in toks:
            feats.add(t)
        if max_n >= 2:
            for i in range(len(toks) - 1):
                feats.add(f"{toks[i]} {toks[i + 1]}")
    return sorted(feats)


def build_evidence_feature_table(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    if label_col not in df.columns:
        return pd.DataFrame()
    work = df[df[label_col].isin([0, 1])].copy()
    if len(work) == 0:
        return pd.DataFrame()

    case_features: List[Tuple[int, set[str]]] = []
    for _, row in work.iterrows():
        feats = set(extract_ngram_features(row.get("key_evidence_list", [])))
        label = int(row[label_col])
        case_features.append((label, feats))

    pos_cases = [fs for y, fs in case_features if y == 1]
    neg_cases = [fs for y, fs in case_features if y == 0]
    n_pos = len(pos_cases)
    n_neg = len(neg_cases)

    pos_counter: Counter[str] = Counter()
    neg_counter: Counter[str] = Counter()
    all_feats: set[str] = set()

    for fs in pos_cases:
        for f in fs:
            pos_counter[f] += 1
            all_feats.add(f)
    for fs in neg_cases:
        for f in fs:
            neg_counter[f] += 1
            all_feats.add(f)

    alpha = 0.5
    rows = []
    for feat in all_feats:
        a = pos_counter[feat]
        b = neg_counter[feat]
        support = a + b
        odds_ratio = ((a + alpha) / (n_pos - a + alpha)) / ((b + alpha) / (n_neg - b + alpha)) if (n_pos > 0 and n_neg > 0) else np.nan
        rows.append({
            "feature": feat,
            "pos_count": a,
            "neg_count": b,
            "support": support,
            "odds_ratio_pos_vs_neg": odds_ratio,
        })
    tbl = pd.DataFrame(rows)
    if len(tbl):
        tbl = tbl.sort_values(["support", "odds_ratio_pos_vs_neg"], ascending=[False, False]).reset_index(drop=True)
    return tbl


def evidence_attribution_report(df: pd.DataFrame) -> Tuple[str, Dict[str, pd.DataFrame]]:
    outputs: Dict[str, pd.DataFrame] = {}
    lines: List[str] = []
    for label_col, title in [
        ("dispersion_high_low_pred", "Predicted dispersion high (1) vs low (0)"),
        ("relapse_pred", "Predicted relapse (1) vs non-relapse (0)"),
    ]:
        tbl = build_evidence_feature_table(df, label_col)
        outputs[label_col] = tbl
        lines.append(f"Evidence attribution analysis: {title}")
        if len(tbl) == 0:
            lines.append("  No valid rows / features.")
            lines.append("")
            continue
        tbl2 = tbl[tbl["support"] >= 3].copy()
        if len(tbl2) == 0:
            lines.append("  No features with support >= 3 cases.")
            lines.append("")
            continue
        top_pos = tbl2.sort_values(["odds_ratio_pos_vs_neg", "support"], ascending=[False, False]).head(10)
        top_neg = tbl2.sort_values(["odds_ratio_pos_vs_neg", "support"], ascending=[True, False]).head(10)
        lines.append("  Top features associated with class=1:")
        for _, r in top_pos.iterrows():
            lines.append(f"    {r['feature']}: pos_count={int(r['pos_count'])}, neg_count={int(r['neg_count'])}, support={int(r['support'])}, OR={r['odds_ratio_pos_vs_neg']:.3f}")
        lines.append("  Top features associated with class=0:")
        for _, r in top_neg.iterrows():
            lines.append(f"    {r['feature']}: pos_count={int(r['pos_count'])}, neg_count={int(r['neg_count'])}, support={int(r['support'])}, OR={r['odds_ratio_pos_vs_neg']:.3f}")
        lines.append("")
    return "\n".join(lines), outputs


# -----------------------------
# Plotting
# -----------------------------

def plot_dispersion_scatter(used: pd.DataFrame, out_path: str, title_suffix: str) -> None:
    y_true = used["dispersion_true"].astype(float).values
    y_pred = used["dispersion_score_pred"].astype(float).values
    plt.figure()
    plt.scatter(y_true, y_pred)
    lo = float(np.nanmin([y_true.min(), y_pred.min()]))
    hi = float(np.nanmax([y_true.max(), y_pred.max()]))
    if lo == hi:
        lo -= 1.0
        hi += 1.0
    plt.plot([lo, hi], [lo, hi])
    plt.xlabel("True dispersion score")
    plt.ylabel("Predicted dispersion score")
    plt.title(f"True vs Predicted Dispersion Score ({title_suffix})")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_dispersion_residuals(used: pd.DataFrame, out_path: str, title_suffix: str) -> None:
    residuals = used["dispersion_score_pred"].astype(float).values - used["dispersion_true"].astype(float).values
    plt.figure()
    plt.hist(residuals, bins=25)
    plt.xlabel("Residual (predicted - true)")
    plt.ylabel("Count")
    plt.title(f"Dispersion Score Residuals ({title_suffix})")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_label_confusion_matrix(cm: np.ndarray, out_path: str, title: str) -> None:
    plt.figure()
    plt.imshow(cm)
    plt.title(title)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.xticks([0, 1], ["0", "1"])
    plt.yticks([0, 1], ["0", "1"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(int(cm[i, j])), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_pred_dispersion_by_relapse(df: pd.DataFrame, out_path: str, title_suffix: str) -> None:
    mask = df["relapse_true"].notna() & np.isfinite(df["dispersion_score_pred"])
    used = df.loc[mask].copy()
    if len(used) == 0:
        return
    g0 = used.loc[used["relapse_true"] == 0, "dispersion_score_pred"].astype(float).values
    g1 = used.loc[used["relapse_true"] == 1, "dispersion_score_pred"].astype(float).values
    plt.figure()
    if len(g0) > 0 and len(g1) > 0:
        plt.violinplot([g0, g1], showmeans=True, showextrema=True)
        plt.xticks([1, 2], ["Relapse=0", "Relapse=1"])
    else:
        plt.hist(used["dispersion_score_pred"].astype(float).values, bins=25)
        plt.xlabel("Predicted dispersion score")
    plt.ylabel("Predicted dispersion score")
    plt.title(f"Predicted Dispersion by True Relapse ({title_suffix})")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_relapse_predictor_comparison(metrics: Dict[str, Any], out_path: str, title_suffix: str) -> None:
    methods = ["llm_relapse_pred", "predicted_dispersion_score", "true_dispersion_score"]
    labels = ["LLM relapse_pred", "Pred disp score", "True disp score"]
    auroc_vals: List[float] = []
    auprc_vals: List[float] = []
    f1_vals: List[float] = []
    for m in methods:
        d = metrics.get(m, {})
        auroc_vals.append(d.get("auroc", np.nan) if d.get("auroc", None) is not None else np.nan)
        auprc_vals.append(d.get("auprc", np.nan) if d.get("auprc", None) is not None else np.nan)
        f1_vals.append(d.get("f1", np.nan) if m == "llm_relapse_pred" else d.get("best_f1", np.nan))
    x = np.arange(len(methods))
    width = 0.25
    plt.figure()
    plt.bar(x - width, auroc_vals, width)
    plt.bar(x, auprc_vals, width)
    plt.bar(x + width, f1_vals, width)
    plt.xticks(x, labels, rotation=20, ha="right")
    plt.ylabel("Metric value")
    plt.title(f"Relapse Predictor Comparison ({title_suffix})")
    plt.legend(["AUROC", "AUPRC", "F1"])
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_needle_retrieval_rates(df: pd.DataFrame, out_path: str, title_suffix: str) -> None:
    if "retrieval_token_exact_match" not in df.columns or len(df) == 0:
        return
    val = float(df["retrieval_token_exact_match"].mean())
    plt.figure()
    plt.bar(["Single token"], [val])
    plt.ylim(0, 1.05)
    plt.ylabel("Exact retrieval rate")
    plt.title(f"Needle Retrieval Accuracy ({title_suffix})")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_top_evidence_features(tbl: pd.DataFrame, out_path: str, title: str, positive: bool = True, min_support: int = 3, top_k: int = 15) -> None:
    if tbl is None or len(tbl) == 0:
        return
    sub = tbl[tbl["support"] >= min_support].copy()
    if len(sub) == 0:
        return
    sub = sub.sort_values("odds_ratio_pos_vs_neg", ascending=not positive).head(top_k)
    labels = sub["feature"].astype(str).tolist()
    vals = sub["odds_ratio_pos_vs_neg"].astype(float).tolist()
    plt.figure(figsize=(10, 6))
    y = np.arange(len(labels))
    plt.barh(y, vals)
    plt.yticks(y, labels)
    plt.xlabel("Smoothed odds ratio (class 1 vs class 0)")
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# -----------------------------
# Evaluation runner
# -----------------------------

def explanation_text() -> str:
    return (
        "What the evaluation measures\n"
        "----------------------------\n"
        "Dispersion score (regression)\n"
        "- MAE: average |predicted - true| dispersion score. Lower is better.\n"
        "- RMSE: sqrt(mean((predicted - true)^2)). Penalizes large errors more than MAE.\n"
        "- Spearman rho: rank correlation between true and predicted dispersion.\n"
        "\n"
        "Dispersion high/low (classification)\n"
        "- True high/low is derived from the true dispersion score using the cutoff >= 85.\n"
        "- Accuracy and F1 evaluate the predicted high/low label.\n"
        "- Confusion matrix rows are true labels [0,1] and columns are predicted labels [0,1].\n"
        "\n"
        "Relapse (classification; label-only)\n"
        "- Accuracy and F1 compare relapse_pred against relapse_true.\n"
        "\n"
        "Relapse predictor comparison\n"
        "- A) LLM relapse_pred is a binary label.\n"
        "- B) predicted dispersion score is evaluated as a continuous risk score for relapse.\n"
        "- C) true dispersion score is an upper-bound signal check.\n"
        "\n"
        "Needle-in-the-haystack retrieval\n"
        "- A single synthetic non-clinical token is inserted outside report text.\n"
        "- Exact retrieval checks prompt attention / context retention.\n"
        "\n"
        "Evidence attribution analysis\n"
        "- Evidence quotes are converted into case-level lexical features.\n"
        "- Smoothed odds ratios summarize which evidence terms are associated with each predicted class.\n"
    )


def evaluate_and_plot(pred_df: pd.DataFrame, out_dir: str, title_suffix: str) -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)
    df = prepare_predictions_for_eval(pred_df)

    out_txt = os.path.join(out_dir, "evaluation_metrics_from_csv.txt")
    out_explain = os.path.join(out_dir, "evaluation_explanation.txt")
    metrics_json = os.path.join(out_dir, "evaluation_metrics_summary.json")

    scatter_png = os.path.join(out_dir, "dispersion_true_vs_pred_scatter.png")
    resid_png = os.path.join(out_dir, "dispersion_residuals_hist.png")
    dhl_cm_png = os.path.join(out_dir, "dispersion_high_low_confusion_matrix.png")
    relapse_cm_png = os.path.join(out_dir, "relapse_confusion_matrix.png")
    pred_disp_by_relapse_png = os.path.join(out_dir, "predicted_dispersion_by_true_relapse.png")
    relapse_predictor_compare_png = os.path.join(out_dir, "relapse_predictor_comparison.png")
    needle_rates_png = os.path.join(out_dir, "needle_retrieval_rates.png")
    evidence_disp_pos_png = os.path.join(out_dir, "evidence_features_dispersion_high.png")
    evidence_disp_neg_png = os.path.join(out_dir, "evidence_features_dispersion_low.png")
    evidence_rel_pos_png = os.path.join(out_dir, "evidence_features_relapse_yes.png")
    evidence_rel_neg_png = os.path.join(out_dir, "evidence_features_relapse_no.png")
    evidence_disp_csv = os.path.join(out_dir, "evidence_attribution_dispersion_high_low.csv")
    evidence_rel_csv = os.path.join(out_dir, "evidence_attribution_relapse_yes_no.csv")

    miss = missingness_summary(df)
    disp_report, disp_used, disp_metrics = evaluate_dispersion(df)
    dhl_report, _, dhl_metrics = evaluate_dispersion_high_low(df)
    rel_report, _, rel_metrics = evaluate_relapse_labels(df)
    rel_comp_report, rel_comp_metrics = compare_relapse_predictors(df)
    needle_report, needle_metrics = evaluate_needle_retrieval(df)
    evidence_report, evidence_outputs = evidence_attribution_report(df)

    report = "\n".join([
        "=== SecureGPT Evaluation from Predictions CSV ===",
        f"Run: {title_suffix}",
        f"Total rows: {len(df)}",
        "",
        miss,
        "",
        disp_report,
        "",
        dhl_report,
        "",
        rel_report,
        "",
        rel_comp_report,
        "",
        needle_report,
        "",
        evidence_report,
        "",
    ])

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(report)
    with open(out_explain, "w", encoding="utf-8") as f:
        f.write(explanation_text())

    if len(disp_used) > 0:
        plot_dispersion_scatter(disp_used, scatter_png, title_suffix)
        plot_dispersion_residuals(disp_used, resid_png, title_suffix)
    if "confusion_matrix" in dhl_metrics:
        plot_label_confusion_matrix(
            dhl_metrics["confusion_matrix"],
            dhl_cm_png,
            f"Dispersion High/Low Confusion Matrix ({title_suffix})",
        )
    if "confusion_matrix" in rel_metrics:
        plot_label_confusion_matrix(
            rel_metrics["confusion_matrix"],
            relapse_cm_png,
            f"Relapse Confusion Matrix ({title_suffix})",
        )
    plot_pred_dispersion_by_relapse(df, pred_disp_by_relapse_png, title_suffix)
    plot_relapse_predictor_comparison(rel_comp_metrics, relapse_predictor_compare_png, title_suffix)
    plot_needle_retrieval_rates(df, needle_rates_png, title_suffix)

    disp_tbl = evidence_outputs.get("dispersion_high_low_pred", pd.DataFrame())
    rel_tbl = evidence_outputs.get("relapse_pred", pd.DataFrame())
    if len(disp_tbl):
        disp_tbl.to_csv(evidence_disp_csv, index=False)
        plot_top_evidence_features(disp_tbl, evidence_disp_pos_png, f"Evidence Features: Predicted High Dispersion ({title_suffix})", positive=True)
        plot_top_evidence_features(disp_tbl, evidence_disp_neg_png, f"Evidence Features: Predicted Low Dispersion ({title_suffix})", positive=False)
    if len(rel_tbl):
        rel_tbl.to_csv(evidence_rel_csv, index=False)
        plot_top_evidence_features(rel_tbl, evidence_rel_pos_png, f"Evidence Features: Predicted Relapse ({title_suffix})", positive=True)
        plot_top_evidence_features(rel_tbl, evidence_rel_neg_png, f"Evidence Features: Predicted Non-Relapse ({title_suffix})", positive=False)

    summary = {
        "n_rows": int(len(df)),
        "dispersion_regression": {k: _json_safe(v) for k, v in disp_metrics.items()},
        "dispersion_high_low": {k: _json_safe(v) for k, v in dhl_metrics.items()},
        "relapse_label": {k: _json_safe(v) for k, v in rel_metrics.items()},
        "relapse_predictor_comparison": _json_safe(rel_comp_metrics),
        "needle_retrieval": _json_safe(needle_metrics),
    }
    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[EVAL] Wrote metrics + plots to: {out_dir}")
    return summary


def _json_safe(x: Any) -> Any:
    if isinstance(x, np.ndarray):
        return _json_safe(x.tolist())
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        xf = float(x)
        return xf if math.isfinite(xf) else None
    if isinstance(x, dict):
        return {k: _json_safe(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_json_safe(v) for v in x]
    return x


# -----------------------------
# Run planning and execution
# -----------------------------

def validate_shot_rows(df: pd.DataFrame, shotset: Dict[str, Any]) -> None:
    rows = list(shotset["high_rows"]) + list(shotset["low_rows"])
    if len(rows) != len(set(rows)):
        raise ValueError(f"Duplicate training rows in {shotset['name']}: {rows}")
    if min(rows) < 0 or max(rows) >= len(df):
        raise ValueError(
            f"Training rows for {shotset['name']} out of range for dataframe length {len(df)}: {rows}. "
            "Row numbering is 0-based pandas iloc."
        )
    if len(shotset["high_rows"]) != 2 or len(shotset["low_rows"]) != 2:
        raise ValueError(f"Each shot set must have exactly 2 high and 2 low rows. Got: {shotset}")


def validate_training_modality_availability(training_cases: List[Tuple[int, Case]], modality: str) -> None:
    bad: List[str] = []
    for idx, c in training_cases:
        if modality_requires_mri(modality) and not _has_report_text(c.preop_mri):
            bad.append(f"row {idx} case_id={c.case_id} missing MRI")
        if modality_uses_pathology(modality) and not _has_report_text(c.path_report):
            bad.append(f"row {idx} case_id={c.case_id} missing pathology")
    if bad:
        raise ValueError(
            f"Cannot build {modality} training block because required reports are missing:\n"
            + "\n".join(f"  - {b}" for b in bad)
        )


def build_run_configs(df: pd.DataFrame, root_out_dir: str) -> List[RunConfig]:
    run_configs: List[RunConfig] = []
    all_idxs = list(range(len(df)))

    for shotset in SHOT_SETS:
        validate_shot_rows(df, shotset)
        high_rows = list(shotset["high_rows"])
        low_rows = list(shotset["low_rows"])
        training_rows = high_rows + low_rows
        training_cases_with_idxs = [(idx, make_case_from_row(df, idx)) for idx in training_rows]

        for modality in MODALITY_TIERS:
            validate_training_modality_availability(training_cases_with_idxs, modality)
            training_cases = [c for _, c in training_cases_with_idxs]
            training_block = build_training_block(training_cases, modality)

            test_idxs_all = [i for i in all_idxs if i not in set(training_rows)]
            test_cases_with_idxs: List[Tuple[int, Case]] = []
            skipped_missing_mri: List[Tuple[int, Case]] = []
            for idx in test_idxs_all:
                c = make_case_from_row(df, idx)
                if modality_requires_mri(modality) and not _has_report_text(c.preop_mri):
                    skipped_missing_mri.append((idx, c))
                    continue
                test_cases_with_idxs.append((idx, c))

            run_out_dir = os.path.join(root_out_dir, shotset["name"], modality)
            apriori_cost = estimate_apriori_pipeline_cost(training_block, test_cases_with_idxs, modality)
            run_configs.append(
                RunConfig(
                    shotset_name=shotset["name"],
                    high_rows=high_rows,
                    low_rows=low_rows,
                    training_rows=training_rows,
                    modality=modality,
                    run_out_dir=run_out_dir,
                    training_block=training_block,
                    test_cases_with_idxs=test_cases_with_idxs,
                    skipped_missing_mri=skipped_missing_mri,
                    apriori_cost=apriori_cost,
                )
            )
    return run_configs


def write_run_config(rc: RunConfig) -> None:
    os.makedirs(rc.run_out_dir, exist_ok=True)
    path = os.path.join(rc.run_out_dir, "run_config.json")
    payload = {
        "shotset_name": rc.shotset_name,
        "high_rows": rc.high_rows,
        "low_rows": rc.low_rows,
        "training_rows": rc.training_rows,
        "modality": rc.modality,
        "n_test_cases": len(rc.test_cases_with_idxs),
        "n_skipped_missing_mri": len(rc.skipped_missing_mri),
        "skipped_missing_mri_rows": [idx for idx, _ in rc.skipped_missing_mri],
        "apriori_cost": rc.apriori_cost,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def write_skipped_cases(rc: RunConfig) -> None:
    if not rc.skipped_missing_mri:
        return
    rows = []
    for idx, c in rc.skipped_missing_mri:
        rows.append({
            "row_index": idx,
            "case_id": c.case_id,
            "index_side": c.index_side,
            "skip_reason": "missing_preop_MRI_text_required_for_this_tier",
            "has_preop_mri": _has_report_text(c.preop_mri),
            "has_path_report": _has_report_text(c.path_report),
        })
    pd.DataFrame(rows).to_csv(os.path.join(rc.run_out_dir, "skipped_cases_missing_mri.csv"), index=False)


def empty_predictions_df() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "shotset_name",
        "modality",
        "case_id",
        "row_index",
        "index_side",
        "has_preop_mri",
        "has_path_report",
        "dispersion_true",
        "dispersion_true_high_low",
        "relapse_true",
        "dispersion_score_pred",
        "dispersion_high_low_pred",
        "relapse_pred",
        "key_evidence",
        "retrieval_token_expected",
        "retrieval_check_token_returned",
        "retrieval_check_correct_reported",
        "retrieval_token_exact_match",
        "reasoning_summary",
        "structured_rationale",
    ])


# -----------------------------
# Resume / checkpoint helpers
# -----------------------------

def _config_resume_dir(run_out_dir: str) -> str:
    path = os.path.join(run_out_dir, RESUME_CHECKPOINT_SUBDIR)
    os.makedirs(path, exist_ok=True)
    return path


def _config_completed_marker_path(run_out_dir: str) -> str:
    return os.path.join(_config_resume_dir(run_out_dir), "COMPLETED.json")


def build_config_fingerprint(rc: RunConfig) -> Dict[str, Any]:
    return {
        "resume_script_version": RESUME_SCRIPT_VERSION,
        "shotset_name": rc.shotset_name,
        "high_rows": list(rc.high_rows),
        "low_rows": list(rc.low_rows),
        "training_rows": list(rc.training_rows),
        "modality": rc.modality,
        "n_test_cases": len(rc.test_cases_with_idxs),
        "test_row_indices": [idx for idx, _ in rc.test_cases_with_idxs],
        "skipped_missing_mri_rows": [idx for idx, _ in rc.skipped_missing_mri],
        "deployment": DEPLOYMENT,
        "max_tokens": MAX_TOKENS,
    }


def config_fingerprints_compatible(saved: Dict[str, Any], rc: RunConfig) -> Tuple[bool, str]:
    current = build_config_fingerprint(rc)
    keys = [
        "resume_script_version",
        "shotset_name",
        "high_rows",
        "low_rows",
        "training_rows",
        "modality",
        "n_test_cases",
        "test_row_indices",
        "skipped_missing_mri_rows",
    ]
    for key in keys:
        if saved.get(key) != current.get(key):
            return False, f"{key}: saved={saved.get(key)!r} current={current.get(key)!r}"
    return True, ""


def _parse_jsonish_dict(x: Any) -> Dict[str, Any]:
    if isinstance(x, dict):
        return x
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return {}
    s = str(x).strip()
    if not s:
        return {}
    try:
        val = json.loads(s)
        if isinstance(val, dict):
            return val
    except Exception:
        pass
    try:
        val = ast.literal_eval(s)
        if isinstance(val, dict):
            return val
    except Exception:
        pass
    return {}


def _normalize_pred_record(record: Dict[str, Any]) -> Dict[str, Any]:
    rec = dict(record)
    if isinstance(rec.get("key_evidence"), str):
        rec["key_evidence"] = parse_jsonish_list(rec["key_evidence"])
    if isinstance(rec.get("structured_rationale"), str):
        rec["structured_rationale"] = _parse_jsonish_dict(rec["structured_rationale"])
    if "retrieval_check_correct_reported" in rec and "retrieval_check_correct" not in rec:
        rec["retrieval_check_correct"] = bool(rec["retrieval_check_correct_reported"])
    return rec


def build_pred_record(
    rc: RunConfig,
    idx: int,
    test_case: Case,
    pred_obj: Dict[str, Any],
    validation_token: str,
) -> Dict[str, Any]:
    returned_token = str(pred_obj["retrieval_check_token_returned"])
    token_match = int(returned_token == validation_token)
    dispersion_true_high_low = (
        int(test_case.dispersion_true >= DISPERSION_HIGH_THRESHOLD)
        if test_case.dispersion_true is not None
        else np.nan
    )
    return {
        "shotset_name": rc.shotset_name,
        "modality": rc.modality,
        "case_id": test_case.case_id,
        "row_index": idx,
        "index_side": test_case.index_side,
        "has_preop_mri": _has_report_text(test_case.preop_mri),
        "has_path_report": _has_report_text(test_case.path_report),
        "dispersion_true": test_case.dispersion_true,
        "dispersion_true_high_low": dispersion_true_high_low,
        "relapse_true": test_case.relapse_true,
        "dispersion_score_pred": float(pred_obj["dispersion_score_pred"]),
        "dispersion_high_low_pred": int(pred_obj["dispersion_high_low_pred"]),
        "relapse_pred": int(pred_obj["relapse_pred"]),
        "key_evidence": pred_obj["key_evidence"],
        "retrieval_token_expected": validation_token,
        "retrieval_check_token_returned": returned_token,
        "retrieval_check_correct_reported": bool(pred_obj["retrieval_check_correct"]),
        "retrieval_token_exact_match": token_match,
        "reasoning_summary": str(pred_obj["reasoning_summary"]),
        "structured_rationale": pred_obj["structured_rationale"],
    }


def validate_saved_pred_record(
    record: Dict[str, Any],
    rc: RunConfig,
    idx: int,
    test_case: Case,
) -> Tuple[bool, str]:
    rec = _normalize_pred_record(record)
    try:
        row_index = int(rec.get("row_index", -1))
    except (TypeError, ValueError):
        return False, "row_index is not an integer"
    if row_index != int(idx):
        return False, f"row_index mismatch: got {row_index} expected {idx}"
    if str(rec.get("shotset_name", "")) != str(rc.shotset_name):
        return False, "shotset_name mismatch"
    if str(rec.get("modality", "")) != str(rc.modality):
        return False, "modality mismatch"
    if str(rec.get("case_id", "")) != str(test_case.case_id):
        return False, f"case_id mismatch: got {rec.get('case_id')} expected {test_case.case_id}"

    expected_token = _make_case_token(test_case, idx, rc.modality)
    if str(rec.get("retrieval_token_expected", "")) != expected_token:
        return False, "retrieval_token_expected does not match current prompt/token logic"

    obj = {
        "dispersion_score_pred": rec.get("dispersion_score_pred"),
        "dispersion_high_low_pred": rec.get("dispersion_high_low_pred"),
        "relapse_pred": rec.get("relapse_pred"),
        "key_evidence": rec.get("key_evidence"),
        "retrieval_check_token_returned": rec.get("retrieval_check_token_returned"),
        "retrieval_check_correct": rec.get("retrieval_check_correct", rec.get("retrieval_check_correct_reported")),
        "reasoning_summary": rec.get("reasoning_summary"),
        "structured_rationale": rec.get("structured_rationale"),
    }
    ok, msg = _validate_prediction_obj(
        obj,
        expected_case_id=test_case.case_id,
        expected_token=expected_token,
    )
    if not ok:
        return False, msg
    return True, ""


def _ingest_saved_prediction(
    raw: Dict[str, Any],
    rc: RunConfig,
    test_by_idx: Dict[int, Case],
    source: str,
    line_ref: str,
    by_row: Dict[int, Dict[str, Any]],
    warnings: List[str],
) -> None:
    try:
        idx = int(raw.get("row_index"))
    except (TypeError, ValueError):
        warnings.append(f"{source} {line_ref}: missing/invalid row_index")
        return
    if idx not in test_by_idx:
        warnings.append(f"{source} {line_ref}: row_index={idx} not in current test set; ignored")
        return
    test_case = test_by_idx[idx]
    ok, msg = validate_saved_pred_record(raw, rc, idx, test_case)
    if not ok:
        warnings.append(f"{source} {line_ref}: row_index={idx} invalid: {msg}")
        return
    if idx in by_row:
        warnings.append(f"{source} {line_ref}: duplicate row_index={idx}; keeping latest valid record")
    by_row[idx] = _normalize_pred_record(raw)


def load_predictions_from_jsonl(
    path: str,
    rc: RunConfig,
    test_cases_with_idxs: List[Tuple[int, Case]],
) -> Tuple[Dict[int, Dict[str, Any]], List[str]]:
    by_row: Dict[int, Dict[str, Any]] = {}
    warnings: List[str] = []
    if not os.path.isfile(path):
        return by_row, warnings

    test_by_idx = {idx: c for idx, c in test_cases_with_idxs}
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                warnings.append(f"{path}:{line_no}: invalid JSON: {exc}")
                continue
            if not isinstance(raw, dict):
                warnings.append(f"{path}:{line_no}: record is not a JSON object")
                continue
            _ingest_saved_prediction(raw, rc, test_by_idx, path, f"line {line_no}", by_row, warnings)
    return by_row, warnings


def load_predictions_from_csv(
    path: str,
    rc: RunConfig,
    test_cases_with_idxs: List[Tuple[int, Case]],
) -> Tuple[Dict[int, Dict[str, Any]], List[str]]:
    by_row: Dict[int, Dict[str, Any]] = {}
    warnings: List[str] = []
    if not os.path.isfile(path):
        return by_row, warnings

    test_by_idx = {idx: c for idx, c in test_cases_with_idxs}
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        warnings.append(f"{path}: failed to read CSV: {exc}")
        return by_row, warnings

    for row_no, row in df.iterrows():
        raw = row.to_dict()
        _ingest_saved_prediction(raw, rc, test_by_idx, path, f"csv row {row_no}", by_row, warnings)
    return by_row, warnings


def load_existing_case_predictions(
    run_out_dir: str,
    rc: RunConfig,
) -> Tuple[Dict[int, Dict[str, Any]], str, List[str]]:
    jsonl_path = os.path.join(run_out_dir, "predictions_testing_cases.jsonl")
    csv_path = os.path.join(run_out_dir, "predictions_testing_cases.csv")
    warnings: List[str] = []

    by_row, jsonl_warnings = load_predictions_from_jsonl(jsonl_path, rc, rc.test_cases_with_idxs)
    warnings.extend(jsonl_warnings)
    source = "jsonl" if by_row else ""

    if len(by_row) < len(rc.test_cases_with_idxs):
        csv_by_row, csv_warnings = load_predictions_from_csv(csv_path, rc, rc.test_cases_with_idxs)
        warnings.extend(csv_warnings)
        for idx, rec in csv_by_row.items():
            if idx not in by_row:
                by_row[idx] = rec
        if csv_by_row:
            source = "jsonl+csv" if source else "csv"

    return by_row, source, warnings


def predictions_dict_to_dataframe(
    by_row: Dict[int, Dict[str, Any]],
    rc: RunConfig,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for idx, _ in rc.test_cases_with_idxs:
        if idx not in by_row:
            raise KeyError(f"Missing prediction for row_index={idx}")
        rows.append(by_row[idx])
    return pd.DataFrame(rows)


def write_predictions_jsonl(path: str, records: List[Dict[str, Any]]) -> None:
    text = "\n".join(json.dumps(r, ensure_ascii=False) for r in records)
    if text:
        text += "\n"
    _atomic_write_text(path, text)


def append_prediction_jsonl(path: str, record: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_predictions_csv(path: str, pred_df: pd.DataFrame) -> None:
    pred_df_for_csv = pred_df.copy()
    if len(pred_df_for_csv):
        pred_df_for_csv["key_evidence"] = pred_df_for_csv["key_evidence"].apply(
            lambda x: json.dumps(x, ensure_ascii=False)
        )
        pred_df_for_csv["structured_rationale"] = pred_df_for_csv["structured_rationale"].apply(
            lambda x: json.dumps(x, ensure_ascii=False)
        )
    tmp = f"{path}.tmp.{os.getpid()}"
    pred_df_for_csv.to_csv(tmp, index=False)
    os.replace(tmp, path)


def is_config_checkpoint_complete(run_out_dir: str, rc: RunConfig) -> bool:
    marker_path = _config_completed_marker_path(run_out_dir)
    if not os.path.isfile(marker_path):
        return False
    try:
        with open(marker_path, "r", encoding="utf-8") as f:
            marker = json.load(f)
    except Exception:
        return False
    ok, _ = config_fingerprints_compatible(marker.get("fingerprint", {}), rc)
    if not ok:
        return False
    by_row, _, _ = load_existing_case_predictions(run_out_dir, rc)
    return len(by_row) == len(rc.test_cases_with_idxs)


def save_completed_config_checkpoint(
    run_out_dir: str,
    rc: RunConfig,
    n_new_api_calls: int,
) -> None:
    marker = {
        "completed_at": datetime.now().isoformat(timespec="seconds"),
        "fingerprint": build_config_fingerprint(rc),
        "predictions_csv": os.path.join(run_out_dir, "predictions_testing_cases.csv"),
        "predictions_jsonl": os.path.join(run_out_dir, "predictions_testing_cases.jsonl"),
        "metrics_json": os.path.join(run_out_dir, "evaluation_metrics_summary.json"),
        "cost_json": os.path.join(run_out_dir, "token_cost_report.json"),
        "n_test_cases": len(rc.test_cases_with_idxs),
        "n_new_api_calls_last_session": n_new_api_calls,
        "session_estimated_cost_usd": float(COST_TRACKER["estimated_cost_usd"]),
    }
    marker_path = _config_completed_marker_path(run_out_dir)
    _atomic_write_json(marker_path, marker)
    print(f"[RESUME] Wrote completed config checkpoint: {marker_path}")


def load_completed_config_checkpoint(
    run_out_dir: str,
    rc: RunConfig,
) -> Optional[Tuple[pd.DataFrame, Dict[str, Any]]]:
    marker_path = _config_completed_marker_path(run_out_dir)
    if not os.path.isfile(marker_path):
        return None

    try:
        with open(marker_path, "r", encoding="utf-8") as f:
            marker = json.load(f)
    except Exception as exc:
        print(f"[RESUME] Ignoring unreadable completed checkpoint {marker_path}: {exc}")
        return None

    ok, msg = config_fingerprints_compatible(marker.get("fingerprint", {}), rc)
    if not ok:
        print(f"[RESUME] Ignoring completed checkpoint due to fingerprint mismatch: {msg}")
        return None

    by_row, source, warnings = load_existing_case_predictions(run_out_dir, rc)
    for w in warnings:
        print(f"[RESUME][WARN] {w}")
    if len(by_row) != len(rc.test_cases_with_idxs):
        print(
            f"[RESUME] Completed marker exists but only {len(by_row)}/{len(rc.test_cases_with_idxs)} "
            "valid case records were found; will rerun this config."
        )
        return None

    pred_csv = marker.get("predictions_csv") or os.path.join(run_out_dir, "predictions_testing_cases.csv")
    if not os.path.isfile(pred_csv):
        print(f"[RESUME] Completed marker exists but predictions CSV is missing: {pred_csv}")
        return None

    pred_df = predictions_dict_to_dataframe(by_row, rc)
    metrics_json = marker.get("metrics_json") or os.path.join(run_out_dir, "evaluation_metrics_summary.json")
    eval_summary: Dict[str, Any] = {}
    if os.path.isfile(metrics_json):
        try:
            with open(metrics_json, "r", encoding="utf-8") as f:
                eval_summary = json.load(f)
        except Exception as exc:
            print(f"[RESUME][WARN] Failed to load metrics JSON ({metrics_json}): {exc}")

    title_suffix = f"{rc.shotset_name} / {modality_display_name(rc.modality)}"
    if not eval_summary:
        print(f"[RESUME] Metrics JSON missing; recomputing evaluation for {title_suffix}")
        eval_summary = evaluate_and_plot(pred_df, run_out_dir, title_suffix)
    else:
        print(f"[RESUME] Loaded completed config from checkpoint ({source or 'marker'}): {marker_path}")
    return pred_df, eval_summary


def summarize_resume_plan(
    run_configs: List[RunConfig],
    *,
    resume: bool,
    skip_completed_configs: bool,
    force_rerun_cases: bool,
) -> Dict[str, Any]:
    summary = {
        "resume_enabled": resume,
        "skip_completed_configs": skip_completed_configs,
        "force_rerun_cases": force_rerun_cases,
        "n_configs_total": len(run_configs),
        "n_configs_skip_complete": 0,
        "n_configs_resume_cases": 0,
        "n_configs_refinalize": 0,
        "n_configs_run_fresh": 0,
        "n_cases_total": 0,
        "n_cases_already_done": 0,
        "n_cases_pending_api": 0,
        "per_config": [],
    }
    if not resume:
        for rc in run_configs:
            n_test = len(rc.test_cases_with_idxs)
            summary["n_cases_total"] += n_test
            summary["n_cases_pending_api"] += n_test
            summary["n_configs_run_fresh"] += 1
            summary["per_config"].append({
                "shotset_name": rc.shotset_name,
                "modality": rc.modality,
                "status": "fresh_no_resume",
                "n_done": 0,
                "n_pending": n_test,
            })
        return summary

    for rc in run_configs:
        n_test = len(rc.test_cases_with_idxs)
        summary["n_cases_total"] += n_test
        if skip_completed_configs and is_config_checkpoint_complete(rc.run_out_dir, rc):
            summary["n_configs_skip_complete"] += 1
            summary["n_cases_already_done"] += n_test
            status = "skip_complete"
            n_done = n_test
            n_pending = 0
        else:
            by_row, _, _ = (
                ({}, "", [])
                if force_rerun_cases
                else load_existing_case_predictions(rc.run_out_dir, rc)
            )
            n_done = len(by_row)
            n_pending = n_test - n_done
            summary["n_cases_already_done"] += n_done
            summary["n_cases_pending_api"] += n_pending
            if n_done > 0 and n_pending > 0:
                summary["n_configs_resume_cases"] += 1
                status = "resume_partial"
            elif n_done == 0:
                summary["n_configs_run_fresh"] += 1
                status = "fresh"
            else:
                summary["n_configs_refinalize"] += 1
                status = "all_cases_present_refinalize"
        summary["per_config"].append({
            "shotset_name": rc.shotset_name,
            "modality": rc.modality,
            "status": status,
            "n_done": n_done,
            "n_pending": n_pending,
        })
    return summary


def print_resume_plan(summary: Dict[str, Any]) -> None:
    print("\n[RESUME PLAN]")
    print(f"resume_enabled:           {summary['resume_enabled']}")
    print(f"skip_completed_configs:   {summary['skip_completed_configs']}")
    print(f"force_rerun_cases:        {summary['force_rerun_cases']}")
    print(f"configs total:            {summary['n_configs_total']}")
    print(f"configs skip complete:    {summary['n_configs_skip_complete']}")
    print(f"configs resume partial:   {summary['n_configs_resume_cases']}")
    print(f"configs refinalize only:  {summary['n_configs_refinalize']}")
    print(f"configs run fresh:        {summary['n_configs_run_fresh']}")
    print(f"cases total:              {summary['n_cases_total']}")
    print(f"cases already done:       {summary['n_cases_already_done']}")
    print(f"cases pending API:        {summary['n_cases_pending_api']}")
    for item in summary["per_config"]:
        print(
            f"  {item['shotset_name']}/{item['modality']}: "
            f"status={item['status']} done={item['n_done']} pending={item['n_pending']}"
        )


def run_one_config(
    rc: RunConfig,
    *,
    resume: bool = True,
    skip_completed_configs: bool = True,
    force_rerun_cases: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    os.makedirs(rc.run_out_dir, exist_ok=True)
    predictions_csv = os.path.join(rc.run_out_dir, "predictions_testing_cases.csv")
    predictions_jsonl = os.path.join(rc.run_out_dir, "predictions_testing_cases.jsonl")
    cost_json = os.path.join(rc.run_out_dir, "token_cost_report.json")
    completed_marker = _config_completed_marker_path(rc.run_out_dir)

    write_run_config(rc)
    write_skipped_cases(rc)

    if resume and skip_completed_configs:
        loaded = load_completed_config_checkpoint(rc.run_out_dir, rc)
        if loaded is not None:
            pred_df, eval_summary = loaded
            return pred_df, eval_summary

    if not resume or force_rerun_cases:
        if os.path.isfile(completed_marker):
            os.remove(completed_marker)
            print(f"[RESUME] Removed completed checkpoint: {completed_marker}")

    prior_cost = load_cost_tracker_snapshot(cost_json) if resume and not force_rerun_cases else None
    reset_cost_tracker()

    existing_by_row: Dict[int, Dict[str, Any]] = {}
    if resume and not force_rerun_cases:
        existing_by_row, source, warnings = load_existing_case_predictions(rc.run_out_dir, rc)
        for w in warnings:
            print(f"[RESUME][WARN] {w}")
        if existing_by_row:
            print(
                f"[RESUME] Loaded {len(existing_by_row)}/{len(rc.test_cases_with_idxs)} "
                f"existing case predictions from {source or 'disk'}"
            )

    if not resume or force_rerun_cases:
        open(predictions_jsonl, "w", encoding="utf-8").close()

    print("=" * 80)
    print(f"[RUN] {rc.shotset_name} / {rc.modality}")
    print(f"[RUN] Output directory: {rc.run_out_dir}")
    print(f"[RUN] Training rows: high={rc.high_rows}, low={rc.low_rows}")
    print(f"[RUN] Test cases: {len(rc.test_cases_with_idxs)}")
    print(f"[RUN] Skipped missing MRI: {len(rc.skipped_missing_mri)}")
    print(f"[RUN] Resume enabled: {resume} | force_rerun_cases: {force_rerun_cases}")
    print(f"[RUN] Cases already present: {len(existing_by_row)}")
    print(f"[RUN] Cases pending API: {len(rc.test_cases_with_idxs) - len(existing_by_row)}")
    print("=" * 80)

    n_new_api_calls = 0
    t0 = time.time()

    for n, (idx, test_case) in enumerate(rc.test_cases_with_idxs, 1):
        validation_token = _make_case_token(test_case, idx, rc.modality)
        print("-" * 80)
        print(
            f"[CASE] {n}/{len(rc.test_cases_with_idxs)} "
            f"row_index={idx} case_id={test_case.case_id} modality={rc.modality}"
        )

        if resume and not force_rerun_cases and idx in existing_by_row:
            pred_record = existing_by_row[idx]
            print(
                "[RESUME] Skipping API call; reusing saved prediction: "
                f"dispersion_pred={pred_record['dispersion_score_pred']:.2f} "
                f"dispersion_high_low_pred={pred_record['dispersion_high_low_pred']} "
                f"relapse_pred={pred_record['relapse_pred']}"
            )
            continue

        print(f"[CASE] preop_MRI_chars={len(test_case.preop_mri)} path_report_chars={len(test_case.path_report)}")
        pred_obj = predict_case(rc.training_block, test_case, row_index=idx, modality=rc.modality)
        pred_record = build_pred_record(rc, idx, test_case, pred_obj, validation_token)
        existing_by_row[idx] = pred_record
        append_prediction_jsonl(predictions_jsonl, pred_record)
        n_new_api_calls += 1

        print(
            "[CASE] Prediction OK: "
            f"dispersion_pred={pred_record['dispersion_score_pred']:.2f} "
            f"dispersion_high_low_pred={pred_record['dispersion_high_low_pred']} "
            f"relapse_pred={pred_record['relapse_pred']} "
            f"token_match={pred_record['retrieval_token_exact_match']}"
        )
        time.sleep(RATE_LIMIT_SLEEP_S)

    elapsed = time.time() - t0
    print(
        f"[DONE] Inference pass complete for {rc.shotset_name}/{rc.modality} in {elapsed / 60:.2f} minutes. "
        f"new_api_calls={n_new_api_calls}"
    )

    if len(existing_by_row) != len(rc.test_cases_with_idxs):
        missing = [idx for idx, _ in rc.test_cases_with_idxs if idx not in existing_by_row]
        raise RuntimeError(
            f"Incomplete predictions for {rc.shotset_name}/{rc.modality}. "
            f"Missing row indices: {missing}"
        )

    ordered_records = [existing_by_row[idx] for idx, _ in rc.test_cases_with_idxs]
    write_predictions_jsonl(predictions_jsonl, ordered_records)
    pred_df = predictions_dict_to_dataframe(existing_by_row, rc)
    write_predictions_csv(predictions_csv, pred_df)
    print(f"[SAVE] Wrote predictions JSONL: {predictions_jsonl}")
    print(f"[SAVE] Wrote predictions CSV: {predictions_csv}")

    print_cumulative_report()
    save_cumulative_report_json(cost_json, prior=prior_cost if resume else None)
    print(f"[COST] Wrote token/cost report: {cost_json}")

    title_suffix = f"{rc.shotset_name} / {modality_display_name(rc.modality)}"
    eval_summary = evaluate_and_plot(pred_df, rc.run_out_dir, title_suffix)
    save_completed_config_checkpoint(rc.run_out_dir, rc, n_new_api_calls)
    return pred_df, eval_summary


def save_aggregate_summary(root_out_dir: str, summaries: List[Dict[str, Any]]) -> None:
    rows = []
    for s in summaries:
        metrics = s.get("metrics", {})
        rows.append({
            "shotset_name": s["shotset_name"],
            "modality": s["modality"],
            "n_predictions": metrics.get("n_rows"),
            "n_skipped_missing_mri": s.get("n_skipped_missing_mri"),
            "dispersion_mae": metrics.get("dispersion_regression", {}).get("mae"),
            "dispersion_rmse": metrics.get("dispersion_regression", {}).get("rmse"),
            "dispersion_spearman_rho": metrics.get("dispersion_regression", {}).get("spearman_rho"),
            "dispersion_high_low_accuracy": metrics.get("dispersion_high_low", {}).get("accuracy"),
            "dispersion_high_low_f1": metrics.get("dispersion_high_low", {}).get("f1"),
            "relapse_accuracy": metrics.get("relapse_label", {}).get("accuracy"),
            "relapse_f1": metrics.get("relapse_label", {}).get("f1"),
            "needle_single_token_rate": metrics.get("needle_retrieval", {}).get("single_token_rate"),
        })
    df = pd.DataFrame(rows)
    path = os.path.join(root_out_dir, "all_tiers_metrics_summary.csv")
    df.to_csv(path, index=False)
    print(f"[SUMMARY] Wrote aggregate metrics summary: {path}")


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="SecureGPT 3-tier dispersion/relapse pipeline + evaluation")
    parser.add_argument("--csv-path", "-c", default=CSV_PATH, help="Path to input CSV file.")
    parser.add_argument("--outdir", "-o", default=OUT_DIR, help="Root directory to write all shotset/tier outputs.")
    parser.add_argument("--env-path", default=ENV_PATH, help="Path to .env containing SANDBOX_API_KEY.")
    parser.add_argument("--deployment", default=DEPLOYMENT, help="SecureGPT/Azure deployment name.")
    parser.add_argument("--api-version", default=API_VERSION, help="Azure OpenAI API version.")
    parser.add_argument("--yes", "-y", action="store_true", help="Skip the interactive a-priori cost confirmation prompt.")
    parser.add_argument("--skip-preflight", action="store_true", help="Skip the initial small API connectivity test.")
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse per-case JSONL checkpoints and completed config markers in --outdir (default: enabled).",
    )
    parser.add_argument(
        "--skip-completed-configs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When resuming, skip shotset/modality folders with a valid completed checkpoint (default: enabled).",
    )
    parser.add_argument(
        "--force-rerun-cases",
        action="store_true",
        help="Ignore per-case checkpoints and call SecureGPT again for every test case in non-skipped configs.",
    )
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    log_path = os.path.join(args.outdir, "run.log")
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    log_mode = "a" if args.resume and os.path.isfile(log_path) else "w"

    with open(log_path, log_mode, encoding="utf-8") as log_f:
        sys.stdout = Tee(original_stdout, log_f)
        sys.stderr = Tee(original_stderr, log_f)
        try:
            if log_mode == "a":
                print("\n" + "=" * 80)
                print(f"[RESUME SESSION] {datetime.now().isoformat(timespec='seconds')}")
            print("=" * 80)
            print("[START] SecureGPT 3-tier dispersion/relapse pipeline + evaluation")
            print(f"[START] CSV_PATH={args.csv_path}")
            print(f"[START] OUT_DIR={args.outdir}")
            print(f"[START] ENV_PATH={args.env_path}")
            print(f"[START] DEPLOYMENT={args.deployment}")
            print(f"[START] API_VERSION={args.api_version}")
            print(f"[START] LOG_PATH={log_path}")
            print(f"[START] RESUME={args.resume}")
            print(f"[START] SKIP_COMPLETED_CONFIGS={args.skip_completed_configs}")
            print(f"[START] FORCE_RERUN_CASES={args.force_rerun_cases}")
            print("=" * 80)

            configure_api(args.env_path, args.deployment, args.api_version)
            df = load_cases(args.csv_path)
            run_configs = build_run_configs(df, args.outdir)

            confirm_before_full_run(
                run_configs,
                assume_yes=args.yes,
                resume=args.resume,
                skip_completed_configs=args.skip_completed_configs,
                force_rerun_cases=args.force_rerun_cases,
            )
            if not args.skip_preflight:
                reset_cost_tracker()
                preflight_check()
                print_cumulative_report()

            aggregate_summaries: List[Dict[str, Any]] = []
            for rc in run_configs:
                _, metrics = run_one_config(
                    rc,
                    resume=args.resume,
                    skip_completed_configs=args.skip_completed_configs,
                    force_rerun_cases=args.force_rerun_cases,
                )
                aggregate_summaries.append({
                    "shotset_name": rc.shotset_name,
                    "modality": rc.modality,
                    "n_skipped_missing_mri": len(rc.skipped_missing_mri),
                    "metrics": metrics,
                })

            save_aggregate_summary(args.outdir, aggregate_summaries)
            print("[END] All shotset/tier runs complete.")
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            sys.stdout = original_stdout
            sys.stderr = original_stderr


if __name__ == "__main__":
    main()
