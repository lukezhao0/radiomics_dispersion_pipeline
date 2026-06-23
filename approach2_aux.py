#!/usr/bin/env python3
"""
SecureGPT lexical feature discovery helper / standalone pipeline for breast tumor
spatial dispersiveness NLP.

SUMMARY OF EDITED VERSION
=========================
This file preserves the original SecureGPT extraction workflow but revises the
MRI extraction task so it is deliberately more permissive and ontology-aware.
The downstream nested evaluator can now use pathology as a training-time teacher
for MRI language because each extracted phrase carries richer provenance:

- exact quote grounding remains mandatory
- normalized phrase / candidate concept fields are requested
- MRI phrases may be definite, possible, ambiguous, benign/nonspecific, or
  imaging-pattern-only, rather than only obvious tumor statements
- polarity, uncertainty, directness, confidence, section, and quantitative hints
  are preserved when the model supplies them
- older extraction outputs remain valid because missing optional fields are
  normalized to safe defaults before validation/writing
- summary of pipeline v2 changes: new functionality for saving intermediate steps, run this in the future

Primary usage inside the nested evaluator:

python feature_discovery_eval_ML.py \
  --csv-path /path/to/cases.csv \
  --out_dir /path/to/out \
  --enable-pathology-calibration

Standalone extraction usage:

python feature_discovery_eval_ML_aux.py \
  --csv-path /path/to/cases.csv \
  --outdir /path/to/extractions_mri \
  --report-mode mri \
  --max-api-workers 2

If your repository historically used this file as feature_discovery_pipeline.py,
you may keep that name; feature_discovery_eval_ML.py imports these helper
functions from feature_discovery_eval_ML_aux.py by default.
"""

from __future__ import annotations

import argparse
import contextlib
from concurrent.futures import ThreadPoolExecutor, as_completed
import html
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, TextIO, Tuple

import traceback
import unicodedata
from datetime import datetime
from difflib import SequenceMatcher

import numpy as np
import pandas as pd
from dotenv import load_dotenv
import requests
import threading

# -----------------------------
# Config
# -----------------------------

CSV_PATH = "/Users/lukezhao/projects/onc/PROCESSED_TRIMMED_path_cases_MRI_status.csv"

REPORT_CONFIG = {
    "mri": {
        "field": "preop_MRI_text",
        "label": "post-neoadjuvant, pre-surgery breast MRI report",
        "outdir_suffix": "mri",
    },
    "path": {
        "field": "path_report_text",
        "label": "post-neoadjuvant pre-surgical pathology report",
        "outdir_suffix": "path",
    },
}

DEFAULT_REPORT_MODE = "mri"

# Prefer the project-level .env used by the Stanford AI Sandbox example, but
# fall back to the current working directory so the script remains portable.
ENV_PATH = os.getenv("SANDBOX_ENV_PATH", "/Users/lukezhao/projects/onc/.env")
if os.path.exists(ENV_PATH):
    load_dotenv(ENV_PATH, override=True)
else:
    load_dotenv(os.path.join(os.getcwd(), ".env"), override=True)

API_VERSION = "2024-12-01-preview"
DEPLOYMENT = "gpt-5-nano"  # GPT-5-nano Global deployment name on the SHC AI Sandbox.
SANDBOX_API_KEY = os.getenv("SANDBOX_API_KEY")

if not SANDBOX_API_KEY:
    raise RuntimeError(
        "SANDBOX_API_KEY not found. Set it in /Users/lukezhao/projects/onc/.env, "
        "set SANDBOX_ENV_PATH to another .env file, or export SANDBOX_API_KEY."
    )
SANDBOX_API_KEY = SANDBOX_API_KEY.strip()

URL = (
    "https://aihubapi.stanfordhealthcare.org/azure-openai"
    f"/deployments/{DEPLOYMENT}/chat/completions"
    f"?api-version={API_VERSION}"
)

HEADERS = {
    # This endpoint works with `api-key`, not Ocp-Apim-Subscription-Key.
    "api-key": SANDBOX_API_KEY,
    "Content-Type": "application/json",
    "Accept": "application/json",
}

TEMPERATURE = 0.0
# Includes visible output tokens plus GPT-5 reasoning tokens. Override from the
# shell with MAX_COMPLETION_TOKENS if you need a smaller/larger cap.
MAX_TOKENS = int(os.getenv("MAX_COMPLETION_TOKENS", "16000"))
MAX_RETRIES = 3
BACKOFF_BASE_S = 1.5
RATE_LIMIT_SLEEP_S = 0.25

# MAX_SEED_PHRASES = int(os.getenv("MAX_SEED_PHRASES", "15"))
# MAX_DENOVO_PHRASES = int(os.getenv("MAX_DENOVO_PHRASES", "15"))
REASONING_EFFORT = os.getenv("REASONING_EFFORT", "minimal").strip().lower()
if REASONING_EFFORT in {"", "none", "null"}:
    REASONING_EFFORT = ""

# Editable pricing assumptions for GPT-5-nano Global.
# Units: USD per 1,000,000 tokens.
PRICE_PER_1M_INPUT_TOKENS = 0.05
PRICE_PER_1M_CACHED_INPUT_TOKENS = 0.01
PRICE_PER_1M_OUTPUT_TOKENS = 0.40

SHARED_ONTOLOGY_GUIDANCE = """
SHARED BIOLOGICAL CONCEPT ONTOLOGY FOR DISPERSION-RELEVANT LANGUAGE

Extract phrases that may map to one or more of these concept names. Use these
names in candidate_concepts when appropriate. Do not force a concept if the text
does not support it.

- extent_span: long span, large extent, broad area, disease/tumor bed spanning a dimension
- multiplicity: multiple foci, multifocality, several residual sites
- multicentricity_separate_sites: separate quadrants/regions/sites, multicentric disease
- distribution_linear_segmental_regional: linear, segmental, ductal, regional distribution
- fragmentation_scattered_patchy_discontinuous: patchy, scattered, discontinuous, skip-like pattern
- residual_tumor_presence: residual enhancement or residual viable carcinoma/disease
- non_mass_enhancement: NME or non-mass enhancement pattern
- invasive_disease: invasive carcinoma/disease component
- in_situ_disease_dcis: DCIS / ductal carcinoma in situ / in-situ component
- treatment_response: complete, near-complete, partial, poor, decreased enhancement/cellularity
- treatment_effect_tumor_bed: treatment effect, tumor bed, fibrosis, therapy-related changes
- localized_compact_residual: single focal mass/focus, localized/compact residual disease
- diffuse_scattered_residual: diffuse or scattered residual disease/enhancement
- lymphovascular_invasion: LVI / lymphovascular invasion
- margin_proximity: margin, close/positive margin, distance from margin
- benign_or_nonspecific_enhancement: nonspecific, background, probably benign enhancement
""".strip()

SEED_GUIDANCE = f"""
DISPERSIVENESS SEED GUIDANCE (USE ONLY AS INITIAL CUEING, NOT AS A CLOSED VOCABULARY)

Concept families of interest include:
- spatial scatter / scattered foci / satellites
- multifocality / multicentricity
- discontinuity / separated foci / patchy or discontinuous disease
- infiltrative spread / irregular infiltrative residual disease
- broad extent / long span / large area involved
- localization / compact single residual focus
- minimal residual disease / near-complete response / no substantial residual disease

MRI-oriented examples:
- non-mass enhancement
- clumped / segmental / linear / regional enhancement
- patchy enhancement
- diffuse residual enhancement rather than one compact mass
- broad extent of abnormal enhancement
- scattered enhancing foci or satellites
- possible, nonspecific, indeterminate, favored benign, or treatment-related enhancement
- comparison-to-prior response language such as decreased, resolved, persistent, residual

Pathology-oriented examples:
- multiple residual invasive foci
- discontinuous residual carcinoma
- extensive residual DCIS
- lymphovascular invasion
- satellites / separate microscopic foci
- close margins / broad span of disease
- minimal residual disease / focal residual disease

{SHARED_ONTOLOGY_GUIDANCE}
""".strip()

SYSTEM_MSG = (
    "You are an advanced, careful clinical NLP model operating in a PHI-secure environment. "
    "Your task is lexical feature discovery. "
    "Use only the provided report text. Do not invent content not present in the report. "
    "Return valid JSON only. "
    "For every extracted phrase, copy an exact quote from the report text. "
    "Do not reveal hidden chain-of-thought. Provide only the requested structured fields."
)


# -----------------------------
# Token / cost tracking helpers
# -----------------------------

COST_TRACKER_LOCK = threading.Lock()
COST_TRACKER: Dict[str, Any] = {
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

# Global API gate used when fold-level and modality-level parallelism are enabled.
# Thread pools may create many workers, but only this many active HTTP requests
# can be in flight across the entire Python process.
GLOBAL_API_CONCURRENCY_LOCK = threading.Lock()
GLOBAL_API_MAX_CONCURRENCY = 1
GLOBAL_API_SEMAPHORE = threading.BoundedSemaphore(GLOBAL_API_MAX_CONCURRENCY)


def configure_global_api_concurrency(max_concurrent_requests: int) -> None:
    """Set a process-wide cap on simultaneous Stanford AI Sandbox calls."""
    global GLOBAL_API_MAX_CONCURRENCY, GLOBAL_API_SEMAPHORE
    max_concurrent_requests = max(1, int(max_concurrent_requests or 1))
    with GLOBAL_API_CONCURRENCY_LOCK:
        GLOBAL_API_MAX_CONCURRENCY = max_concurrent_requests
        GLOBAL_API_SEMAPHORE = threading.BoundedSemaphore(max_concurrent_requests)
    print(f"[API_CONCURRENCY] Global API request cap set to {max_concurrent_requests}.")


def get_global_api_concurrency() -> int:
    with GLOBAL_API_CONCURRENCY_LOCK:
        return int(GLOBAL_API_MAX_CONCURRENCY)


def estimate_cost_from_usage(usage: Dict[str, Any]) -> Dict[str, Any]:
    """Estimate post-LLM cost from one API response usage object.

    Billing assumption:
      uncached input tokens use input-token price
      cached input tokens use cached-input-token price
      completion tokens use output-token price

    For GPT-5-style models, completion_tokens can include both visible output
    and hidden reasoning tokens, so all completion_tokens are priced as output.
    """
    usage = usage or {}
    prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
    completion_tokens = int(usage.get("completion_tokens", 0) or 0)
    total_tokens = int(usage.get("total_tokens", 0) or 0)

    prompt_details = usage.get("prompt_tokens_details", {}) or {}
    completion_details = usage.get("completion_tokens_details", {}) or {}

    cached_tokens = int(prompt_details.get("cached_tokens", 0) or 0)
    reasoning_tokens = int(completion_details.get("reasoning_tokens", 0) or 0)
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


def estimate_cost_from_token_counts(
    prompt_tokens: int,
    completion_tokens: int,
    cached_tokens: int = 0,
) -> Dict[str, Any]:
    """Estimate cost from token counts before or after an API call."""
    prompt_tokens = int(prompt_tokens or 0)
    completion_tokens = int(completion_tokens or 0)
    cached_tokens = min(max(int(cached_tokens or 0), 0), prompt_tokens)
    usage = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "prompt_tokens_details": {"cached_tokens": cached_tokens},
        "completion_tokens_details": {"reasoning_tokens": 0},
    }
    return estimate_cost_from_usage(usage)


def update_cost_tracker(cost_info: Dict[str, Any]) -> None:
    """Thread-safe cumulative token/cost accounting."""
    with COST_TRACKER_LOCK:
        COST_TRACKER["calls"] += 1
        COST_TRACKER["prompt_tokens"] += int(cost_info["prompt_tokens"])
        COST_TRACKER["cached_tokens"] += int(cost_info["cached_tokens"])
        COST_TRACKER["uncached_prompt_tokens"] += int(cost_info["uncached_prompt_tokens"])
        COST_TRACKER["completion_tokens"] += int(cost_info["completion_tokens"])
        COST_TRACKER["reasoning_tokens"] += int(cost_info["reasoning_tokens"])
        COST_TRACKER["total_tokens"] += int(cost_info["total_tokens"])
        COST_TRACKER["estimated_cost_usd"] += float(cost_info["estimated_cost_usd"])
        COST_TRACKER["estimated_cache_savings_usd"] += float(cost_info["estimated_cache_savings_usd"])


def get_cost_tracker_snapshot() -> Dict[str, Any]:
    with COST_TRACKER_LOCK:
        return dict(COST_TRACKER)


def print_cumulative_report() -> None:
    snapshot = get_cost_tracker_snapshot()
    print("\n[CUMULATIVE TOKEN / COST REPORT]")
    print(f"model:                    {DEPLOYMENT}")
    print(f"pricing_per_1M:           input=${PRICE_PER_1M_INPUT_TOKENS:.4f} cached_input=${PRICE_PER_1M_CACHED_INPUT_TOKENS:.4f} output=${PRICE_PER_1M_OUTPUT_TOKENS:.4f}")
    print(f"calls:                    {snapshot['calls']}")
    print(f"prompt_tokens:            {snapshot['prompt_tokens']}")
    print(f"cached_tokens:            {snapshot['cached_tokens']}")
    print(f"uncached_prompt_tokens:   {snapshot['uncached_prompt_tokens']}")
    print(f"completion_tokens:        {snapshot['completion_tokens']}")
    print(f"reasoning_tokens:         {snapshot['reasoning_tokens']}")
    print(f"total_tokens:             {snapshot['total_tokens']}")
    print(f"estimated_total_cost_usd: ${snapshot['estimated_cost_usd']:.8f}")
    print(f"estimated_cache_savings:  ${snapshot['estimated_cache_savings_usd']:.8f}")


def write_cost_tracker_json(out_dir: str, filename: str = "llm_token_cost_report.json") -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    payload = get_cost_tracker_snapshot()
    payload.update({
        "model": DEPLOYMENT,
        "api_version": API_VERSION,
        "price_per_1M_input_tokens": PRICE_PER_1M_INPUT_TOKENS,
        "price_per_1M_cached_input_tokens": PRICE_PER_1M_CACHED_INPUT_TOKENS,
        "price_per_1M_output_tokens": PRICE_PER_1M_OUTPUT_TOKENS,
        "written_at": datetime.now().isoformat(timespec="seconds"),
    })
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"[SAVE] Wrote LLM token/cost report: {path}")
    return path


def _rough_token_count(text: str) -> int:
    """Approximate tokens without requiring an external tokenizer.

    If tiktoken is installed locally, use it. Otherwise use a conservative
    mixed character/word heuristic. This is for pre-run estimates only; actual
    billing uses the API response usage object.
    """
    text = str(text or "")
    try:
        import tiktoken  # type: ignore
        try:
            enc = tiktoken.encoding_for_model(DEPLOYMENT)
        except Exception:
            enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        words = len(re.findall(r"\S+", text))
        chars = len(text)
        return max(1, int(max(chars / 4.0, words * 1.33)))


def build_chat_messages(prompt: str) -> List[Dict[str, str]]:
    """Build the exact chat message structure sent to the API.

    Keeping the long, stable instruction block before the case-specific text
    gives repeated calls a shared prompt prefix, improving the chance that the
    platform can cache input tokens across cases.
    """
    return [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user", "content": prompt},
    ]


def estimate_prompt_tokens_from_messages(messages: Sequence[Dict[str, str]]) -> int:
    # Small per-message overhead approximates Chat Completions formatting.
    return sum(_rough_token_count(m.get("content", "")) + 4 for m in messages) + 3


def estimate_prompt_tokens_for_case(case: "Case", report_mode: str) -> int:
    return estimate_prompt_tokens_from_messages(build_chat_messages(build_user_prompt(case, report_mode)))


def estimate_static_cached_prefix_tokens(report_mode: str) -> int:
    """Approximate the stable prefix that can be cached within one modality.

    This excludes case_id, index_side, and report text. It is an optimistic
    estimate; actual cached_tokens are taken only from API usage.
    """
    dummy = Case(case_id="__CASE_ID__", preop_mri="", path_report="", index_side="__INDEX_SIDE__")
    prompt = build_user_prompt(dummy, report_mode)
    marker = "CASE\n"
    if marker in prompt:
        prompt = prompt.split(marker, 1)[0] + marker
    return estimate_prompt_tokens_from_messages(build_chat_messages(prompt))


def summarize_apriori_cost_estimate(
    prompt_token_counts: Sequence[int],
    report_modes: Sequence[str],
    max_completion_tokens: int = MAX_TOKENS,
    assume_static_prefix_cache: bool = True,
) -> Dict[str, Any]:
    """Summarize conservative and cache-aware a-priori cost estimates."""
    prompt_counts = [int(x) for x in prompt_token_counts]
    modes = [str(x) for x in report_modes]
    if len(prompt_counts) != len(modes):
        raise ValueError("prompt_token_counts and report_modes must have the same length.")

    total_prompt_tokens = sum(prompt_counts)
    total_completion_cap_tokens = len(prompt_counts) * int(max_completion_tokens)
    no_cache = estimate_cost_from_token_counts(
        prompt_tokens=total_prompt_tokens,
        completion_tokens=total_completion_cap_tokens,
        cached_tokens=0,
    )

    estimated_cached_tokens = 0
    if assume_static_prefix_cache and prompt_counts:
        seen_mode_counts: Dict[str, int] = {}
        static_prefix_by_mode = {mode: estimate_static_cached_prefix_tokens(mode) for mode in sorted(set(modes))}
        for prompt_tokens, mode in zip(prompt_counts, modes):
            seen_mode_counts[mode] = seen_mode_counts.get(mode, 0) + 1
            if seen_mode_counts[mode] > 1:
                estimated_cached_tokens += min(prompt_tokens, static_prefix_by_mode.get(mode, 0))

    cache_aware = estimate_cost_from_token_counts(
        prompt_tokens=total_prompt_tokens,
        completion_tokens=total_completion_cap_tokens,
        cached_tokens=estimated_cached_tokens,
    )

    return {
        "model": DEPLOYMENT,
        "api_version": API_VERSION,
        "n_calls": len(prompt_counts),
        "estimated_prompt_tokens": total_prompt_tokens,
        "estimated_completion_cap_tokens": total_completion_cap_tokens,
        "max_completion_tokens_per_call": int(max_completion_tokens),
        "no_cache_estimated_cost_usd": no_cache["estimated_cost_usd"],
        "cache_aware_estimated_cost_usd": cache_aware["estimated_cost_usd"],
        "cache_aware_estimated_cached_tokens": estimated_cached_tokens,
        "cache_aware_estimated_cache_savings_usd": cache_aware["estimated_cache_savings_usd"],
        "price_per_1M_input_tokens": PRICE_PER_1M_INPUT_TOKENS,
        "price_per_1M_cached_input_tokens": PRICE_PER_1M_CACHED_INPUT_TOKENS,
        "price_per_1M_output_tokens": PRICE_PER_1M_OUTPUT_TOKENS,
    }


def print_apriori_cost_estimate_report(estimate: Dict[str, Any], label: str = "planned pipeline") -> None:
    print("\n[A-PRIORI LLM TOKEN / COST ESTIMATE]")
    print(f"scope:                         {label}")
    print(f"model:                         {estimate['model']}")
    print(f"pricing_per_1M:                input=${estimate['price_per_1M_input_tokens']:.4f} cached_input=${estimate['price_per_1M_cached_input_tokens']:.4f} output=${estimate['price_per_1M_output_tokens']:.4f}")
    print(f"planned_llm_calls:             {estimate['n_calls']}")
    print(f"estimated_prompt_tokens:       {estimate['estimated_prompt_tokens']}")
    print(f"max_completion_tokens_per_call:{estimate['max_completion_tokens_per_call']}")
    print(f"completion_token_cap_total:    {estimate['estimated_completion_cap_tokens']}")
    print(f"estimated_cost_no_cache:       ${estimate['no_cache_estimated_cost_usd']:.8f}")
    print(f"cache_aware_estimated_cost:    ${estimate['cache_aware_estimated_cost_usd']:.8f}")
    print(f"cache_aware_cached_tokens:     {estimate['cache_aware_estimated_cached_tokens']}")
    print(f"cache_aware_cache_savings:     ${estimate['cache_aware_estimated_cache_savings_usd']:.8f}")
    print("[A-PRIORI NOTE] This uses the full prompts that will be sent, plus the configured completion-token cap. Actual cost is recomputed from API usage after each call.")


def confirm_cost_estimate_or_exit(estimate: Dict[str, Any], assume_yes: bool = False) -> None:
    if assume_yes:
        print("[CONFIRM] --yes supplied; continuing without interactive cost confirmation.")
        return
    if not sys.stdin.isatty():
        raise RuntimeError(
            "Interactive cost confirmation is required but stdin is not a TTY. "
            "Rerun with --yes after reviewing the printed a-priori estimate."
        )
    reply = input("Continue with LLM extraction calls? Type YES to continue: ").strip()
    if reply != "YES":
        print("[ABORT] User did not type YES; exiting before LLM extraction calls.")
        raise SystemExit(1)


# -----------------------------
# Logging helpers
# -----------------------------

class Tee(TextIO):
    def __init__(self, *streams: TextIO):
        self.streams = streams

    def write(self, data: str) -> int:
        for s in self.streams:
            s.write(data)
            s.flush()
        return len(data)

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


# -----------------------------
# Helpers
# -----------------------------

def _safe_text(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and np.isnan(x):
        return ""
    return str(x)


def _word_count(s: str) -> int:
    return len([w for w in re.split(r"\s+", s.strip()) if w])


def _is_missing_text(x: Any) -> bool:
    return _safe_text(x).strip() == ""


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


def _extract_json_from_text(text: str) -> Dict[str, Any]:
    text = (text or "").strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    fenced = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
    fenced = re.sub(r"\s*```$", "", fenced.strip())
    try:
        return json.loads(fenced)
    except Exception:
        pass

    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        raise ValueError("No JSON object found in model output.")
    return json.loads(m.group(0))


def _coerce_candidate_concepts(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(v).strip() for v in x if str(v).strip()]
    if isinstance(x, str):
        parts = re.split(r"[,;|]", x)
        return [p.strip() for p in parts if p.strip()]
    return []


def _coerce_float_or_none(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        v = float(x)
    except Exception:
        return None
    if np.isnan(v):
        return None
    return v


def _normalize_phrase_item_schema(item: Dict[str, Any], report_mode: str) -> Dict[str, Any]:
    """Fill optional fields so older extraction outputs remain downstream-compatible."""
    item = dict(item)
    quote = str(item.get("quote", "")).strip()
    concept = str(item.get("concept", "")).strip()
    item.setdefault("normalized_phrase", re.sub(r"\s+", " ", quote.lower()).strip())
    item["candidate_concepts"] = _coerce_candidate_concepts(item.get("candidate_concepts"))
    if not item["candidate_concepts"] and concept:
        item["candidate_concepts"] = [concept]
    item.setdefault("section", "unknown")
    item.setdefault("directness", "direct" if report_mode == "path" else "unknown")
    item.setdefault("directly_asserts_tumor", None)
    item.setdefault("imaging_pattern_only", None)
    item.setdefault("biological_ambiguity", "unknown")
    item.setdefault("mapping_confidence", None)
    item["mapping_confidence"] = _coerce_float_or_none(item.get("mapping_confidence"))
    item.setdefault("quantitative_attributes", {})
    if not isinstance(item["quantitative_attributes"], dict):
        item["quantitative_attributes"] = {}
    return item

def _sanitize_phrase_item_for_validation(
    item: Dict[str, Any],
    report_text: str,
    item_name: str,
    report_mode: str,
) -> Tuple[Optional[Dict[str, Any]], List[str], int]:
    """Normalize, repair, or drop one phrase item before hard validation.

    A single invalid model phrase should not invalidate the entire case. If the
    quote is a near miss, repair it to an exact report substring. If it cannot be
    repaired, drop only that phrase item and preserve a warning.
    """
    warnings: List[str] = []
    repaired = 0

    if not isinstance(item, dict):
        return None, [f"{item_name} dropped: item is not an object"], repaired

    item = _normalize_phrase_item_schema(item, report_mode=report_mode)

    # Normalize controlled vocab fields conservatively.
    item["polarity"] = str(item.get("polarity", "affirmed")).strip().lower()
    item["certainty"] = str(item.get("certainty", "certain")).strip().lower()
    if item["polarity"] not in {"affirmed", "negated", "uncertain"}:
        warnings.append(f"{item_name}.polarity coerced from {item.get('polarity')!r} to 'uncertain'")
        item["polarity"] = "uncertain"
    if item["certainty"] not in {"certain", "uncertain"}:
        warnings.append(f"{item_name}.certainty coerced from {item.get('certainty')!r} to 'uncertain'")
        item["certainty"] = "uncertain"

    concept = str(item.get("concept", "")).strip()
    if not concept:
        return None, [f"{item_name} dropped: missing/empty concept"], repaired

    quote = str(item.get("quote", "")).strip()
    if not quote:
        return None, [f"{item_name} dropped: missing/empty quote"], repaired

    if _word_count(quote) > 35 or not _quote_present_in_report(quote, report_text):
        repaired_quote, similarity, overlap = _repair_quote_to_exact_report_span(quote, report_text)
        if repaired_quote and _word_count(repaired_quote) <= 35:
            item["original_quote_before_repair"] = quote
            item["quote"] = repaired_quote
            item["quote_repair_status"] = "repaired_to_exact_report_substring"
            item["quote_repair_similarity"] = similarity
            item["quote_repair_token_overlap"] = overlap
            repaired = 1
            warnings.append(
                f"{item_name}.quote repaired to exact report substring "
                f"(similarity={similarity:.3f}, token_overlap={overlap:.3f})"
            )
        else:
            return None, [
                f"{item_name} dropped: quote not found and could not be repaired; "
                f"best_similarity={similarity:.3f}, best_token_overlap={overlap:.3f}; "
                f"quote={quote[:160]!r}"
            ], repaired

    return item, warnings, repaired


def _sanitize_extraction_obj_for_validation(
    obj: Dict[str, Any],
    report_text: str,
    report_mode: str,
) -> Dict[str, Any]:
    """Repair/drop invalid phrase-level outputs before strict object validation."""
    obj = dict(obj)
    validation_warnings: List[str] = []
    n_repaired = 0
    n_dropped = 0

    for list_key in ["seed_aligned_phrases", "denovo_candidate_phrases"]:
        raw_items = obj.get(list_key, [])
        if not isinstance(raw_items, list):
            validation_warnings.append(f"{list_key} was not a list; replacing with empty list")
            obj[list_key] = []
            continue

        cleaned: List[Dict[str, Any]] = []
        for i, item in enumerate(raw_items):
            cleaned_item, warnings, repaired = _sanitize_phrase_item_for_validation(
                item=item,
                report_text=report_text,
                item_name=f"{list_key}[{i}]",
                report_mode=report_mode,
            )
            validation_warnings.extend(warnings)
            n_repaired += int(repaired)
            if cleaned_item is None:
                n_dropped += 1
            else:
                cleaned.append(cleaned_item)
        obj[list_key] = cleaned

    obj["validation_warnings"] = validation_warnings
    obj["n_validation_warnings"] = len(validation_warnings)
    obj["n_repaired_phrase_quotes"] = n_repaired
    obj["n_dropped_phrase_items"] = n_dropped
    return obj

# -----------------------------
# Validation
# -----------------------------

def _validate_phrase_item(item: Dict[str, Any], report_text: str, item_name: str) -> Tuple[bool, str]:
    required = ["quote", "concept", "polarity", "certainty"]
    for k in required:
        if k not in item:
            return False, f"{item_name} missing key: {k}"

    quote = item["quote"]
    if not isinstance(quote, str) or not quote.strip():
        return False, f"{item_name}.quote must be a non-empty string"
    if _word_count(quote) > 35:
        return False, f"{item_name}.quote exceeds 35 words"
    if not _quote_present_in_report(quote, report_text):
        return False, f"{item_name}.quote is not found in report text"

    concept = item["concept"]
    if not isinstance(concept, str) or not concept.strip():
        return False, f"{item_name}.concept must be a non-empty string"

    polarity = str(item["polarity"]).strip().lower()
    if polarity not in {"affirmed", "negated", "uncertain"}:
        return False, f"{item_name}.polarity must be affirmed/negated/uncertain"

    certainty = str(item["certainty"]).strip().lower()
    if certainty not in {"certain", "uncertain"}:
        return False, f"{item_name}.certainty must be certain/uncertain"

    if "candidate_concepts" in item and not isinstance(item["candidate_concepts"], list):
        return False, f"{item_name}.candidate_concepts must be a list when supplied"

    return True, "ok"


def _validate_quantitative_attributes(q: Dict[str, Any]) -> Tuple[bool, str]:
    required = [
        "extent_cm",
        "largest_focus_cm",
        "margin_distance_mm",
        "lvi_present",
        "dcis_burden",
        "nme_present",
        "satellite_lesions_present",
        "multifocal_present",
        "multicentric_present",
        "residual_disease_minimal",
        "single_localized_residual",
        "diffuse_scattered_residual",
    ]
    for k in required:
        if k not in q:
            return False, f"quantitative_attributes missing key: {k}"
    return True, "ok"


def _normalize_extraction_obj(obj: Dict[str, Any], report_mode: str) -> Dict[str, Any]:
    obj = dict(obj)
    for list_key in ["seed_aligned_phrases", "denovo_candidate_phrases"]:
        items = obj.get(list_key, [])
        if isinstance(items, list):
            obj[list_key] = [
                _normalize_phrase_item_schema(item, report_mode) if isinstance(item, dict) else item
                for item in items
            ]
    return obj


def _validate_extraction_obj(
    obj: Dict[str, Any],
    expected_case_id: str,
    report_text: str,
    report_mode: str,
) -> Tuple[bool, str]:
    required = [
        "case_id",
        "report_mode",
        "selected_report_field",
        "selected_report_missing",
        "seed_aligned_phrases",
        "denovo_candidate_phrases",
        "quantitative_attributes",
        "report_level_summary",
    ]
    for k in required:
        if k not in obj:
            return False, f"Missing key: {k}"

    if str(obj["case_id"]) != str(expected_case_id):
        return False, f"case_id mismatch: got {obj['case_id']} expected {expected_case_id}"

    if str(obj["report_mode"]).strip().lower() != report_mode:
        return False, f"report_mode mismatch: got {obj['report_mode']} expected {report_mode}"

    if bool(obj["selected_report_missing"]):
        return False, "selected_report_missing must be false for a non-missing report extraction"

    for list_key in ["seed_aligned_phrases", "denovo_candidate_phrases"]:
        if not isinstance(obj[list_key], list):
            return False, f"{list_key} must be a list"
        for i, item in enumerate(obj[list_key]):
            if not isinstance(item, dict):
                return False, f"{list_key}[{i}] must be an object"
            ok, msg = _validate_phrase_item(item, report_text, f"{list_key}[{i}]")
            if not ok:
                return False, msg

    qa = obj["quantitative_attributes"]
    if not isinstance(qa, dict):
        return False, "quantitative_attributes must be an object"
    ok, msg = _validate_quantitative_attributes(qa)
    if not ok:
        return False, msg

    rls = obj["report_level_summary"]
    if not isinstance(rls, dict):
        return False, "report_level_summary must be an object"

    return True, "ok"


# -----------------------------
# Prompt construction
# -----------------------------

# OUTPUT BUDGET
# - Extract at most {MAX_SEED_PHRASES} seed_aligned_phrases.
# - Extract at most {MAX_DENOVO_PHRASES} denovo_candidate_phrases.
# - Prefer the most biologically informative, quote-grounded phrases.
# - Do not extract every possible mention.
# - Avoid duplicate or near-duplicate quotes.
# - Keep each quote <= 25 words when possible and never >35 words.
# - Keep report_level_summary concise.

def _mri_permissive_instructions() -> str:
    return """
MRI-SPECIFIC PERMISSIVE EXTRACTION INSTRUCTIONS

For MRI reports, be intentionally permissive. Capture candidate phrases even if
they are ambiguous, hedged, nonspecific, or imaging-pattern-only, provided they
could plausibly help later distinguish scattered/dispersed residual disease from
localized/compact/minimal disease. Include:

- definite tumor descriptors
- possible residual disease
- ambiguous or nonspecific enhancement
- benign/favored benign enhancement when it could create false-positive signal
- non-mass enhancement / NME
- distribution pattern: segmental, regional, linear, ductal, diffuse, clumped
- multiplicity/focality: multiple foci, scattered foci, single focus, focal
- extent/span and comparison-to-prior measurements
- continuity vs fragmentation: patchy, discontinuous, scattered, separated
- treatment response: decreased, resolved, persistent, residual, near complete response
- absence of a discrete mass or absence of residual enhancement
- uncertainty/hedging words such as possible, may represent, indeterminate, favored

A phrase may be imaging_pattern_only=true if it describes enhancement without
proving viable tumor. Do not over-interpret it as residual carcinoma; preserve
that ambiguity in directness and biological_ambiguity fields.
""".strip()


def build_user_prompt(case: Case, report_mode: str) -> str:
    selected_text = _selected_report_text(case, report_mode)
    selected_field = _selected_report_field(report_mode)
    selected_label = _selected_report_label(report_mode)
    modality_extra = _mri_permissive_instructions() if report_mode == "mri" else ""

    return f"""
TASK
You are performing lexical feature discovery for breast tumor dispersiveness from a single clinical report.

Use ONLY the selected report copied in the CASE block below.
The CASE block provides selected_report_type, selected_report_field, index_side, case_id, and the report text.

Do NOT use any information outside the selected report.

Your job is to extract:
1) seed_aligned_phrases:
   exact quoted phrases aligned to the seed guidance or shared ontology concepts
2) denovo_candidate_phrases:
   exact quoted phrases that may indicate dispersiveness, localization, minimal residual disease,
   broad extent, multifocality, discontinuity, infiltrative spread, satellites, LVI, DCIS burden,
   non-mass enhancement, ambiguous enhancement, treatment response, or related concepts even if
   not explicitly listed in the seed guidance
3) quantitative_attributes:
   structured quantitative or binary attributes if present in the report
4) report_level_summary:
   concise high-level summary of whether the report suggests scattered/discontinuous/extensive disease
   versus compact/localized/minimal residual disease

STRICT RULES
- Return valid JSON only.
- Every extracted phrase quote must be copied exactly from the selected report.
- If a concept is negated in the report, mark polarity="negated".
- If a concept is uncertain/suspected/possible, mark certainty="uncertain" and polarity="uncertain" if appropriate.
- If a structured attribute is not stated, use null.
- Do not invent measurements.
- For "concept", use short human-readable labels.
- For "candidate_concepts", use zero or more concept names from the shared ontology guidance.
- Prefer the shortest exact quoted span that still preserves the finding.
- Each quote must be <= 30 words unless a slightly longer span is needed for quote grounding.
- Good examples: "irregular mass with spiculated margins", "heterogeneous internal enhancement", "multiple enhancing foci".

SEED GUIDANCE
{SEED_GUIDANCE}

OUTPUT JSON SCHEMA (RETURN ONLY THIS OBJECT)
{{
  "case_id": "<case_id copied exactly from CASE block>",
  "report_mode": "<selected_report_type copied exactly from CASE block>",
  "selected_report_field": "<selected_report_field copied exactly from CASE block>",
  "selected_report_missing": false,
  "seed_aligned_phrases": [
    {{
      "quote": "<exact quote from report>",
      "normalized_phrase": "<lowercase normalized paraphrase of the quote>",
      "concept": "<short concept label>",
      "candidate_concepts": ["<shared ontology concept name>", "<optional second concept>"],
      "polarity": "<affirmed|negated|uncertain>",
      "certainty": "<certain|uncertain>",
      "laterality": "<left|right|bilateral|unknown>",
      "span_type": "<finding|measurement|distribution|response_pattern|pathology_feature|ambiguity>",
      "section": "<report section if identifiable, else unknown>",
      "directness": "<direct_tumor|imaging_pattern_only|treatment_effect|benign_or_nonspecific|unknown>",
      "directly_asserts_tumor": <true|false|null>,
      "imaging_pattern_only": <true|false|null>,
      "biological_ambiguity": "<low|moderate|high|unknown>",
      "mapping_confidence": <float 0-1 or null>,
      "quantitative_attributes": {{}}
    }}
  ],
  "denovo_candidate_phrases": [
    {{
      "quote": "<exact quote from report>",
      "normalized_phrase": "<lowercase normalized paraphrase of the quote>",
      "concept": "<short concept label>",
      "candidate_concepts": ["<shared ontology concept name>", "<optional second concept>"],
      "polarity": "<affirmed|negated|uncertain>",
      "certainty": "<certain|uncertain>",
      "laterality": "<left|right|bilateral|unknown>",
      "span_type": "<finding|measurement|distribution|response_pattern|pathology_feature|ambiguity>",
      "section": "<report section if identifiable, else unknown>",
      "directness": "<direct_tumor|imaging_pattern_only|treatment_effect|benign_or_nonspecific|unknown>",
      "directly_asserts_tumor": <true|false|null>,
      "imaging_pattern_only": <true|false|null>,
      "biological_ambiguity": "<low|moderate|high|unknown>",
      "mapping_confidence": <float 0-1 or null>,
      "quantitative_attributes": {{}}
    }}
  ],
  "quantitative_attributes": {{
    "extent_cm": <float or null>,
    "largest_focus_cm": <float or null>,
    "margin_distance_mm": <float or null>,
    "lvi_present": <0 or 1 or null>,
    "dcis_burden": "<none|minimal|focal|limited|intermediate|extensive|unknown|null>",
    "nme_present": <0 or 1 or null>,
    "satellite_lesions_present": <0 or 1 or null>,
    "multifocal_present": <0 or 1 or null>,
    "multicentric_present": <0 or 1 or null>,
    "residual_disease_minimal": <0 or 1 or null>,
    "single_localized_residual": <0 or 1 or null>,
    "diffuse_scattered_residual": <0 or 1 or null>
  }},
  "report_level_summary": {{
    "distribution_pattern": "<scattered|multifocal|multicentric|diffuse|discontinuous|localized|minimal_residual|mixed|unknown>",
    "distribution_evidence_quote": "<exact short quote or empty string>",
    "localization_vs_scatter_note": "<1-2 concise sentences grounded in the report>"
  }}
}}

CASE
case_id: {case.case_id}
selected_report_type: {report_mode}
selected_report_field: {selected_field}
selected_report_description: {selected_label}
index_side: {case.index_side}
selected_report_text:
{selected_text}

REMINDER: Output JSON only.
""".strip()


# -----------------------------
# Stanford AI Sandbox chat-completions calls
# -----------------------------

print("[INIT] Stanford AI Sandbox request client configured.")
print(f"[INIT] URL={URL}")
print(f"[INIT] DEPLOYMENT={DEPLOYMENT} API_VERSION={API_VERSION}")


def _post_chat_completion(messages: Sequence[Dict[str, str]], max_completion_tokens: int = MAX_TOKENS) -> Dict[str, Any]:
    payload = {
        "model": DEPLOYMENT,
        "messages": list(messages),
        # GPT-5-style models account for both visible output and reasoning tokens here.
        "max_completion_tokens": int(max_completion_tokens),
    }
    if REASONING_EFFORT:
        payload["reasoning_effort"] = REASONING_EFFORT

    with GLOBAL_API_SEMAPHORE:
        response = requests.post(URL, headers=HEADERS, json=payload, timeout=180)
    if not response.ok:
        print("[API_ERROR] API returned an error:")
        print(f"[API_ERROR] Status: {response.status_code}")
        try:
            print(json.dumps(response.json(), indent=2))
        except Exception:
            print(response.text)
        response.raise_for_status()
    return response.json()


def preflight_check() -> None:
    print("[PREFLIGHT] Testing Stanford AI Sandbox connectivity with a small request...")
    t0 = time.time()
    data = _post_chat_completion(
        messages=[{"role": "user", "content": "Reply with exactly: OK"}],
        max_completion_tokens=32,
    )
    dt = time.time() - t0
    usage = data.get("usage", {}) or {}
    cost_info = estimate_cost_from_usage(usage)
    update_cost_tracker(cost_info)
    choice = data["choices"][0]
    reply = choice["message"].get("content") or ""
    print(
        f"[PREFLIGHT] Success in {dt:.2f}s. "
        f"Model={data.get('model', DEPLOYMENT)} Reply={reply!r} "
        f"prompt_tokens={cost_info['prompt_tokens']} cached_tokens={cost_info['cached_tokens']} "
        f"completion_tokens={cost_info['completion_tokens']} estimated_cost=${cost_info['estimated_cost_usd']:.8f}"
    )


def call_securegpt_chat(prompt: str) -> str:
    messages = build_chat_messages(prompt)
    prompt_token_estimate = estimate_prompt_tokens_from_messages(messages)
    print(
        f"[API] Sending request... prompt_chars={len(prompt)} "
        f"estimated_prompt_tokens={prompt_token_estimate} max_completion_tokens={MAX_TOKENS}"
    )
    t0 = time.time()
    data = _post_chat_completion(messages=messages, max_completion_tokens=MAX_TOKENS)
    dt = time.time() - t0

    usage = data.get("usage", {}) or {}
    cost_info = estimate_cost_from_usage(usage)
    update_cost_tracker(cost_info)

    choice = data["choices"][0]
    content = choice["message"].get("content") or ""
    finish_reason = choice.get("finish_reason")
    print(
        f"[API] Response in {dt:.2f}s. response_chars={len(content)} "
        f"model={data.get('model', DEPLOYMENT)} finish_reason={finish_reason} "
        f"prompt_tokens={cost_info['prompt_tokens']} cached_tokens={cost_info['cached_tokens']} "
        f"completion_tokens={cost_info['completion_tokens']} reasoning_tokens={cost_info['reasoning_tokens']} "
        f"estimated_cost=${cost_info['estimated_cost_usd']:.8f} "
        f"cache_savings=${cost_info['estimated_cache_savings_usd']:.8f}"
    )
    if not content:
        print(
            "[WARNING] Empty visible output. "
            f"finish_reason={finish_reason}. Try increasing MAX_COMPLETION_TOKENS / max_completion_tokens."
        )
    return content

def extract_case_features(test_case: Case, report_mode: str) -> Dict[str, Any]:
    user_prompt = build_user_prompt(test_case, report_mode)
    selected_text = _selected_report_text(test_case, report_mode)
    last_err: Optional[str] = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(
                f"[EXTRACT] case_id={test_case.case_id} attempt={attempt}/{MAX_RETRIES} "
                f"report_mode={report_mode}"
            )
            raw = call_securegpt_chat(user_prompt)
            obj = _extract_json_from_text(raw)
            obj = _normalize_extraction_obj(obj, report_mode=report_mode)
            obj = _sanitize_extraction_obj_for_validation(
                obj=obj,
                report_text=selected_text,
                report_mode=report_mode,
            )

            warnings = obj.get("validation_warnings", []) or []
            if warnings:
                print(
                    f"[VALIDATION] case_id={test_case.case_id} "
                    f"n_warnings={len(warnings)} "
                    f"n_repaired={obj.get('n_repaired_phrase_quotes', 0)} "
                    f"n_dropped={obj.get('n_dropped_phrase_items', 0)}"
                )
                for w in warnings[:8]:
                    print(f"[VALIDATION] {w}")
                if len(warnings) > 8:
                    print(f"[VALIDATION] ... {len(warnings) - 8} additional warnings omitted from console")

            ok, msg = _validate_extraction_obj(
                obj,
                expected_case_id=test_case.case_id,
                report_text=selected_text,
                report_mode=report_mode,
            )
            if not ok:
                raise ValueError(f"Validation failed after sanitization: {msg}. Raw head: {raw[:500]}")

            print(f"[EXTRACT] case_id={test_case.case_id} VALID JSON received.")
            return obj

        except Exception as e:
            last_err = str(e)
            print(f"[RETRY] case_id={test_case.case_id} attempt={attempt}/{MAX_RETRIES} error={last_err}")
            sleep_s = BACKOFF_BASE_S ** (attempt - 1)
            print(f"[RETRY] Sleeping for {sleep_s:.2f}s before retry...")
            time.sleep(sleep_s)

    raise RuntimeError(
        f"Failed to get valid extraction for case_id={test_case.case_id}. Last error: {last_err}"
    )

# -----------------------------
# Main pipeline helpers
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


def summarize_run(extractions: List[Dict[str, Any]], report_mode: str, split_id: Optional[str] = None) -> str:
    total = len(extractions)
    missing = sum(int(bool(x.get("selected_report_missing", 0))) for x in extractions)
    used = total - missing
    n_seed = sum(len(x.get("seed_aligned_phrases", [])) for x in extractions)
    n_denovo = sum(len(x.get("denovo_candidate_phrases", [])) for x in extractions)

    lines = []
    lines.append("=== Lexical Feature Discovery Run Summary ===")
    lines.append(f"report_mode = {report_mode}")
    if split_id:
        lines.append(f"outer_split_id = {split_id}")
    lines.append(f"N_total_rows = {total}")
    lines.append(f"N_selected_report_missing = {missing}")
    lines.append(f"N_rows_extracted = {used}")
    lines.append(f"N_seed_aligned_phrases_total = {n_seed}")
    lines.append(f"N_denovo_candidate_phrases_total = {n_denovo}")
    return "\n".join(lines)


def _extract_single_subset_record(
    df: pd.DataFrame,
    row_index: int,
    report_mode: str,
    split_id: Optional[str],
    split_role: Optional[str],
    sleep_between_calls_s: float,
    position: int,
    total: int,
) -> Dict[str, Any]:
    test_case = make_case_from_row(df, row_index)
    selected_text = _selected_report_text(test_case, report_mode)
    selected_field = _selected_report_field(report_mode)

    print('-' * 80)
    print(
        f"[CASE] {position}/{total} case_id={test_case.case_id} "
        f"row_index={row_index} report_mode={report_mode}"
    )
    print(
        f"[CASE] selected_report_field={selected_field} "
        f"selected_chars={len(selected_text)} split_id={split_id or 'NA'} split_role={split_role or 'NA'}"
    )

    if _is_missing_text(selected_text):
        rec = make_missing_extraction_record(
            test_case=test_case,
            row_index=row_index,
            report_mode=report_mode,
            split_id=split_id,
            split_role=split_role,
        )
        print('[CASE] Selected report missing; wrote placeholder row.')
        return rec

    obj = extract_case_features(test_case, report_mode)

    rec = {
        'case_id': test_case.case_id,
        'row_index': row_index,
        'report_mode': report_mode,
        'selected_report_field': obj['selected_report_field'],
        'selected_report_text': selected_text,
        'selected_report_missing': 0,
        'selected_report_missing_reason': '',
        'has_preop_mri': int(not _is_missing_text(test_case.preop_mri)),
        'has_path_report': int(not _is_missing_text(test_case.path_report)),
        'dispersion_true': test_case.dispersion_true,
        'dispersion_true_high_low': _true_dispersion_high_low(test_case.dispersion_true),
        'relapse_true': test_case.relapse_true,
        'outer_split_id': split_id or '',
        'outer_split_role': split_role or '',
        "validation_warnings": [],
        "n_validation_warnings": 0,
        "n_repaired_phrase_quotes": 0,
        "n_dropped_phrase_items": 0,
        'seed_aligned_phrases': obj['seed_aligned_phrases'],
        'denovo_candidate_phrases': obj['denovo_candidate_phrases'],
        'quantitative_attributes': obj['quantitative_attributes'],
        'report_level_summary': obj['report_level_summary'],
    }

    print(
        '[CASE] Extraction OK: '
        f"n_seed={len(rec['seed_aligned_phrases'])} "
        f"n_denovo={len(rec['denovo_candidate_phrases'])}"
    )
    time.sleep(sleep_between_calls_s)
    return rec


def _resolve_default_api_workers(requested_workers: Optional[int]) -> int:
    if requested_workers is not None:
        return max(1, int(requested_workers))
    cpu_count = os.cpu_count() or 1
    if cpu_count >= 8:
        return 2
    return 1

def _safe_filename_component(s: Any, max_len: int = 100) -> str:
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s or "NA")).strip("_")
    return (s or "NA")[:max_len]


def _checkpoint_root_dir(
    checkpoint_dir: Optional[str],
    report_mode: str,
    split_id: Optional[str],
    split_role: Optional[str],
) -> Optional[str]:
    if not checkpoint_dir:
        return None
    root = os.path.join(
        checkpoint_dir,
        "_case_checkpoints",
        _safe_filename_component(report_mode),
        _safe_filename_component(split_id or "standalone"),
        _safe_filename_component(split_role or "subset"),
    )
    os.makedirs(root, exist_ok=True)
    return root


def _case_checkpoint_path(checkpoint_root: str, row_index: int, case_id: str, report_mode: str) -> str:
    fname = (
        f"row_{int(row_index):06d}__"
        f"case_{_safe_filename_component(case_id)}__"
        f"mode_{_safe_filename_component(report_mode)}.json"
    )
    return os.path.join(checkpoint_root, fname)


def _write_json_atomic(obj: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, path)


def _load_json_record(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else None
    except Exception as e:
        print(f"[CHECKPOINT] Could not load checkpoint {path}: {e}")
        return None


def _record_matches_request(
    rec: Dict[str, Any],
    row_index: int,
    case_id: str,
    report_mode: str,
    split_id: Optional[str],
    split_role: Optional[str],
) -> bool:
    return (
        str(rec.get("case_id")) == str(case_id)
        and int(rec.get("row_index", -999999)) == int(row_index)
        and str(rec.get("report_mode")) == str(report_mode)
        and str(rec.get("outer_split_id", "")) == str(split_id or "")
        and str(rec.get("outer_split_role", "")) == str(split_role or "")
    )


def _extract_single_subset_record_with_checkpoint(
    df: pd.DataFrame,
    row_index: int,
    report_mode: str,
    split_id: Optional[str],
    split_role: Optional[str],
    sleep_between_calls_s: float,
    position: int,
    total: int,
    checkpoint_root: Optional[str],
    resume: bool,
    force_reextract: bool,
) -> Dict[str, Any]:
    test_case = make_case_from_row(df, row_index)
    ckpt_path = (
        _case_checkpoint_path(checkpoint_root, row_index, test_case.case_id, report_mode)
        if checkpoint_root else None
    )

    if ckpt_path and resume and not force_reextract and os.path.exists(ckpt_path):
        rec = _load_json_record(ckpt_path)
        if rec is not None and _record_matches_request(rec, row_index, test_case.case_id, report_mode, split_id, split_role):
            print(
                f"[CHECKPOINT] Reusing cached extraction: case_id={test_case.case_id} "
                f"row_index={row_index} report_mode={report_mode} path={ckpt_path}"
            )
            rec["checkpoint_status"] = "loaded"
            return rec
        print(f"[CHECKPOINT] Existing checkpoint did not match request; re-extracting: {ckpt_path}")

    try:
        rec = _extract_single_subset_record(
            df=df,
            row_index=row_index,
            report_mode=report_mode,
            split_id=split_id,
            split_role=split_role,
            sleep_between_calls_s=sleep_between_calls_s,
            position=position,
            total=total,
        )
        rec["checkpoint_status"] = "fresh"
        rec["checkpoint_written_at"] = datetime.now().isoformat(timespec="seconds")
        if ckpt_path:
            _write_json_atomic(rec, ckpt_path)
            print(f"[CHECKPOINT] Wrote case checkpoint: {ckpt_path}")
        return rec
    except Exception:
        print(
            f"[CHECKPOINT] Extraction failed before checkpoint could be written for "
            f"case_id={test_case.case_id} row_index={row_index} report_mode={report_mode}"
        )
        print(traceback.format_exc())
        raise

def extract_subset_records(
    df: pd.DataFrame,
    row_indices: Sequence[int],
    report_mode: str,
    split_id: Optional[str] = None,
    split_role: Optional[str] = None,
    sleep_between_calls_s: float = RATE_LIMIT_SLEEP_S,
    max_workers: int = 1,
    checkpoint_dir: Optional[str] = None,
    resume: bool = True,
    force_reextract: bool = False,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    row_indices = [int(x) for x in row_indices]
    max_workers = max(1, int(max_workers))
    checkpoint_root = _checkpoint_root_dir(
        checkpoint_dir=checkpoint_dir,
        report_mode=report_mode,
        split_id=split_id,
        split_role=split_role,
    )

    print(
        f"[RUN] Beginning extraction over subset. "
        f"report_mode={report_mode} split_id={split_id or 'NA'} split_role={split_role or 'NA'} "
        f"n_rows={len(row_indices)} max_api_workers={max_workers} "
        f"global_api_cap={get_global_api_concurrency()} "
        f"resume={resume} force_reextract={force_reextract} "
        f"checkpoint_root={checkpoint_root or 'disabled'}"
    )

    if max_workers == 1 or len(row_indices) <= 1:
        for n, idx in enumerate(row_indices, 1):
            rec = _extract_single_subset_record_with_checkpoint(
                df=df,
                row_index=idx,
                report_mode=report_mode,
                split_id=split_id,
                split_role=split_role,
                sleep_between_calls_s=sleep_between_calls_s,
                position=n,
                total=len(row_indices),
                checkpoint_root=checkpoint_root,
                resume=resume,
                force_reextract=force_reextract,
            )
            records.append(rec)
        return records

    print(
        f"[RUN] Parallel API extraction enabled for report_mode={report_mode} "
        f"with max_workers={max_workers}. Logs may interleave across cases."
    )
    indexed_records: Dict[int, Dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_position = {
            executor.submit(
                _extract_single_subset_record_with_checkpoint,
                df,
                idx,
                report_mode,
                split_id,
                split_role,
                sleep_between_calls_s,
                position,
                len(row_indices),
                checkpoint_root,
                resume,
                force_reextract,
            ): position - 1
            for position, idx in enumerate(row_indices, 1)
        }
        for future in as_completed(future_to_position):
            pos = future_to_position[future]
            indexed_records[pos] = future.result()

    records = [indexed_records[i] for i in sorted(indexed_records)]
    return records


def write_extractions(
    extractions: List[Dict[str, Any]],
    out_dir: str,
    report_mode: str,
    filename_prefix: Optional[str] = None,
    split_id: Optional[str] = None,
) -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    prefix = f"{filename_prefix}_" if filename_prefix else ""

    extractions_csv = os.path.join(out_dir, f"{prefix}case_phrase_extractions_{report_mode}.csv")
    extractions_jsonl = os.path.join(out_dir, f"{prefix}case_phrase_extractions_{report_mode}.jsonl")
    summary_txt = os.path.join(out_dir, f"{prefix}run_summary_{report_mode}.txt")

    # with open(extractions_jsonl, "w", encoding="utf-8") as f_jsonl:
    #     for rec in extractions:
    #         f_jsonl.write(json.dumps(rec, ensure_ascii=False) + "\n")

    out_df = pd.DataFrame(extractions)
    if len(out_df) == 0:
        out_df = pd.DataFrame(columns=[
            "case_id",
            "row_index",
            "report_mode",
            "selected_report_field",
            "selected_report_text",
            "selected_report_missing",
            "selected_report_missing_reason",
            "has_preop_mri",
            "has_path_report",
            "dispersion_true",
            "dispersion_true_high_low",
            "relapse_true",
            "outer_split_id",
            "outer_split_role",
            "seed_aligned_phrases",
            "denovo_candidate_phrases",
            "quantitative_attributes",
            "report_level_summary",
        ])

    for col in [
        "seed_aligned_phrases",
        "denovo_candidate_phrases",
        "quantitative_attributes",
        "report_level_summary",
    ]:
        if col in out_df.columns:
            out_df[col] = out_df[col].apply(lambda x: json.dumps(x, ensure_ascii=False))

    tmp_csv = f"{extractions_csv}.tmp.{os.getpid()}"
    out_df.to_csv(tmp_csv, index=False)
    os.replace(tmp_csv, extractions_csv)

    summary = summarize_run(extractions, report_mode, split_id=split_id)
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write(summary)

    tmp_jsonl = f"{extractions_jsonl}.tmp.{os.getpid()}"
    with open(tmp_jsonl, "w", encoding="utf-8") as f_jsonl:
        for rec in extractions:
            f_jsonl.write(json.dumps(rec, ensure_ascii=False) + "\n")
    os.replace(tmp_jsonl, extractions_jsonl)

    print(f"[SAVE] Wrote JSONL: {extractions_jsonl}")
    print(f"[SAVE] Wrote CSV:   {extractions_csv}")
    print(f"[SAVE] Wrote summary: {summary_txt}")

    return {
        "csv": extractions_csv,
        "jsonl": extractions_jsonl,
        "summary": summary_txt,
    }


# -----------------------------
# Main pipeline
# -----------------------------

def run_pipeline(
    csv_path: str,
    out_dir: str,
    report_mode: str,
    resume: bool = True,
    force_reextract: bool = False,
    row_indices: Optional[Sequence[int]] = None,
    split_id: Optional[str] = None,
    split_role: Optional[str] = None,
    filename_prefix: Optional[str] = None,
    max_api_workers: Optional[int] = None,
    assume_yes: bool = False,
) -> Dict[str, str]:
    print("=" * 80)
    print("[START] SecureGPT lexical feature discovery pipeline")
    print(f"[START] CSV_PATH={csv_path}")
    print(f"[START] OUT_DIR={out_dir}")
    print(f"[START] REPORT_MODE={report_mode}")
    if split_id:
        print(f"[START] SPLIT_ID={split_id}")
    if split_role:
        print(f"[START] SPLIT_ROLE={split_role}")
    print("=" * 80)

    os.makedirs(out_dir, exist_ok=True)
    max_api_workers = _resolve_default_api_workers(max_api_workers)
    configure_global_api_concurrency(max_api_workers)
    print(f"[START] MAX_API_WORKERS={max_api_workers}")
    df = load_cases(csv_path)

    if row_indices is None:
        row_indices = list(range(len(df)))
    else:
        row_indices = [int(x) for x in row_indices]

    prompt_counts: List[int] = []
    prompt_modes: List[str] = []
    for idx in row_indices:
        case = make_case_from_row(df, int(idx))
        if _is_missing_text(_selected_report_text(case, report_mode)):
            continue
        prompt_counts.append(estimate_prompt_tokens_for_case(case, report_mode))
        prompt_modes.append(report_mode)
    estimate = summarize_apriori_cost_estimate(prompt_counts, prompt_modes, max_completion_tokens=MAX_TOKENS)
    print_apriori_cost_estimate_report(estimate, label=f"standalone {report_mode} extraction")
    confirm_cost_estimate_or_exit(estimate, assume_yes=assume_yes)

    preflight_check()

    print(f"[RUN] Total cases to iterate = {len(row_indices)}")
    print(f"[RUN] Writing outputs to: {out_dir}")

    t_run0 = time.time()
    extractions = extract_subset_records(
        df=df,
        row_indices=row_indices,
        report_mode=report_mode,
        split_id=split_id,
        split_role=split_role,
        sleep_between_calls_s=RATE_LIMIT_SLEEP_S,
        max_workers=max_api_workers,
        checkpoint_dir=out_dir,
        resume=resume,
        force_reextract=force_reextract,
    )
    paths = write_extractions(
        extractions=extractions,
        out_dir=out_dir,
        report_mode=report_mode,
        filename_prefix=filename_prefix,
        split_id=split_id,
    )

    dt_run = time.time() - t_run0
    print("=" * 80)
    print(f"[DONE] Extraction complete in {dt_run/60:.2f} minutes.")
    print("=" * 80)
    print("\n" + summarize_run(extractions, report_mode, split_id=split_id))
    print_cumulative_report()
    write_cost_tracker_json(out_dir)
    print("[END] All done.")
    return paths


def _load_row_indices_json(path: Optional[str]) -> Optional[List[int]]:
    if path is None:
        return None
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, list):
        raise ValueError("--row-indices-json must contain a JSON list of integer row indices.")
    return [int(x) for x in obj]


# -----------------------------
# HTML report rendering
# -----------------------------

REPORT_HTML_STYLES = """
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  line-height: 1.5;
  color: #1f2933;
  background: #f7f9fc;
  margin: 0;
  padding: 0;
}
.page {
  max-width: 1200px;
  margin: 0 auto;
  padding: 2rem 1.5rem 3rem;
}
h1 { font-size: 1.9rem; margin: 0 0 0.75rem; }
h2 {
  font-size: 1.35rem;
  margin: 2rem 0 0.75rem;
  padding-bottom: 0.35rem;
  border-bottom: 2px solid #d9e2ec;
}
h3 { font-size: 1.05rem; margin: 0 0 0.35rem; }
.lead { color: #52606d; margin-bottom: 1.5rem; }
.section { margin-bottom: 1.75rem; }
.plot-card {
  background: #fff;
  border: 1px solid #d9e2ec;
  border-radius: 8px;
  padding: 1rem 1.1rem 1.2rem;
  margin: 1rem 0 1.25rem;
  box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
}
.plot-caption {
  color: #52606d;
  font-size: 0.95rem;
  margin: 0 0 0.75rem;
}
.plot-card img {
  display: block;
  max-width: 100%;
  height: auto;
  margin: 0 auto;
}
.table-wrap {
  overflow-x: auto;
  background: #fff;
  border: 1px solid #d9e2ec;
  border-radius: 8px;
  margin: 0.75rem 0 1rem;
}
table.data-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.88rem;
}
table.data-table th,
table.data-table td {
  border-bottom: 1px solid #e4e7eb;
  padding: 0.45rem 0.6rem;
  text-align: left;
  vertical-align: top;
}
table.data-table th {
  background: #f0f4f8;
  font-weight: 600;
  position: sticky;
  top: 0;
}
table.data-table tr:nth-child(even) td { background: #fbfdff; }
.note { color: #52606d; font-size: 0.95rem; margin: 0.5rem 0 0.75rem; }
"""


PLOT_EXPLANATIONS: Dict[str, str] = {
    "coefficient_sign_stability.png": (
        "Fraction of outer folds in which each feature's fitted coefficient kept the same sign. "
        "Values near 1.0 indicate stable directional effects across resamples."
    ),
    "weighted_mri_concepts.png": (
        "Top MRI ontology concepts ranked by pathology-calibrated weight. Higher weights reflect "
        "concepts that were both reliable against pathology language and informative for dispersion."
    ),
    "mri_pathology_reliability_heatmap.png": (
        "Average cross-modal concordance between MRI and pathology concepts within outer-training folds. "
        "Brighter cells indicate stronger MRI-to-pathology alignment used for calibration."
    ),
    "top_regression_predicted_vs_true.png": (
        "Held-out predictions from the best aggregate regression model versus true dispersion score. "
        "Points near the dashed identity line indicate better calibration of continuous predictions."
    ),
    "top_regression_residuals.png": (
        "Residuals (predicted minus true) for the top regression model. Random scatter around zero "
        "suggests no strong systematic bias across the prediction range."
    ),
    "top_dispersion_high_low_confusion_matrix.png": (
        "Confusion matrix for the best dispersion high/low classifier on deduplicated held-out cases."
    ),
    "top_dispersion_high_low_roc.png": (
        "ROC curve for the best dispersion high/low classifier. Higher curves indicate better "
        "discrimination across thresholds."
    ),
    "top_dispersion_high_low_pr.png": (
        "Precision-recall curve for dispersion high/low prediction. The dashed line is the no-skill "
        "baseline equal to event prevalence."
    ),
    "top_dispersion_high_low_calibration.png": (
        "Reliability diagram for predicted high-dispersion risk. Points near the diagonal indicate "
        "well-calibrated probabilities; labels show bin counts."
    ),
    "top_relapse_status_confusion_matrix.png": (
        "Confusion matrix for the best relapse classifier on deduplicated held-out cases."
    ),
    "top_relapse_status_roc.png": (
        "ROC curve for relapse prediction from the top aggregate classification model."
    ),
    "top_relapse_status_pr.png": (
        "Precision-recall curve for relapse prediction. Useful when relapse events are imbalanced."
    ),
    "top_relapse_status_calibration.png": (
        "Calibration plot for relapse risk predictions from the top model."
    ),
    "ranked_model_mae.png": (
        "Lowest mean absolute error (MAE) regression models across dataset, representation, and learner. "
        "Lower bars are better."
    ),
    "ranked_model_auprc.png": (
        "Highest area under the precision-recall curve (AUPRC) classification models. Higher bars are better."
    ),
    "ranked_model_auroc.png": (
        "Highest area under the ROC curve (AUROC) classification models. Higher bars are better."
    ),
    "bootstrap_ci_primary_metrics.png": (
        "Primary metric point estimates with 95% bootstrap confidence intervals after case-level "
        "deduplication."
    ),
    "nested_classification_comparison.png": (
        "AUROC for every evaluated classification setting in the nested resampling pipeline."
    ),
    "nested_relapse_auroc_comparison.png": (
        "Held-out AUROC for relapse-status models only, sorted for side-by-side comparison."
    ),
    "nested_relapse_auprc_comparison.png": (
        "Held-out AUPRC for relapse-status models. More informative than AUROC when relapse is rare."
    ),
    "nested_relapse_f1_comparison.png": (
        "Held-out F1 score for relapse-status models at the default classification threshold."
    ),
    "nested_relapse_brier_comparison.png": (
        "Held-out Brier score for relapse-status models. Lower scores indicate better probabilistic calibration."
    ),
    "nested_relapse_roc_curves_top_models.png": (
        "ROC curves for the top relapse models by aggregate AUROC/AUPRC on deduplicated predictions."
    ),
    "nested_relapse_pr_curves_top_models.png": (
        "Precision-recall curves for the top relapse models on deduplicated held-out predictions."
    ),
    "nested_regression_error_comparison.png": (
        "Held-out mean absolute error (MAE) for all regression settings. Lower bars are better."
    ),
    "nested_regression_correlation_comparison.png": (
        "Held-out Spearman rank correlation between predicted and true dispersion scores."
    ),
}


def plot_explanation_for(plot_path: str) -> str:
    base = os.path.basename(str(plot_path))
    return PLOT_EXPLANATIONS.get(
        base,
        "Diagnostic figure generated from nested held-out evaluation outputs.",
    )


def df_to_html_table(
    df: Optional[pd.DataFrame],
    max_rows: int = 25,
    float_digits: int = 3,
    table_class: str = "data-table",
) -> str:
    if df is None or len(df) == 0:
        return '<p class="note"><em>No rows available.</em></p>'
    tmp = df.head(max_rows).copy()
    for col in tmp.columns:
        if pd.api.types.is_float_dtype(tmp[col]):
            tmp[col] = tmp[col].map(lambda x: "" if pd.isna(x) else f"{x:.{float_digits}f}")
    table_html = tmp.to_html(index=False, border=0, classes=table_class, escape=True)
    return f'<div class="table-wrap">{table_html}</div>'


def html_paragraph(text: str) -> str:
    return f"<p>{html.escape(str(text))}</p>"


def html_section(title: str, parts: Sequence[str]) -> str:
    body = "\n".join(part for part in parts if part)
    return f'<section class="section"><h2>{html.escape(title)}</h2>\n{body}</section>'


def html_plot_block(plot_path: str, image_src: str, title: Optional[str] = None) -> str:
    plot_title = title or os.path.basename(str(plot_path))
    caption = plot_explanation_for(plot_path)
    return (
        f'<div class="plot-card">'
        f"<h3>{html.escape(plot_title)}</h3>"
        f'<p class="plot-caption">{html.escape(caption)}</p>'
        f'<img src="{html.escape(image_src)}" alt="{html.escape(plot_title)}">'
        f"</div>"
    )


def build_html_report(title: str, intro: str, sections: Sequence[str]) -> str:
    body = "\n".join(sections)
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n<head>\n'
        '<meta charset="utf-8">\n'
        f"<title>{html.escape(title)}</title>\n"
        f"<style>{REPORT_HTML_STYLES}</style>\n"
        "</head>\n<body>\n"
        '<div class="page">\n'
        f"<h1>{html.escape(title)}</h1>\n"
        f'<p class="lead">{html.escape(intro)}</p>\n'
        f"{body}\n"
        "</div>\n</body>\n</html>"
    )


def main(
    csv_path: str = CSV_PATH,
    out_dir: Optional[str] = None,
    report_mode: str = DEFAULT_REPORT_MODE,
    row_indices_json: Optional[str] = None,
    split_id: Optional[str] = None,
    split_role: Optional[str] = None,
    filename_prefix: Optional[str] = None,
    max_api_workers: Optional[int] = None,
    resume: bool = True,
    force_reextract: bool = False,
    assume_yes: bool = False,
) -> None:
    if report_mode not in REPORT_CONFIG:
        raise ValueError(f"--report-mode must be one of {sorted(REPORT_CONFIG)}")

    if out_dir is None:
        out_dir = os.path.join(
            os.getcwd(),
            f"securegpt_dispersion_feature_discovery_{REPORT_CONFIG[report_mode]['outdir_suffix']}",
        )

    os.makedirs(out_dir, exist_ok=True)
    log_dir = os.path.join(out_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_suffix = f"_{filename_prefix}" if filename_prefix else ""
    log_path = os.path.join(log_dir, f"run_log_feature_discovery_{report_mode}{log_suffix}.txt")
    row_indices = _load_row_indices_json(row_indices_json)

    log_mode = "a" if resume and os.path.exists(log_path) else "w"
    with open(log_path, log_mode, encoding="utf-8") as log_f:
        tee_out = Tee(sys.__stdout__, log_f)
        tee_err = Tee(sys.__stderr__, log_f)
        with contextlib.redirect_stdout(tee_out), contextlib.redirect_stderr(tee_err):
            print(f"[LOG] Logging stdout/stderr to: {log_path}")
            if row_indices is not None:
                print(f"[LOG] Loaded row subset from {row_indices_json} with n={len(row_indices)}")
            run_pipeline(
                csv_path=csv_path,
                out_dir=out_dir,
                report_mode=report_mode,
                resume=resume,
                force_reextract=force_reextract,
                row_indices=row_indices,
                split_id=split_id,
                split_role=split_role,
                filename_prefix=filename_prefix,
                max_api_workers=max_api_workers,
                assume_yes=assume_yes,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SecureGPT modality-specific lexical feature discovery pipeline"
    )
    parser.add_argument(
        "--csv-path",
        "-c",
        dest="csv_path",
        default=CSV_PATH,
        help="Path to input CSV file.",
    )
    parser.add_argument(
        "--outdir",
        "-o",
        dest="out_dir",
        default=None,
        help="Directory to write outputs.",
    )
    parser.add_argument(
        "--report-mode",
        "-m",
        dest="report_mode",
        choices=sorted(REPORT_CONFIG.keys()),
        default=DEFAULT_REPORT_MODE,
        help="Which modality to extract from: mri or path.",
    )
    parser.add_argument(
        "--row-indices-json",
        dest="row_indices_json",
        default=None,
        help="Optional JSON file containing a list of integer row indices to extract.",
    )
    parser.add_argument(
        "--split-id",
        dest="split_id",
        default=None,
        help="Optional outer split identifier recorded in the outputs.",
    )
    parser.add_argument(
        "--split-role",
        dest="split_role",
        default=None,
        help="Optional split role recorded in the outputs, e.g. train/test.",
    )
    parser.add_argument(
        "--filename-prefix",
        dest="filename_prefix",
        default=None,
        help="Optional filename prefix for outputs written to outdir.",
    )
    parser.add_argument(
        "--max-api-workers",
        dest="max_api_workers",
        type=int,
        default=None,
        help="Maximum number of concurrent API extraction workers. Defaults to a conservative hardware-aware value.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        default=False,
        help="Accept the printed a-priori LLM cost estimate and skip the interactive confirmation prompt.",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        default=True,
        help="Disable reuse of per-case extraction checkpoints.",
    )
    parser.add_argument(
        "--force-reextract",
        dest="force_reextract",
        action="store_true",
        default=False,
        help="Ignore existing per-case checkpoints and call the API again.",
    )
    args = parser.parse_args()
    main(
        csv_path=args.csv_path,
        out_dir=args.out_dir,
        report_mode=args.report_mode,
        row_indices_json=args.row_indices_json,
        split_id=args.split_id,
        split_role=args.split_role,
        filename_prefix=args.filename_prefix,
        max_api_workers=args.max_api_workers,
        resume=args.resume,
        force_reextract=args.force_reextract,
        assume_yes=args.yes,
    )
