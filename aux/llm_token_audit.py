#!/usr/bin/env python3
"""
Audit LLM token usage and estimated costs for the breast tumor dispersiveness
SecureGPT extraction pipeline.

What this script measures
-------------------------
1. Reconstructs the original extraction prompt used by feature_discovery_eval_ML_aux.py
   for each saved extraction row or for all cases in the input CSV.
2. Counts input tokens from:
      system message + reconstructed user prompt + small chat-message overhead.
3. Reconstructs the model's JSON output from saved extraction CSV/JSONL rows and
   counts output tokens.
4. Computes total estimated cost under several pricing tables, including GPT-5
   2025-08-07 and GPT-5.4 family prices supplied by the user.
5. Writes:
      llm_token_audit_calls.csv
      llm_token_audit_summary_by_group.csv
      llm_token_cost_estimates.csv
      llm_token_audit_summary.txt

Important limitations
---------------------
- This is a tokenizer-based estimate. Exact billed tokens require API response.usage
  captured at call time.
- Saved extraction CSVs usually store the parsed JSON object, not the model's raw
  text. If the model returned markdown fences or extra whitespace, output tokens
  may be slightly underestimated.
- The original code defined MAX_TOKENS but did not pass it to the API call, so this
  script estimates actual saved output length rather than assuming a fixed cap.
- Prompt-cache savings cannot be known exactly from saved CSVs. The script reports
  no-cache cost and optional cache-estimate scenarios.

python llm_token_audit.py \
  --csv-path /Users/lukezhao/projects/onc/PROCESSED_TRIMMED_path_cases_MRI_status.csv \
  --out-dir /Users/lukezhao/projects/onc/out/llm_token_audit_260504 \
  --extraction-glob "/Users/lukezhao/projects/onc/out/securegpt_dispersion_nested_path_informed_eval7/outer_split*/**/*case_phrase_extractions_*.csv"

"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

try:
    import tiktoken
except Exception:  # pragma: no cover
    tiktoken = None


# -----------------------------
# Original prompt constants copied from the user's feature_discovery_eval_ML_aux.py
# This avoids importing that file, which creates an API client at import time.
# -----------------------------

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

SYSTEM_MSG = (
    "You are an advanced, careful clinical NLP model operating in a PHI-secure environment. "
    "Your task is lexical feature discovery. "
    "Use only the provided report text. Do not invent content not present in the report. "
    "Return valid JSON only. "
    "For every extracted phrase, copy an exact quote from the report text. "
    "Do not reveal hidden chain-of-thought. Provide only the requested structured fields."
)

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


@dataclass
class CaseLike:
    case_id: str
    preop_mri: str
    path_report: str
    index_side: str = ""


def _safe_text(x: Any) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x)


def _is_missing_text(x: Any) -> bool:
    return _safe_text(x).strip() == ""


def _selected_report_text(case: CaseLike, report_mode: str) -> str:
    if report_mode == "mri":
        return case.preop_mri
    if report_mode == "path":
        return case.path_report
    raise ValueError(f"Unsupported report_mode: {report_mode}")


def _selected_report_field(report_mode: str) -> str:
    return REPORT_CONFIG[report_mode]["field"]


def _selected_report_label(report_mode: str) -> str:
    return REPORT_CONFIG[report_mode]["label"]


def build_user_prompt(case: CaseLike, report_mode: str) -> str:
    """Reconstructs the prompt from the user's feature_discovery_eval_ML_aux.py."""
    selected_text = _selected_report_text(case, report_mode)
    selected_field = _selected_report_field(report_mode)
    selected_label = _selected_report_label(report_mode)
    modality_extra = _mri_permissive_instructions() if report_mode == "mri" else ""

    return f"""
TASK
You are performing lexical feature discovery for breast tumor dispersiveness from a single clinical report.

Use ONLY:
- selected_report_type: {report_mode}
- selected_report_field: {selected_field}
- index_side: {case.index_side}

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

{modality_extra}

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
  "case_id": "{case.case_id}",
  "report_mode": "{report_mode}",
  "selected_report_field": "{selected_field}",
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
index_side: {case.index_side}
selected_report_type: {report_mode}
selected_report_description: {selected_label}
{selected_field}:
{selected_text}

REMINDER: Output JSON only.
""".strip()


# -----------------------------
# Token counting
# -----------------------------

class TokenCounter:
    def __init__(self, encoding_name: str = "o200k_base") -> None:
        self.encoding_name = encoding_name
        if tiktoken is None:
            self.enc = None
            print(
                "[WARN] tiktoken is not installed. Falling back to rough char/4 estimates. "
                "Install with: pip install tiktoken",
                file=sys.stderr,
            )
        else:
            try:
                self.enc = tiktoken.get_encoding(encoding_name)
            except Exception:
                print(f"[WARN] Could not load {encoding_name}; falling back to cl100k_base", file=sys.stderr)
                self.enc = tiktoken.get_encoding("cl100k_base")

    def count(self, text: str) -> int:
        text = text or ""
        if self.enc is None:
            return int(math.ceil(len(text) / 4.0))
        return len(self.enc.encode(text))

    def count_chat_input(self, system_msg: str, user_prompt: str, include_overhead: bool = True) -> int:
        # Approximate chat-message wrapper overhead. Exact GPT-5 billing is only available
        # from response.usage, but this small correction is more realistic than counting
        # content alone.
        content_tokens = self.count(system_msg) + self.count(user_prompt)
        if not include_overhead:
            return content_tokens
        return content_tokens + 3 * 2 + 3  # two messages + assistant priming


def token_prefix_len(a: List[int], b: List[int]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def common_prefix_token_count(prompts: Sequence[str], counter: TokenCounter) -> int:
    if not prompts or counter.enc is None:
        return 0
    encoded = [counter.enc.encode(p or "") for p in prompts]
    prefix = encoded[0]
    for e in encoded[1:]:
        k = token_prefix_len(prefix, e)
        prefix = prefix[:k]
        if not prefix:
            break
    return len(prefix)


# -----------------------------
# Extraction output parsing
# -----------------------------

def parse_jsonish(x: Any) -> Any:
    if isinstance(x, (dict, list)):
        return x
    if x is None:
        return None
    try:
        if pd.isna(x):
            return None
    except Exception:
        pass
    s = str(x).strip()
    if not s:
        return None
    try:
        return json.loads(s)
    except Exception:
        return None


def load_raw_cases(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = ["preop_MRI_text", "path_report_text", "index_side"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing}")
    if "case_id" not in df.columns:
        df = df.copy()
        df["case_id"] = [f"row_{i}" for i in range(len(df))]
    df = df.copy()
    df["row_index"] = list(range(len(df)))
    return df


def case_from_raw_row(row: pd.Series) -> CaseLike:
    return CaseLike(
        case_id=_safe_text(row.get("case_id", f"row_{int(row.get('row_index', -1))}")),
        preop_mri=_safe_text(row.get("preop_MRI_text", "")),
        path_report=_safe_text(row.get("path_report_text", "")),
        index_side=_safe_text(row.get("index_side", "")),
    )


def discover_extraction_files(patterns: Sequence[str]) -> List[str]:
    out: List[str] = []
    for pat in patterns:
        matches = glob.glob(pat, recursive=True)
        out.extend(matches)
    out = sorted(set(out))
    return [p for p in out if os.path.isfile(p) and p.lower().endswith((".csv", ".jsonl"))]


def read_extraction_file(path: str) -> pd.DataFrame:
    if path.lower().endswith(".jsonl"):
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        df = pd.DataFrame(rows)
    else:
        df = pd.read_csv(path)
    for c in ["seed_aligned_phrases", "denovo_candidate_phrases", "quantitative_attributes", "report_level_summary"]:
        if c in df.columns:
            df[c] = df[c].apply(parse_jsonish)
    df["source_file"] = path
    return df


def reconstruct_model_output_obj(row: pd.Series) -> Dict[str, Any]:
    """Approximate the actual model JSON object, excluding wrapper fields added after parsing."""
    return {
        "case_id": _safe_text(row.get("case_id", "")),
        "report_mode": _safe_text(row.get("report_mode", "")),
        "selected_report_field": _safe_text(row.get("selected_report_field", "")),
        "selected_report_missing": bool(int(row.get("selected_report_missing", 0) or 0)),
        "seed_aligned_phrases": row.get("seed_aligned_phrases") if isinstance(row.get("seed_aligned_phrases"), list) else [],
        "denovo_candidate_phrases": row.get("denovo_candidate_phrases") if isinstance(row.get("denovo_candidate_phrases"), list) else [],
        "quantitative_attributes": row.get("quantitative_attributes") if isinstance(row.get("quantitative_attributes"), dict) else {},
        "report_level_summary": row.get("report_level_summary") if isinstance(row.get("report_level_summary"), dict) else {},
    }


def row_call_key(row: pd.Series) -> str:
    parts = [
        _safe_text(row.get("case_id", "")),
        _safe_text(row.get("row_index", "")),
        _safe_text(row.get("report_mode", "")),
        _safe_text(row.get("outer_split_id", row.get("split_id", ""))),
        _safe_text(row.get("outer_split_role", row.get("split_role", ""))),
    ]
    return "|".join(parts)


def audit_saved_extractions(
    raw_df: pd.DataFrame,
    extraction_files: Sequence[str],
    counter: TokenCounter,
    include_chat_overhead: bool = True,
    dedupe_call_key: bool = False,
) -> pd.DataFrame:
    raw_by_row_index = {int(r["row_index"]): r for _, r in raw_df.iterrows()}
    raw_by_case_id = {str(r["case_id"]): r for _, r in raw_df.iterrows()}

    all_rows: List[pd.DataFrame] = []
    for p in extraction_files:
        try:
            df = read_extraction_file(p)
            all_rows.append(df)
        except Exception as e:
            print(f"[WARN] Failed to read {p}: {e}", file=sys.stderr)
    if not all_rows:
        return pd.DataFrame()

    ext = pd.concat(all_rows, ignore_index=True)
    if "selected_report_missing" not in ext.columns:
        ext["selected_report_missing"] = 0
    ext["selected_report_missing"] = pd.to_numeric(ext["selected_report_missing"], errors="coerce").fillna(0).astype(int)

    # Missing-report rows did not call the API in the original helper.
    ext = ext[ext["selected_report_missing"] == 0].copy()

    if dedupe_call_key:
        ext["call_key"] = ext.apply(row_call_key, axis=1)
        before = len(ext)
        ext = ext.drop_duplicates("call_key", keep="first").copy()
        print(f"[DEDUPE] Dropped {before - len(ext)} duplicate saved-call rows using call key.")

    records: List[Dict[str, Any]] = []
    prompts_by_mode: Dict[str, List[str]] = defaultdict(list)
    row_prompt_cache: List[Tuple[int, str, str]] = []

    for i, row in ext.iterrows():
        report_mode = _safe_text(row.get("report_mode", "")).lower().strip()
        if report_mode not in REPORT_CONFIG:
            print(f"[WARN] Skipping row with unsupported report_mode={report_mode!r}", file=sys.stderr)
            continue

        raw_row = None
        if "row_index" in row and not pd.isna(row.get("row_index")):
            try:
                raw_row = raw_by_row_index.get(int(row["row_index"]))
            except Exception:
                raw_row = None
        if raw_row is None:
            raw_row = raw_by_case_id.get(str(row.get("case_id", "")))
        if raw_row is None:
            print(
                f"[WARN] Could not find original raw CSV row for case_id={row.get('case_id')} row_index={row.get('row_index')}; skipping.",
                file=sys.stderr,
            )
            continue

        case = case_from_raw_row(raw_row)
        prompt = build_user_prompt(case, report_mode)
        output_obj = reconstruct_model_output_obj(row)
        output_json = json.dumps(output_obj, ensure_ascii=False, separators=(",", ":"))

        input_tokens = counter.count_chat_input(SYSTEM_MSG, prompt, include_overhead=include_chat_overhead)
        output_tokens = counter.count(output_json)
        total_tokens = input_tokens + output_tokens
        selected_report_text = _selected_report_text(case, report_mode)

        records.append({
            "source_type": "saved_extraction",
            "source_file": row.get("source_file", ""),
            "case_id": case.case_id,
            "row_index": int(raw_row["row_index"]),
            "report_mode": report_mode,
            "outer_split_id": row.get("outer_split_id", row.get("split_id", "")),
            "outer_split_role": row.get("outer_split_role", row.get("split_role", "")),
            "selected_report_chars": len(selected_report_text),
            "selected_report_words": len(re.findall(r"\S+", selected_report_text)),
            "prompt_chars": len(prompt),
            "output_json_chars": len(output_json),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "n_seed_phrases": len(output_obj.get("seed_aligned_phrases", [])),
            "n_denovo_phrases": len(output_obj.get("denovo_candidate_phrases", [])),
        })
        prompts_by_mode[report_mode].append(prompt)
        row_prompt_cache.append((len(records) - 1, report_mode, prompt))

    out = pd.DataFrame(records)
    if len(out) == 0:
        return out

    # Estimate reusable common prompt prefix by modality for prompt-cache scenario.
    common_by_mode = {mode: common_prefix_token_count(prompts, counter) for mode, prompts in prompts_by_mode.items()}
    out["common_prefix_cacheable_input_tokens_est"] = out["report_mode"].map(common_by_mode).fillna(0).astype(int)
    # First call per mode cannot benefit from cache in a simple sequential assumption.
    seen_mode: Dict[str, int] = defaultdict(int)
    cacheable = []
    for _, r in out.iterrows():
        mode = str(r["report_mode"])
        if seen_mode[mode] == 0:
            cacheable.append(0)
        else:
            cacheable.append(int(r["common_prefix_cacheable_input_tokens_est"]))
        seen_mode[mode] += 1
    out["estimated_cached_input_tokens_common_prefix"] = cacheable
    out["estimated_uncached_input_tokens_common_prefix"] = (
        out["input_tokens"] - out["estimated_cached_input_tokens_common_prefix"]
    ).clip(lower=0)
    return out


def audit_all_cases_prompt_only(
    raw_df: pd.DataFrame,
    report_modes: Sequence[str],
    counter: TokenCounter,
    include_chat_overhead: bool = True,
    assumed_output_tokens: Optional[int] = None,
) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    prompts_by_mode: Dict[str, List[str]] = defaultdict(list)
    for _, raw_row in raw_df.iterrows():
        case = case_from_raw_row(raw_row)
        for mode in report_modes:
            if mode not in REPORT_CONFIG:
                continue
            selected_text = _selected_report_text(case, mode)
            if _is_missing_text(selected_text):
                continue
            prompt = build_user_prompt(case, mode)
            input_tokens = counter.count_chat_input(SYSTEM_MSG, prompt, include_overhead=include_chat_overhead)
            output_tokens = assumed_output_tokens if assumed_output_tokens is not None else math.nan
            records.append({
                "source_type": "all_cases_prompt_only",
                "source_file": "",
                "case_id": case.case_id,
                "row_index": int(raw_row["row_index"]),
                "report_mode": mode,
                "outer_split_id": "",
                "outer_split_role": "",
                "selected_report_chars": len(selected_text),
                "selected_report_words": len(re.findall(r"\S+", selected_text)),
                "prompt_chars": len(prompt),
                "output_json_chars": math.nan,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens if not pd.isna(output_tokens) else math.nan,
                "n_seed_phrases": math.nan,
                "n_denovo_phrases": math.nan,
            })
            prompts_by_mode[mode].append(prompt)
    out = pd.DataFrame(records)
    if len(out):
        common_by_mode = {mode: common_prefix_token_count(prompts, counter) for mode, prompts in prompts_by_mode.items()}
        out["common_prefix_cacheable_input_tokens_est"] = out["report_mode"].map(common_by_mode).fillna(0).astype(int)
        seen_mode: Dict[str, int] = defaultdict(int)
        cacheable = []
        for _, r in out.iterrows():
            mode = str(r["report_mode"])
            cacheable.append(0 if seen_mode[mode] == 0 else int(r["common_prefix_cacheable_input_tokens_est"]))
            seen_mode[mode] += 1
        out["estimated_cached_input_tokens_common_prefix"] = cacheable
        out["estimated_uncached_input_tokens_common_prefix"] = (
            out["input_tokens"] - out["estimated_cached_input_tokens_common_prefix"]
        ).clip(lower=0)
    return out


# -----------------------------
# Pricing
# -----------------------------

PRICING: List[Dict[str, Any]] = [
    # User-supplied GPT-5 2025-08-07 Global pricing.
    {"model": "GPT-5 2025-08-07 Global", "processing": "standard", "context": "all", "input_per_m": 1.25, "cached_input_per_m": 0.13, "output_per_m": 10.00},
    {"model": "GPT-5 2025-08-07 Global", "processing": "priority", "context": "all", "input_per_m": 2.50, "cached_input_per_m": 0.25, "output_per_m": 20.00},
    {"model": "GPT-5 2025-08-07 Global", "processing": "batch", "context": "all", "input_per_m": 0.63, "cached_input_per_m": 0.07, "output_per_m": 5.00},

    # User-supplied GPT-5.4 family pricing.
    {"model": "GPT-5.4 Global", "processing": "standard", "context": "<272k", "input_per_m": 2.50, "cached_input_per_m": 0.25, "output_per_m": 15.00},
    {"model": "GPT-5.4 Global", "processing": "priority", "context": "<272k", "input_per_m": 5.00, "cached_input_per_m": 0.50, "output_per_m": 30.00},
    {"model": "GPT-5.4 Global", "processing": "standard", "context": ">272k", "input_per_m": 5.00, "cached_input_per_m": 0.50, "output_per_m": 22.50},

    {"model": "GPT-5.4 Pro Global", "processing": "standard", "context": "<272k", "input_per_m": 30.00, "cached_input_per_m": None, "output_per_m": 180.00},
    {"model": "GPT-5.4 Pro Global", "processing": "standard", "context": ">272k", "input_per_m": 60.00, "cached_input_per_m": None, "output_per_m": 270.00},

    {"model": "GPT-5.4 mini Global", "processing": "standard", "context": "all", "input_per_m": 0.75, "cached_input_per_m": 0.08, "output_per_m": 4.50},
    {"model": "GPT-5.4 nano Global", "processing": "standard", "context": "all", "input_per_m": 0.20, "cached_input_per_m": 0.02, "output_per_m": 1.25},
]


def compute_costs(calls_df: pd.DataFrame, custom_cached_input_frac: float = 0.0) -> pd.DataFrame:
    if len(calls_df) == 0:
        return pd.DataFrame()

    total_input = float(pd.to_numeric(calls_df["input_tokens"], errors="coerce").fillna(0).sum())
    total_output = float(pd.to_numeric(calls_df["output_tokens"], errors="coerce").fillna(0).sum())
    total_input_custom_cached = total_input * max(0.0, min(1.0, custom_cached_input_frac))
    total_input_custom_uncached = total_input - total_input_custom_cached
    total_cached_common = float(pd.to_numeric(calls_df.get("estimated_cached_input_tokens_common_prefix", 0), errors="coerce").fillna(0).sum())
    total_uncached_common = max(0.0, total_input - total_cached_common)

    rows: List[Dict[str, Any]] = []
    for price in PRICING:
        for scenario, cached_input, uncached_input in [
            ("no_cache", 0.0, total_input),
            (f"custom_cached_fraction_{custom_cached_input_frac:.2f}", total_input_custom_cached, total_input_custom_uncached),
            ("common_prefix_cache_estimate", total_cached_common, total_uncached_common),
        ]:
            cached_rate = price["cached_input_per_m"]
            if cached_rate is None:
                # Pro rows in the user table did not include cached-input pricing; bill all input as uncached.
                bill_uncached = total_input
                bill_cached = 0.0
                cache_note = "cached input pricing unavailable in supplied table; all input billed at input rate"
            else:
                bill_uncached = uncached_input
                bill_cached = cached_input
                cache_note = ""
            cost_input = bill_uncached / 1_000_000.0 * float(price["input_per_m"])
            cost_cached = bill_cached / 1_000_000.0 * float(cached_rate or 0.0)
            cost_output = total_output / 1_000_000.0 * float(price["output_per_m"])
            rows.append({
                "model": price["model"],
                "processing": price["processing"],
                "context": price["context"],
                "cache_scenario": scenario,
                "input_tokens_total": int(round(total_input)),
                "cached_input_tokens_billed": int(round(bill_cached)),
                "uncached_input_tokens_billed": int(round(bill_uncached)),
                "output_tokens_total": int(round(total_output)),
                "input_per_1m_usd": price["input_per_m"],
                "cached_input_per_1m_usd": price["cached_input_per_m"],
                "output_per_1m_usd": price["output_per_m"],
                "estimated_input_cost_usd": cost_input,
                "estimated_cached_input_cost_usd": cost_cached,
                "estimated_output_cost_usd": cost_output,
                "estimated_total_cost_usd": cost_input + cost_cached + cost_output,
                "note": cache_note,
            })
    return pd.DataFrame(rows)


def summarize_calls(calls_df: pd.DataFrame) -> pd.DataFrame:
    if len(calls_df) == 0:
        return pd.DataFrame()
    group_cols = ["source_type", "report_mode"]
    agg = calls_df.groupby(group_cols).agg(
        n_calls=("case_id", "size"),
        n_unique_cases=("case_id", "nunique"),
        input_tokens_total=("input_tokens", "sum"),
        input_tokens_mean=("input_tokens", "mean"),
        input_tokens_median=("input_tokens", "median"),
        output_tokens_total=("output_tokens", "sum"),
        output_tokens_mean=("output_tokens", "mean"),
        output_tokens_median=("output_tokens", "median"),
        total_tokens_total=("total_tokens", "sum"),
        selected_report_words_mean=("selected_report_words", "mean"),
        selected_report_words_median=("selected_report_words", "median"),
        prompt_chars_mean=("prompt_chars", "mean"),
        output_json_chars_mean=("output_json_chars", "mean"),
        estimated_cached_input_tokens_common_prefix=("estimated_cached_input_tokens_common_prefix", "sum"),
    ).reset_index()
    return agg


def write_summary_txt(
    out_path: str,
    calls_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    cost_df: pd.DataFrame,
    extraction_files: Sequence[str],
    encoding_name: str,
    include_chat_overhead: bool,
) -> None:
    lines: List[str] = []
    lines.append("LLM Token Audit Summary")
    lines.append("=" * 80)
    lines.append(f"Tokenizer encoding: {encoding_name}")
    lines.append(f"Included approximate chat-message overhead: {include_chat_overhead}")
    lines.append(f"Extraction files read: {len(extraction_files)}")
    if extraction_files:
        lines.append("First few extraction files:")
        for p in list(extraction_files)[:10]:
            lines.append(f"  - {p}")
        if len(extraction_files) > 10:
            lines.append(f"  ... {len(extraction_files) - 10} more")
    lines.append("")

    if len(calls_df) == 0:
        lines.append("No calls were audited. Provide --extraction-glob or use --audit-all-cases-prompt-only.")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        return

    total_calls = len(calls_df)
    total_input = int(pd.to_numeric(calls_df["input_tokens"], errors="coerce").fillna(0).sum())
    total_output = int(pd.to_numeric(calls_df["output_tokens"], errors="coerce").fillna(0).sum())
    total_tokens = total_input + total_output
    lines.append("Overall totals")
    lines.append("-" * 80)
    lines.append(f"Audited calls: {total_calls:,}")
    lines.append(f"Unique cases: {calls_df['case_id'].nunique():,}")
    lines.append(f"Input tokens: {total_input:,}")
    lines.append(f"Output tokens: {total_output:,}")
    lines.append(f"Total tokens: {total_tokens:,}")
    lines.append(f"Mean input tokens / call: {calls_df['input_tokens'].mean():,.1f}")
    if calls_df["output_tokens"].notna().any():
        lines.append(f"Mean output tokens / call: {calls_df['output_tokens'].mean():,.1f}")
    lines.append("")

    lines.append("By source type and modality")
    lines.append("-" * 80)
    if len(summary_df):
        for _, r in summary_df.iterrows():
            lines.append(
                f"{r['source_type']} | {r['report_mode']}: "
                f"n_calls={int(r['n_calls']):,}, "
                f"input_total={int(r['input_tokens_total']):,}, "
                f"output_total={int(r['output_tokens_total']) if not pd.isna(r['output_tokens_total']) else 'NA'}, "
                f"mean_input={r['input_tokens_mean']:,.1f}, "
                f"mean_output={r['output_tokens_mean']:,.1f}" if not pd.isna(r['output_tokens_mean']) else ""
            )
    lines.append("")

    lines.append("Cost estimates")
    lines.append("-" * 80)
    if len(cost_df):
        preferred_order = [
            ("GPT-5 2025-08-07 Global", "standard", "no_cache"),
            ("GPT-5 2025-08-07 Global", "batch", "no_cache"),
            ("GPT-5.4 Global", "standard", "no_cache"),
            ("GPT-5.4 mini Global", "standard", "no_cache"),
            ("GPT-5.4 nano Global", "standard", "no_cache"),
            ("GPT-5.4 Pro Global", "standard", "no_cache"),
        ]
        for model, processing, cache_scenario in preferred_order:
            sub = cost_df[
                (cost_df["model"] == model)
                & (cost_df["processing"] == processing)
                & (cost_df["cache_scenario"] == cache_scenario)
            ]
            if len(sub):
                r = sub.iloc[0]
                lines.append(
                    f"{r['model']} | {r['processing']} | {r['context']} | {r['cache_scenario']}: "
                    f"${r['estimated_total_cost_usd']:.4f} "
                    f"(input=${r['estimated_input_cost_usd']:.4f}, "
                    f"cached_input=${r['estimated_cached_input_cost_usd']:.4f}, "
                    f"output=${r['estimated_output_cost_usd']:.4f})"
                )
        lines.append("")
        cheapest = cost_df.sort_values("estimated_total_cost_usd").iloc[0]
        most_expensive = cost_df.sort_values("estimated_total_cost_usd").iloc[-1]
        lines.append(
            f"Cheapest row in full cost table: {cheapest['model']} | {cheapest['processing']} | "
            f"{cheapest['context']} | {cheapest['cache_scenario']} = ${cheapest['estimated_total_cost_usd']:.4f}"
        )
        lines.append(
            f"Most expensive row in full cost table: {most_expensive['model']} | {most_expensive['processing']} | "
            f"{most_expensive['context']} | {most_expensive['cache_scenario']} = ${most_expensive['estimated_total_cost_usd']:.4f}"
        )
    lines.append("")

    lines.append("Files written")
    lines.append("-" * 80)
    out_dir = os.path.dirname(out_path)
    for name in [
        "llm_token_audit_calls.csv",
        "llm_token_audit_summary_by_group.csv",
        "llm_token_cost_estimates.csv",
        "llm_token_audit_summary.txt",
    ]:
        lines.append(f"  - {os.path.join(out_dir, name)}")
    lines.append("")

    lines.append("Caveats")
    lines.append("-" * 80)
    lines.append("- Exact billed tokens require capturing resp.usage during API calls.")
    lines.append("- Output token counts are reconstructed from saved parsed JSON, not raw model text.")
    lines.append("- Prompt-cache cost scenarios are estimates; actual cached tokens depend on provider/API behavior.")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit LLM token usage and estimated cost for saved extraction outputs.")
    parser.add_argument("--csv-path", required=True, help="Original raw CSV used by the extraction pipeline.")
    parser.add_argument("--out-dir", required=True, help="Directory where audit outputs will be written.")
    parser.add_argument(
        "--extraction-glob",
        action="append",
        default=[],
        help=(
            "Glob pattern for saved extraction CSV/JSONL files. Can be repeated. "
            "Example: '/path/to/outer_splits/outer_split*/**/*case_phrase_extractions_*.csv'"
        ),
    )
    parser.add_argument(
        "--audit-all-cases-prompt-only",
        action="store_true",
        help="Also reconstruct prompts for all non-missing cases/modalities in the raw CSV, even without saved outputs.",
    )
    parser.add_argument(
        "--report-modes",
        nargs="+",
        default=["mri", "path"],
        choices=["mri", "path"],
        help="Modalities to audit for --audit-all-cases-prompt-only.",
    )
    parser.add_argument(
        "--assumed-output-tokens",
        type=int,
        default=None,
        help="Optional output-token assumption for prompt-only all-case audit rows.",
    )
    parser.add_argument(
        "--encoding",
        default="o200k_base",
        help="tiktoken encoding to use. Default: o200k_base.",
    )
    parser.add_argument(
        "--no-chat-overhead",
        action="store_true",
        help="Do not add approximate chat-message overhead tokens.",
    )
    parser.add_argument(
        "--dedupe-call-key",
        action="store_true",
        help=(
            "Deduplicate saved extraction rows by case_id,row_index,report_mode,outer_split_id,outer_split_role. "
            "Use this when files were copied/duplicated and you want unique calls, not actual repeated usage."
        ),
    )
    parser.add_argument(
        "--custom-cached-input-frac",
        type=float,
        default=0.0,
        help="Additional hypothetical cached-input fraction used in cost table. Default 0.0.",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    raw_df = load_raw_cases(args.csv_path)
    counter = TokenCounter(args.encoding)
    include_chat_overhead = not args.no_chat_overhead

    extraction_files = discover_extraction_files(args.extraction_glob)
    print(f"[INFO] Raw cases loaded: {len(raw_df):,}")
    print(f"[INFO] Extraction files discovered: {len(extraction_files):,}")

    call_tables: List[pd.DataFrame] = []
    if extraction_files:
        saved_df = audit_saved_extractions(
            raw_df=raw_df,
            extraction_files=extraction_files,
            counter=counter,
            include_chat_overhead=include_chat_overhead,
            dedupe_call_key=args.dedupe_call_key,
        )
        if len(saved_df):
            call_tables.append(saved_df)

    if args.audit_all_cases_prompt_only:
        prompt_df = audit_all_cases_prompt_only(
            raw_df=raw_df,
            report_modes=args.report_modes,
            counter=counter,
            include_chat_overhead=include_chat_overhead,
            assumed_output_tokens=args.assumed_output_tokens,
        )
        if len(prompt_df):
            call_tables.append(prompt_df)

    calls_df = pd.concat(call_tables, ignore_index=True) if call_tables else pd.DataFrame()
    summary_df = summarize_calls(calls_df)
    cost_df = compute_costs(calls_df, custom_cached_input_frac=args.custom_cached_input_frac)

    calls_csv = os.path.join(args.out_dir, "llm_token_audit_calls.csv")
    summary_csv = os.path.join(args.out_dir, "llm_token_audit_summary_by_group.csv")
    cost_csv = os.path.join(args.out_dir, "llm_token_cost_estimates.csv")
    summary_txt = os.path.join(args.out_dir, "llm_token_audit_summary.txt")

    calls_df.to_csv(calls_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    cost_df.to_csv(cost_csv, index=False)
    write_summary_txt(
        out_path=summary_txt,
        calls_df=calls_df,
        summary_df=summary_df,
        cost_df=cost_df,
        extraction_files=extraction_files,
        encoding_name=args.encoding,
        include_chat_overhead=include_chat_overhead,
    )

    print(f"[SAVE] {calls_csv}")
    print(f"[SAVE] {summary_csv}")
    print(f"[SAVE] {cost_csv}")
    print(f"[SAVE] {summary_txt}")
    if len(calls_df):
        print(
            f"[DONE] audited_calls={len(calls_df):,} "
            f"input_tokens={int(calls_df['input_tokens'].fillna(0).sum()):,} "
            f"output_tokens={int(calls_df['output_tokens'].fillna(0).sum()):,}"
        )
    else:
        print("[DONE] No call rows audited. Check your --extraction-glob or use --audit-all-cases-prompt-only.")


if __name__ == "__main__":
    main()
