"""Dynamic user prompt construction for extraction."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence
from ..extraction.data import (
    Case,
    _selected_report_field,
    _selected_report_label,
    _selected_report_text,
)
from .extraction import SEED_GUIDANCE, SHARED_ONTOLOGY_GUIDANCE

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
