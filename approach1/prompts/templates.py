"""User prompt and few-shot training block construction."""

from __future__ import annotations

from typing import List

from ..models import Case
from ..text_utils import modality_display_name, shorten_for_prompt
from .descriptors import DESCRIPTORS_TEXT
from .tokens import make_case_token


def report_fields_for_prompt(c: Case, modality: str) -> str:
    parts: List[str] = []
    if modality in {"mri_only", "mri_plus_pathology"}:
        parts.append(f"preop_MRI_text:\n{shorten_for_prompt(c.preop_mri)}")
    else:
        parts.append("preop_MRI_text:\n<NOT PROVIDED IN THIS TIER>")

    if modality in {"pathology_only", "mri_plus_pathology"}:
        parts.append(f"path_report_text:\n{shorten_for_prompt(c.path_report)}")
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
            f"{report_fields_for_prompt(c, modality)}\n"
        )
    return "\n\n".join(blocks)


def build_user_prompt(training_block: str, test_case: Case, row_index: int, modality: str) -> str:
    validation_token = make_case_token(test_case, row_index, modality)
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
{report_fields_for_prompt(test_case, modality)}

VALIDATION_TOKEN_FOR_THIS_CASE_DO_NOT_USE_AS_CLINICAL_EVIDENCE: {validation_token}""".strip()
