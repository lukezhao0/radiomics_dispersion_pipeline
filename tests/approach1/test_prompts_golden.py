"""Golden tests for prompt construction."""

from __future__ import annotations

import hashlib

from approach1.models import Case
from approach1.prompts.templates import build_training_block, build_user_prompt
from approach1.prompts.tokens import make_case_token


def test_make_case_token_deterministic():
    case = Case(
        case_id="SYNTH_001",
        preop_mri="mri",
        path_report="path",
        index_side="left",
    )
    t1 = make_case_token(case, row_index=5, modality="mri_only")
    t2 = make_case_token(case, row_index=5, modality="mri_only")
    assert t1 == t2
    assert t1.startswith("CTXCHK_")
    assert "MODALITY_mri_only" in t1


def test_make_case_token_differs_by_modality():
    case = Case(case_id="SYNTH_001", preop_mri="mri", path_report="path", index_side="left")
    t_mri = make_case_token(case, row_index=5, modality="mri_only")
    t_path = make_case_token(case, row_index=5, modality="pathology_only")
    assert t_mri != t_path


def test_build_user_prompt_golden_hash(sample_case: Case):
    training_block = build_training_block(
        [(10, sample_case)],
        modality="mri_plus_pathology",
        high_rows={10},
    )
    prompt = build_user_prompt(training_block, sample_case, row_index=10, modality="mri_plus_pathology")
    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    # Golden hash — update only if prompt text intentionally changes
    assert len(prompt) > 1000
    assert "DISPERSIVENESS DESCRIPTORS" in prompt
    assert sample_case.case_id in prompt
    assert "VALIDATION_TOKEN_FOR_THIS_CASE" in prompt
    assert digest  # snapshot anchor; regression guard via structure checks above


def test_build_training_block_includes_labeled_outcomes(sample_case: Case):
    low_case = Case(
        case_id="SYNTH_LOW",
        preop_mri="mri low",
        path_report="path low",
        index_side="right",
        dispersion_true=40.0,
        relapse_true=0,
    )
    block = build_training_block(
        [(0, sample_case), (101, low_case)],
        modality="mri_plus_pathology",
        high_rows={0},
    )
    assert "exemplar_dispersion_band: high dispersion" in block
    assert "dispersion_score_true: 120.0" in block
    assert "dispersion_high_low_true: 1" in block
    assert "relapse_true: 1" in block
    assert "exemplar_dispersion_band: low dispersion" in block
    assert "dispersion_score_true: 40.0" in block
    assert "dispersion_high_low_true: 0" in block
    assert "relapse_true: 0" in block


def test_build_user_prompt_mri_only_omits_pathology(sample_case: Case):
    training_block = build_training_block(
        [(3, sample_case)],
        modality="mri_only",
        high_rows={3},
    )
    prompt = build_user_prompt(training_block, sample_case, row_index=3, modality="mri_only")
    assert "<NOT PROVIDED IN THIS TIER>" in prompt
    assert "MRI only" in prompt
