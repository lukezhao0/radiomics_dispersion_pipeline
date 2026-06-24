"""Golden tests for approach2 configuration constants."""

from __future__ import annotations

from approach2.config import (
    DISPERSION_TRUE_HIGH_THRESHOLD,
    META_COLS,
    RANDOM_SEED,
    SHARED_CONCEPT_ONTOLOGY,
    TARGET_NAME_DISPERSION_SCORE,
)


def test_dispersion_threshold():
    assert DISPERSION_TRUE_HIGH_THRESHOLD == 85.0


def test_random_seed():
    assert RANDOM_SEED == 17


def test_target_names():
    assert TARGET_NAME_DISPERSION_SCORE == "dispersion_score"


def test_ontology_keys_stable():
    expected = {
        "extent_span",
        "multiplicity",
        "multicentricity_separate_sites",
        "distribution_linear_segmental_regional",
        "fragmentation_scattered_patchy_discontinuous",
        "residual_tumor_presence",
        "non_mass_enhancement",
        "invasive_disease",
        "in_situ_disease_dcis",
        "treatment_response",
        "treatment_effect_tumor_bed",
        "localized_compact_residual",
        "diffuse_scattered_residual",
        "lymphovascular_invasion",
        "margin_proximity",
        "benign_or_nonspecific_enhancement",
    }
    assert set(SHARED_CONCEPT_ONTOLOGY.keys()) == expected


def test_meta_cols():
    assert "case_id" in META_COLS
    assert "dispersion_true" in META_COLS
    assert "relapse_true" in META_COLS
