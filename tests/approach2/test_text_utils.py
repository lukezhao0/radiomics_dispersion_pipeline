"""Tests for approach2 text utility helpers."""

from __future__ import annotations

from approach2.text_utils import (
    detect_negation,
    detect_uncertainty,
    make_slug,
    normalize_text,
)


def test_normalize_text_lowercases_and_collapses_whitespace():
    assert normalize_text("  Patchy   Enhancement  ") == "patchy enhancement"


def test_make_slug():
    assert make_slug("Multi-Focal Residual") == "multi_focal_residual"


def test_detect_negation():
    assert detect_negation("No residual enhancement identified") is True
    assert detect_negation("Residual enhancement present") is False


def test_detect_uncertainty():
    assert detect_uncertainty("Possible residual disease") is True
    assert detect_uncertainty("Definite residual mass") is False
