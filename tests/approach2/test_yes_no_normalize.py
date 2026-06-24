"""Tests for case-insensitive cost confirmation helpers."""

from __future__ import annotations

import pytest

from approach2.text_utils import is_affirmative_response, is_negative_response, normalize_yes_no


@pytest.mark.parametrize("reply", ["yes", "YES", "Yes", "yEs", "y", "Y"])
def test_affirmative_variants(reply: str) -> None:
    assert is_affirmative_response(reply)
    assert not is_negative_response(reply)


@pytest.mark.parametrize("reply", ["no", "NO", "n", "N"])
def test_negative_variants(reply: str) -> None:
    assert is_negative_response(reply)
    assert not is_affirmative_response(reply)


def test_normalize_yes_no_strips_and_lowercases() -> None:
    assert normalize_yes_no("  YES \n") == "yes"
