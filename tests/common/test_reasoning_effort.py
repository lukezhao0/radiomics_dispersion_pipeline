"""Tests for shared reasoning-effort normalization."""

from __future__ import annotations

import pytest

from common.reasoning_effort import DEFAULT_REASONING_EFFORT, normalize_reasoning_effort


def test_default_reasoning_effort_is_medium() -> None:
    assert DEFAULT_REASONING_EFFORT == "medium"
    assert normalize_reasoning_effort() == "medium"


@pytest.mark.parametrize("raw", ["none", "null", "", "NONE"])
def test_none_values_omit_payload(raw: str) -> None:
    assert normalize_reasoning_effort(raw) == ""


def test_invalid_reasoning_effort_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported reasoning effort"):
        normalize_reasoning_effort("turbo")
