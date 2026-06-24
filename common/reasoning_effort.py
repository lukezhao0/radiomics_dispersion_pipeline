"""Shared reasoning-effort normalization for GPT-5-style chat completions."""

from __future__ import annotations

import os

DEFAULT_REASONING_EFFORT = "medium"
REASONING_EFFORT_CHOICES = ("minimal", "low", "medium", "high", "none")
_SUPPORTED_PAYLOAD_VALUES = frozenset({"minimal", "low", "medium", "high"})


def normalize_reasoning_effort(
    value: str | None = None,
    *,
    default: str | None = None,
) -> str:
    """Return a payload-ready reasoning effort, or '' to omit from the API request."""
    resolved_default = default if default is not None else os.getenv("REASONING_EFFORT", DEFAULT_REASONING_EFFORT)
    raw = (value if value is not None else resolved_default).strip().lower()
    if raw in {"", "none", "null"}:
        return ""
    if raw not in _SUPPORTED_PAYLOAD_VALUES:
        supported = ", ".join(REASONING_EFFORT_CHOICES)
        raise ValueError(f"Unsupported reasoning effort {value!r}. Choose one of: {supported}")
    return raw
