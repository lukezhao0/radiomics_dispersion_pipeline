"""Prompt templates for approach2 extraction."""

from __future__ import annotations

from .builder import build_user_prompt
from .extraction import SEED_GUIDANCE, SHARED_ONTOLOGY_GUIDANCE, SYSTEM_MSG

__all__ = ["SYSTEM_MSG", "SEED_GUIDANCE", "SHARED_ONTOLOGY_GUIDANCE", "build_user_prompt"]
