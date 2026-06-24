"""Tests for approach2 cumulative cost persistence across resume."""

from __future__ import annotations

import json

import pytest

import approach2.extraction  # noqa: F401
from approach2.api import cost as cost_mod


def test_cost_tracker_persists_and_resumes(tmp_path, monkeypatch) -> None:
    report_path = tmp_path / "llm_token_cost_report.json"

    monkeypatch.setattr(cost_mod, "COST_PERSIST_PATH", str(report_path))
    cost_mod.reset_cost_tracker()
    cost_mod.update_cost_tracker(
        {
            "prompt_tokens": 10,
            "cached_tokens": 0,
            "uncached_prompt_tokens": 10,
            "completion_tokens": 5,
            "reasoning_tokens": 2,
            "total_tokens": 15,
            "estimated_cost_usd": 0.01,
            "estimated_cache_savings_usd": 0.0,
        }
    )

    saved = json.loads(report_path.read_text(encoding="utf-8"))
    assert saved["calls"] == 1
    assert saved["estimated_cost_usd"] == 0.01

    cost_mod.reset_cost_tracker()
    cost_mod.initialize_cost_tracker_for_resume(str(report_path), resume=True)
    snapshot = cost_mod.get_cost_tracker_snapshot()
    assert snapshot["calls"] == 1
    assert snapshot["estimated_cost_usd"] == 0.01

    cost_mod.update_cost_tracker(
        {
            "prompt_tokens": 4,
            "cached_tokens": 0,
            "uncached_prompt_tokens": 4,
            "completion_tokens": 1,
            "reasoning_tokens": 0,
            "total_tokens": 5,
            "estimated_cost_usd": 0.02,
            "estimated_cache_savings_usd": 0.0,
        }
    )
    resumed = cost_mod.get_cost_tracker_snapshot()
    assert resumed["calls"] == 2
    assert resumed["estimated_cost_usd"] == pytest.approx(0.03)
