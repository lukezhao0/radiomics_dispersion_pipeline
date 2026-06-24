"""Tests for shared LLM model configuration."""

from __future__ import annotations

import pytest

from common.llm_models import (
    DEFAULT_MODEL,
    get_model_config,
    normalize_model_name,
    resolve_api_key,
)


def test_default_model_is_gpt5_nano() -> None:
    assert DEFAULT_MODEL == "gpt-5-nano"
    assert normalize_model_name("gpt-5-nano") == "gpt-5-nano"


def test_gpt5_pricing() -> None:
    cfg = get_model_config("gpt-5")
    assert cfg.deployment == "gpt-5"
    assert cfg.api_key_env_var == "NEW_SECUREGPT_API_KEY"
    assert cfg.price_per_1m_input_tokens == 1.25
    assert cfg.price_per_1m_cached_input_tokens == 0.13
    assert cfg.price_per_1m_output_tokens == 10.00


def test_gpt5_nano_pricing_unchanged() -> None:
    cfg = get_model_config("gpt-5-nano")
    assert cfg.api_key_env_var == "SANDBOX_API_KEY"
    assert cfg.price_per_1m_input_tokens == 0.05
    assert cfg.price_per_1m_cached_input_tokens == 0.01
    assert cfg.price_per_1m_output_tokens == 0.40


def test_resolve_api_key_switches_by_model() -> None:
    env = {
        "SANDBOX_API_KEY": "sandbox-key",
        "NEW_SECUREGPT_API_KEY": "securegpt-key",
    }
    assert resolve_api_key("gpt-5-nano", env=env) == "sandbox-key"
    assert resolve_api_key("gpt-5", env=env) == "securegpt-key"


def test_resolve_api_key_missing_raises() -> None:
    with pytest.raises(RuntimeError, match="NEW_SECUREGPT_API_KEY"):
        resolve_api_key("gpt-5", env={"SANDBOX_API_KEY": "sandbox-key"})


def test_estimate_cost_uses_model_pricing(monkeypatch) -> None:
    from approach2.api.cost import estimate_cost_from_usage
    from approach2.extraction.config import configure_llm

    monkeypatch.setenv("SANDBOX_API_KEY", "sandbox-key")
    monkeypatch.setenv("NEW_SECUREGPT_API_KEY", "securegpt-key")

    usage = {
        "prompt_tokens": 1_000_000,
        "completion_tokens": 0,
        "total_tokens": 1_000_000,
        "prompt_tokens_details": {"cached_tokens": 0},
        "completion_tokens_details": {"reasoning_tokens": 0},
    }
    configure_llm("gpt-5")
    gpt5_cost = estimate_cost_from_usage(usage)["estimated_cost_usd"]

    configure_llm("gpt-5-nano")
    nano_cost = estimate_cost_from_usage(usage)["estimated_cost_usd"]

    assert gpt5_cost == pytest.approx(1.25)
    assert nano_cost == pytest.approx(0.05)
