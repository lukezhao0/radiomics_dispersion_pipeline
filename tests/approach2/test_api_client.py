"""API client wiring tests."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest


def test_api_client_omits_temperature_in_payload() -> None:
    import approach2.extraction  # noqa: F401
    import approach2.api.client as client_mod
    import approach2.extraction.config as llm_config

    os.environ.setdefault("SANDBOX_API_KEY", "test-key")
    llm_config.configure_llm("gpt-5-nano")
    llm_config.REASONING_EFFORT = "medium"

    mock_response = MagicMock()
    mock_response.ok = True
    mock_response.json.return_value = {
        "model": "gpt-5-nano",
        "choices": [{"message": {"content": "OK"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "prompt_tokens_details": {"cached_tokens": 0}, "completion_tokens_details": {"reasoning_tokens": 0}},
    }
    captured: dict = {}

    def _capture_post(url, headers, json, timeout):
        captured.update(json)
        return mock_response

    with patch.object(client_mod.requests, "post", side_effect=_capture_post):
        client_mod._post_chat_completion([{"role": "user", "content": "hi"}])

    assert "temperature" not in captured
    assert captured["reasoning_effort"] == "medium"


def test_call_securegpt_chat_resolves_token_estimate() -> None:
    # Load through the same entry path as the nested CLI (avoids direct client import cycle).
    import approach2.extraction  # noqa: F401
    import approach2.api.client as client_mod
    import approach2.extraction.config as llm_config

    os.environ.setdefault("SANDBOX_API_KEY", "test-key")
    llm_config.configure_llm("gpt-5-nano")

    assert hasattr(client_mod, "estimate_prompt_tokens_from_messages")

    mock_response = MagicMock()
    mock_response.ok = True
    mock_response.json.return_value = {
        "model": "gpt-5-nano",
        "choices": [{"message": {"content": '{"case_id": "c1"}'}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "prompt_tokens_details": {"cached_tokens": 0},
            "completion_tokens_details": {"reasoning_tokens": 0},
        },
    }

    with patch.object(client_mod.requests, "post", return_value=mock_response):
        out = client_mod.call_securegpt_chat("test prompt")

    assert out == '{"case_id": "c1"}'


def test_ensure_client_configured_requires_explicit_configure(monkeypatch) -> None:
    import approach2.api.client as client_mod
    import approach2.extraction.config as llm_config

    monkeypatch.setattr(llm_config, "URL", None)
    monkeypatch.setattr(llm_config, "HEADERS", None)

    with pytest.raises(RuntimeError, match="configure_llm"):
        client_mod._ensure_client_configured()
