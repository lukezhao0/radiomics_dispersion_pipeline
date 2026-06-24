"""Approach 1 API client tests."""

from __future__ import annotations

import pytest

from unittest.mock import MagicMock, patch

from approach1 import config
from approach1.api import client as client_mod
from approach1.api.client import SecureGPTClient
from common.llm_models import get_model_config


def _mock_response() -> MagicMock:
    mock_response = MagicMock()
    mock_response.ok = True
    mock_response.json.return_value = {
        "model": "gpt-5-nano",
        "choices": [{"message": {"content": "OK"}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "prompt_tokens_details": {"cached_tokens": 0},
            "completion_tokens_details": {"reasoning_tokens": 0},
        },
    }
    return mock_response


def test_chat_includes_reasoning_effort_when_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "REASONING_EFFORT", "minimal")
    monkeypatch.setattr(client_mod, "load_model_env", lambda *args, **kwargs: (get_model_config("gpt-5-nano"), "test-key"))
    captured: dict = {}

    def _capture_post(url, headers, json, timeout):
        captured.update(json)
        return _mock_response()

    client = SecureGPTClient(
        env_path="/dev/null",
        deployment="gpt-5-nano",
        api_version="2024-12-01-preview",
    )
    with patch.object(client_mod.requests, "post", side_effect=_capture_post):
        client.chat("hello")

    assert captured["reasoning_effort"] == "minimal"


def test_chat_omits_reasoning_effort_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "REASONING_EFFORT", "")
    monkeypatch.setattr(client_mod, "load_model_env", lambda *args, **kwargs: (get_model_config("gpt-5-nano"), "test-key"))
    captured: dict = {}

    def _capture_post(url, headers, json, timeout):
        captured.update(json)
        return _mock_response()

    client = SecureGPTClient(
        env_path="/dev/null",
        deployment="gpt-5-nano",
        api_version="2024-12-01-preview",
    )
    with patch.object(client_mod.requests, "post", side_effect=_capture_post):
        client.chat("hello")

    assert "reasoning_effort" not in captured
