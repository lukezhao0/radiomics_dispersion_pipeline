"""Stanford AI Sandbox chat-completions client."""

from __future__ import annotations

import json
import time
from typing import Any, Dict, Sequence

import requests

from ..extraction import config as llm_config
from ..api.cost import (
    GLOBAL_API_SEMAPHORE,
    build_chat_messages,
    estimate_prompt_tokens_from_messages,
    estimate_cost_from_usage,
    update_cost_tracker,
)

# -----------------------------
# Stanford AI Sandbox chat-completions calls
# -----------------------------


def _ensure_client_configured() -> None:
    if not llm_config.URL or not llm_config.HEADERS:
        llm_config.configure_llm()


def _post_chat_completion(
    messages: Sequence[Dict[str, str]],
    max_completion_tokens: int = llm_config.MAX_TOKENS,
) -> Dict[str, Any]:
    _ensure_client_configured()
    payload = {
        "model": llm_config.DEPLOYMENT,
        "messages": list(messages),
        # GPT-5-style models account for both visible output and reasoning tokens here.
        "max_completion_tokens": int(max_completion_tokens),
    }
    if llm_config.REASONING_EFFORT:
        payload["reasoning_effort"] = llm_config.REASONING_EFFORT

    with GLOBAL_API_SEMAPHORE:
        response = requests.post(
            llm_config.URL,
            headers=llm_config.HEADERS,
            json=payload,
            timeout=180,
        )
    if not response.ok:
        print("[API_ERROR] API returned an error:")
        print(f"[API_ERROR] Status: {response.status_code}")
        try:
            print(json.dumps(response.json(), indent=2))
        except Exception:
            print(response.text)
        response.raise_for_status()
    return response.json()


def preflight_check() -> None:
    print("[PREFLIGHT] Testing Stanford AI Sandbox connectivity with a small request...")
    t0 = time.time()
    data = _post_chat_completion(
        messages=[{"role": "user", "content": "Reply with exactly: OK"}],
        max_completion_tokens=32,
    )
    dt = time.time() - t0
    usage = data.get("usage", {}) or {}
    cost_info = estimate_cost_from_usage(usage)
    update_cost_tracker(cost_info)
    choice = data["choices"][0]
    reply = choice["message"].get("content") or ""
    print(
        f"[PREFLIGHT] Success in {dt:.2f}s. "
        f"Model={data.get('model', llm_config.DEPLOYMENT)} Reply={reply!r} "
        f"prompt_tokens={cost_info['prompt_tokens']} cached_tokens={cost_info['cached_tokens']} "
        f"completion_tokens={cost_info['completion_tokens']} estimated_cost=${cost_info['estimated_cost_usd']:.8f}"
    )


def call_securegpt_chat(prompt: str) -> str:
    messages = build_chat_messages(prompt)
    prompt_token_estimate = estimate_prompt_tokens_from_messages(messages)
    print(
        f"[API] Sending request... prompt_chars={len(prompt)} "
        f"estimated_prompt_tokens={prompt_token_estimate} max_completion_tokens={llm_config.MAX_TOKENS}"
    )
    t0 = time.time()
    data = _post_chat_completion(messages=messages, max_completion_tokens=llm_config.MAX_TOKENS)
    dt = time.time() - t0

    usage = data.get("usage", {}) or {}
    cost_info = estimate_cost_from_usage(usage)
    update_cost_tracker(cost_info)

    choice = data["choices"][0]
    content = choice["message"].get("content") or ""
    finish_reason = choice.get("finish_reason")
    print(
        f"[API] Response in {dt:.2f}s. response_chars={len(content)} "
        f"model={data.get('model', llm_config.DEPLOYMENT)} finish_reason={finish_reason} "
        f"prompt_tokens={cost_info['prompt_tokens']} cached_tokens={cost_info['cached_tokens']} "
        f"completion_tokens={cost_info['completion_tokens']} reasoning_tokens={cost_info['reasoning_tokens']} "
        f"estimated_cost=${cost_info['estimated_cost_usd']:.8f} "
        f"cache_savings=${cost_info['estimated_cache_savings_usd']:.8f}"
    )
    if not content:
        print(
            "[WARNING] Empty visible output. "
            f"finish_reason={finish_reason}. Try increasing MAX_COMPLETION_TOKENS / max_completion_tokens."
        )
    return content
