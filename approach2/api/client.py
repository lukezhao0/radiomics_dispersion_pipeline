"""Stanford AI Sandbox chat-completions client."""

from __future__ import annotations

import json
import sys
import time
import traceback
from typing import Any, Dict, List, Sequence

import requests

from ..extraction.config import (
    API_VERSION,
    BACKOFF_BASE_S,
    DEPLOYMENT,
    HEADERS,
    MAX_RETRIES,
    MAX_TOKENS,
    RATE_LIMIT_SLEEP_S,
    REASONING_EFFORT,
    TEMPERATURE,
    URL,
)
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

print("[INIT] Stanford AI Sandbox request client configured.")
print(f"[INIT] URL={URL}")
print(f"[INIT] DEPLOYMENT={DEPLOYMENT} API_VERSION={API_VERSION}")


def _post_chat_completion(messages: Sequence[Dict[str, str]], max_completion_tokens: int = MAX_TOKENS) -> Dict[str, Any]:
    payload = {
        "model": DEPLOYMENT,
        "messages": list(messages),
        # GPT-5-style models account for both visible output and reasoning tokens here.
        "max_completion_tokens": int(max_completion_tokens),
    }
    if REASONING_EFFORT:
        payload["reasoning_effort"] = REASONING_EFFORT

    with GLOBAL_API_SEMAPHORE:
        response = requests.post(URL, headers=HEADERS, json=payload, timeout=180)
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
        f"Model={data.get('model', DEPLOYMENT)} Reply={reply!r} "
        f"prompt_tokens={cost_info['prompt_tokens']} cached_tokens={cost_info['cached_tokens']} "
        f"completion_tokens={cost_info['completion_tokens']} estimated_cost=${cost_info['estimated_cost_usd']:.8f}"
    )


def call_securegpt_chat(prompt: str) -> str:
    messages = build_chat_messages(prompt)
    prompt_token_estimate = estimate_prompt_tokens_from_messages(messages)
    print(
        f"[API] Sending request... prompt_chars={len(prompt)} "
        f"estimated_prompt_tokens={prompt_token_estimate} max_completion_tokens={MAX_TOKENS}"
    )
    t0 = time.time()
    data = _post_chat_completion(messages=messages, max_completion_tokens=MAX_TOKENS)
    dt = time.time() - t0

    usage = data.get("usage", {}) or {}
    cost_info = estimate_cost_from_usage(usage)
    update_cost_tracker(cost_info)

    choice = data["choices"][0]
    content = choice["message"].get("content") or ""
    finish_reason = choice.get("finish_reason")
    print(
        f"[API] Response in {dt:.2f}s. response_chars={len(content)} "
        f"model={data.get('model', DEPLOYMENT)} finish_reason={finish_reason} "
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
