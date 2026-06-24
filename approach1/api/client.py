"""SecureGPT HTTP client with retry-safe chat completions."""

from __future__ import annotations

import json
import time
from typing import Dict, Optional

import requests
from dotenv import load_dotenv

from common.llm_models import get_model_config, resolve_api_key

from .. import config
from ..prompts.system import SYSTEM_MSG
from .cost import CostTracker


def response_to_json(response: requests.Response) -> Dict:
    try:
        return response.json()
    except Exception as exc:
        raise RuntimeError(f"API returned non-JSON response: {response.text[:1000]}") from exc


class SecureGPTClient:
    """Configured SecureGPT/Azure OpenAI chat client."""

    def __init__(
        self,
        *,
        env_path: str,
        deployment: str,
        api_version: str,
        cost_tracker: Optional[CostTracker] = None,
    ) -> None:
        self.deployment = deployment
        self.api_version = api_version
        self.cost_tracker = cost_tracker or CostTracker()

        load_dotenv(env_path, override=True)
        cfg = get_model_config(deployment)
        api_key = resolve_api_key(deployment, env_path=env_path)

        self.url = (
            config.SECUREGPT_BASE_URL
            + f"/deployments/{deployment}/chat/completions"
            + f"?api-version={api_version}"
        )
        self.headers = {
            "api-key": api_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        config.apply_model_config(deployment)
        config.DEPLOYMENT = deployment
        config.API_VERSION = api_version
        self.model_label = cfg.pricing_label

    def chat(self, prompt: str, max_completion_tokens: int = config.MAX_TOKENS) -> str:
        print(f"[API] Sending request... prompt_chars={len(prompt)}")
        payload = {
            "model": self.deployment,
            "messages": [
                {"role": "system", "content": SYSTEM_MSG},
                {"role": "user", "content": prompt},
            ],
            "max_completion_tokens": max_completion_tokens,
        }
        if config.REASONING_EFFORT:
            payload["reasoning_effort"] = config.REASONING_EFFORT

        t0 = time.time()
        response = requests.post(
            self.url,
            headers=self.headers,
            json=payload,
            timeout=config.REQUEST_TIMEOUT_S,
        )
        dt = time.time() - t0

        if not response.ok:
            print("[API] Error:")
            print(f"[API] Status: {response.status_code}")
            try:
                print(json.dumps(response.json(), indent=2))
            except Exception:
                print(response.text)
            response.raise_for_status()

        data = response_to_json(response)
        usage = data.get("usage", {}) or {}
        cost_info = self.cost_tracker.update_from_usage(usage)

        choice = data["choices"][0]
        content = choice["message"].get("content") or ""

        print(
            f"[API] Response in {dt:.2f}s. response_chars={len(content)} "
            f"model={data.get('model', self.deployment)} "
            f"prompt_tokens={cost_info['prompt_tokens']} "
            f"cached_tokens={cost_info['cached_tokens']} "
            f"completion_tokens={cost_info['completion_tokens']} "
            f"reasoning_tokens={cost_info['reasoning_tokens']} "
            f"estimated_cost=${cost_info['estimated_cost_usd']:.8f} "
            f"cache_savings=${cost_info['estimated_cache_savings_usd']:.8f}"
        )

        if not content:
            finish_reason = choice.get("finish_reason")
            print(
                "\n[WARNING] Empty visible output. "
                f"finish_reason={finish_reason}. "
                "Try increasing MAX_TOKENS / max_completion_tokens."
            )

        return content

    def preflight_check(self) -> None:
        print("[PREFLIGHT] Testing SecureGPT connectivity with a small request...")
        t0 = time.time()
        reply = self.chat("Reply with exactly: OK", max_completion_tokens=20)
        dt = time.time() - t0
        print(f"[PREFLIGHT] Success in {dt:.2f}s. Reply={reply!r}")
