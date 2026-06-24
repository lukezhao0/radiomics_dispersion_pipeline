"""LLM extraction layer configuration (API, pricing, report modes)."""

from __future__ import annotations

import os
from typing import Optional

from common.llm_models import (
    DEFAULT_MODEL,
    get_model_config,
    load_model_env,
    normalize_model_name,
    resolve_env_path,
)
from common.reasoning_effort import DEFAULT_REASONING_EFFORT, normalize_reasoning_effort

CSV_PATH = "/Users/lukezhao/projects/onc/PROCESSED_TRIMMED_path_cases_MRI_status.csv"

REPORT_CONFIG = {
    "mri": {
        "field": "preop_MRI_text",
        "label": "post-neoadjuvant, pre-surgery breast MRI report",
        "outdir_suffix": "mri",
    },
    "path": {
        "field": "path_report_text",
        "label": "post-neoadjuvant pre-surgical pathology report",
        "outdir_suffix": "path",
    },
}

DEFAULT_REPORT_MODE = "mri"

# Prefer the project-level .env used by the Stanford AI Sandbox example, but
# fall back to the current working directory so the script remains portable.
ENV_PATH = os.getenv("SANDBOX_ENV_PATH") or os.getenv("ENV_PATH") or "/Users/lukezhao/projects/onc/.env"

API_VERSION = "2024-12-01-preview"
MODEL = DEFAULT_MODEL
DEPLOYMENT = MODEL
API_KEY: Optional[str] = None
URL: Optional[str] = None
HEADERS: Optional[dict] = None
PRICING_LABEL = get_model_config(DEFAULT_MODEL).pricing_label

# Includes visible output tokens plus GPT-5 reasoning tokens. Override from the
# shell with MAX_COMPLETION_TOKENS if you need a smaller/larger cap.
MAX_TOKENS = int(os.getenv("MAX_COMPLETION_TOKENS", "16000"))
MAX_RETRIES = 3
BACKOFF_BASE_S = 1.5
RATE_LIMIT_SLEEP_S = 0.25

# MAX_SEED_PHRASES = int(os.getenv("MAX_SEED_PHRASES", "15"))
# MAX_DENOVO_PHRASES = int(os.getenv("MAX_DENOVO_PHRASES", "15"))
REASONING_EFFORT = normalize_reasoning_effort()


def set_reasoning_effort(value: str | None = None) -> str:
    """Update the module-level reasoning effort used in API payloads."""
    global REASONING_EFFORT
    REASONING_EFFORT = normalize_reasoning_effort(value, default=DEFAULT_REASONING_EFFORT)
    return REASONING_EFFORT

_default_pricing = get_model_config(DEFAULT_MODEL)
PRICE_PER_1M_INPUT_TOKENS = _default_pricing.price_per_1m_input_tokens
PRICE_PER_1M_CACHED_INPUT_TOKENS = _default_pricing.price_per_1m_cached_input_tokens
PRICE_PER_1M_OUTPUT_TOKENS = _default_pricing.price_per_1m_output_tokens


def _build_url(deployment: str) -> str:
    return (
        "https://aihubapi.stanfordhealthcare.org/azure-openai"
        f"/deployments/{deployment}/chat/completions"
        f"?api-version={API_VERSION}"
    )


def configure_llm(
    model: str = DEFAULT_MODEL,
    *,
    env_path: Optional[str] = None,
    reasoning_effort: str | None = None,
) -> None:
    """Select deployment, API key, pricing, and reasoning effort for the requested model."""
    global MODEL, DEPLOYMENT, API_KEY, URL, HEADERS
    global PRICE_PER_1M_INPUT_TOKENS, PRICE_PER_1M_CACHED_INPUT_TOKENS
    global PRICE_PER_1M_OUTPUT_TOKENS, PRICING_LABEL

    if reasoning_effort is not None:
        set_reasoning_effort(reasoning_effort)

    model = normalize_model_name(model)
    cfg = get_model_config(model)
    MODEL = cfg.deployment
    DEPLOYMENT = cfg.deployment
    PRICE_PER_1M_INPUT_TOKENS = cfg.price_per_1m_input_tokens
    PRICE_PER_1M_CACHED_INPUT_TOKENS = cfg.price_per_1m_cached_input_tokens
    PRICE_PER_1M_OUTPUT_TOKENS = cfg.price_per_1m_output_tokens
    PRICING_LABEL = cfg.pricing_label

    resolved_env_path = resolve_env_path(env_path=env_path or ENV_PATH, default=ENV_PATH)
    _, API_KEY = load_model_env(model, env_path=resolved_env_path, default_env_path=ENV_PATH)
    URL = _build_url(DEPLOYMENT)
    HEADERS = {
        # This endpoint works with `api-key`, not Ocp-Apim-Subscription-Key.
        "api-key": API_KEY,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    print(
        f"[INIT] MODEL={MODEL} DEPLOYMENT={DEPLOYMENT} API_VERSION={API_VERSION} "
        f"api_key_env={cfg.api_key_env_var} pricing={PRICING_LABEL}"
    )
    print(f"[INIT] URL={URL}")


def _auto_configure_default() -> None:
    try:
        configure_llm(DEFAULT_MODEL, env_path=ENV_PATH)
    except RuntimeError:
        # Allow imports in environments without credentials; CLI entry points
        # call configure_llm explicitly before API use.
        pass


_auto_configure_default()
