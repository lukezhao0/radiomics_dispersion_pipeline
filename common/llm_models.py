"""Supported SecureGPT / Stanford AI Sandbox LLM deployments and pricing."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Mapping, Optional

SUPPORTED_MODELS = ("gpt-5-nano", "gpt-5")
DEFAULT_MODEL = "gpt-5-nano"


@dataclass(frozen=True)
class ModelConfig:
    deployment: str
    api_key_env_var: str
    price_per_1m_input_tokens: float
    price_per_1m_cached_input_tokens: float
    price_per_1m_output_tokens: float
    pricing_label: str


MODEL_REGISTRY: Dict[str, ModelConfig] = {
    "gpt-5-nano": ModelConfig(
        deployment="gpt-5-nano",
        api_key_env_var="SANDBOX_API_KEY",
        price_per_1m_input_tokens=0.05,
        price_per_1m_cached_input_tokens=0.01,
        price_per_1m_output_tokens=0.40,
        pricing_label="GPT-5-nano Global",
    ),
    "gpt-5": ModelConfig(
        deployment="gpt-5",
        api_key_env_var="NEW_SECUREGPT_API_KEY",
        price_per_1m_input_tokens=1.25,
        price_per_1m_cached_input_tokens=0.13,
        price_per_1m_output_tokens=10.00,
        pricing_label="GPT-5 2025-08-07 Global",
    ),
}


def normalize_model_name(model: str) -> str:
    normalized = (model or DEFAULT_MODEL).strip().lower()
    if normalized not in MODEL_REGISTRY:
        supported = ", ".join(SUPPORTED_MODELS)
        raise ValueError(f"Unsupported model {model!r}. Choose one of: {supported}")
    return normalized


def get_model_config(model: str) -> ModelConfig:
    return MODEL_REGISTRY[normalize_model_name(model)]


def resolve_api_key(
    model: str,
    *,
    env: Optional[Mapping[str, str]] = None,
    env_path: Optional[str] = None,
) -> str:
    cfg = get_model_config(model)
    if env_path:
        from dotenv import load_dotenv

        load_dotenv(env_path, override=True)
    env_map = env if env is not None else os.environ
    api_key = env_map.get(cfg.api_key_env_var)
    if not api_key:
        raise RuntimeError(
            f"{cfg.api_key_env_var} not found for model {cfg.deployment!r}. "
            "Set it in your .env file or export it in the environment."
        )
    return str(api_key).strip()
