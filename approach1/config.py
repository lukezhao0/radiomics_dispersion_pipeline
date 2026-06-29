"""Pipeline configuration constants and defaults."""

from __future__ import annotations

import os

from common.llm_models import DEFAULT_MODEL, get_model_config
from common.reasoning_effort import DEFAULT_REASONING_EFFORT, normalize_reasoning_effort

__version__ = "1.0.0"

CSV_PATH = "/Users/lukezhao/projects/onc/PROCESSED_TRIMMED_path_cases_MRI_status.csv"
OUT_DIR = os.path.join(os.getcwd(), "securegpt_dispersion_3tier_outputs")
ENV_PATH = os.getenv("SANDBOX_ENV_PATH") or os.getenv("ENV_PATH") or "/Users/lukezhao/projects/onc/.env"

API_VERSION = "2024-12-01-preview"
MODEL = DEFAULT_MODEL
DEPLOYMENT = MODEL
SECUREGPT_BASE_URL = "https://aihubapi.stanfordhealthcare.org/azure-openai"

MAX_TOKENS = 16000
REQUEST_TIMEOUT_S = 120
MAX_RETRIES = 3
BACKOFF_BASE_S = 1.5
RATE_LIMIT_SLEEP_S = 0.25

DISPERSION_HIGH_THRESHOLD = 85.0
DEFAULT_BOOTSTRAP_N = 1000
DEFAULT_BOOTSTRAP_SEED = 42
RESUME_CHECKPOINT_SUBDIR = "_resume_checkpoint"
RESUME_SCRIPT_VERSION = "approach1-3-v2"

SHOT_SETS = [
    {
        "name": "shotset_high_0_2_low_101_102",
        "high_rows": [0, 2],
        "low_rows": [101, 102],
    },
    {
        "name": "shotset_high_0_19_low_82_85",
        "high_rows": [0, 19],
        "low_rows": [82, 85],
    },
]

MODALITY_TIERS = ["mri_only", "pathology_only", "mri_plus_pathology"]

_default_pricing = get_model_config(DEFAULT_MODEL)
PRICE_PER_1M_INPUT_TOKENS = _default_pricing.price_per_1m_input_tokens
PRICE_PER_1M_CACHED_INPUT_TOKENS = _default_pricing.price_per_1m_cached_input_tokens
PRICE_PER_1M_OUTPUT_TOKENS = _default_pricing.price_per_1m_output_tokens
PRICING_LABEL = _default_pricing.pricing_label

REASONING_EFFORT = normalize_reasoning_effort()


def set_reasoning_effort(value: str | None = None) -> str:
    """Update the module-level reasoning effort used in API payloads."""
    global REASONING_EFFORT
    REASONING_EFFORT = normalize_reasoning_effort(value, default=DEFAULT_REASONING_EFFORT)
    return REASONING_EFFORT


def apply_model_config(model: str) -> None:
    """Update deployment and pricing globals for the selected model."""
    global MODEL, DEPLOYMENT, PRICE_PER_1M_INPUT_TOKENS, PRICE_PER_1M_CACHED_INPUT_TOKENS
    global PRICE_PER_1M_OUTPUT_TOKENS, PRICING_LABEL

    cfg = get_model_config(model)
    MODEL = cfg.deployment
    DEPLOYMENT = cfg.deployment
    PRICE_PER_1M_INPUT_TOKENS = cfg.price_per_1m_input_tokens
    PRICE_PER_1M_CACHED_INPUT_TOKENS = cfg.price_per_1m_cached_input_tokens
    PRICE_PER_1M_OUTPUT_TOKENS = cfg.price_per_1m_output_tokens
    PRICING_LABEL = cfg.pricing_label


STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "for", "on", "with", "is", "are",
    "was", "were", "by", "at", "from", "as", "this", "that", "it", "be", "has", "have",
    "had", "but", "not", "no", "into", "than", "then", "there", "their", "its", "also",
    "may", "can", "which", "within", "without", "after", "before", "left", "right",
    "breast", "tumor", "carcinoma", "report", "reports", "provided", "tier",
}
