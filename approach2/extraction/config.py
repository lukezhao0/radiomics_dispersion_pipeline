"""LLM extraction layer configuration (API, pricing, report modes)."""

from __future__ import annotations

import os

from dotenv import load_dotenv

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
ENV_PATH = os.getenv("SANDBOX_ENV_PATH", "/Users/lukezhao/projects/onc/.env")
if os.path.exists(ENV_PATH):
    load_dotenv(ENV_PATH, override=True)
else:
    load_dotenv(os.path.join(os.getcwd(), ".env"), override=True)

API_VERSION = "2024-12-01-preview"
DEPLOYMENT = "gpt-5-nano"  # GPT-5-nano Global deployment name on the SHC AI Sandbox.
SANDBOX_API_KEY = os.getenv("SANDBOX_API_KEY")

if not SANDBOX_API_KEY:
    raise RuntimeError(
        "SANDBOX_API_KEY not found. Set it in /Users/lukezhao/projects/onc/.env, "
        "set SANDBOX_ENV_PATH to another .env file, or export SANDBOX_API_KEY."
    )
SANDBOX_API_KEY = SANDBOX_API_KEY.strip()

URL = (
    "https://aihubapi.stanfordhealthcare.org/azure-openai"
    f"/deployments/{DEPLOYMENT}/chat/completions"
    f"?api-version={API_VERSION}"
)

HEADERS = {
    # This endpoint works with `api-key`, not Ocp-Apim-Subscription-Key.
    "api-key": SANDBOX_API_KEY,
    "Content-Type": "application/json",
    "Accept": "application/json",
}

TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", os.getenv("TEMPERATURE", "0.0")))
# Includes visible output tokens plus GPT-5 reasoning tokens. Override from the
# shell with MAX_COMPLETION_TOKENS if you need a smaller/larger cap.
MAX_TOKENS = int(os.getenv("MAX_COMPLETION_TOKENS", "16000"))
MAX_RETRIES = 3
BACKOFF_BASE_S = 1.5
RATE_LIMIT_SLEEP_S = 0.25

# MAX_SEED_PHRASES = int(os.getenv("MAX_SEED_PHRASES", "15"))
# MAX_DENOVO_PHRASES = int(os.getenv("MAX_DENOVO_PHRASES", "15"))
REASONING_EFFORT = os.getenv("REASONING_EFFORT", "minimal").strip().lower()
if REASONING_EFFORT in {"", "none", "null"}:
    REASONING_EFFORT = ""

# Editable pricing assumptions for GPT-5-nano Global.
# Units: USD per 1,000,000 tokens.
PRICE_PER_1M_INPUT_TOKENS = 0.05
PRICE_PER_1M_CACHED_INPUT_TOKENS = 0.01
PRICE_PER_1M_OUTPUT_TOKENS = 0.40
