"""Pipeline configuration constants and defaults."""

from __future__ import annotations

import os

__version__ = "1.0.0"

CSV_PATH = "/Users/lukezhao/projects/onc/PROCESSED_TRIMMED_path_cases_MRI_status.csv"
OUT_DIR = os.path.join(os.getcwd(), "securegpt_dispersion_3tier_outputs")
ENV_PATH = os.getenv("ENV_PATH", "/Users/lukezhao/projects/onc/.env")

API_VERSION = "2024-12-01-preview"
DEPLOYMENT = "gpt-5-nano"
SECUREGPT_BASE_URL = "https://aihubapi.stanfordhealthcare.org/azure-openai"

MAX_TOKENS = 16000
REQUEST_TIMEOUT_S = 120
MAX_RETRIES = 3
BACKOFF_BASE_S = 1.5
RATE_LIMIT_SLEEP_S = 0.25

DISPERSION_HIGH_THRESHOLD = 85.0
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

PRICE_PER_1M_INPUT_TOKENS = 0.05
PRICE_PER_1M_CACHED_INPUT_TOKENS = 0.01
PRICE_PER_1M_OUTPUT_TOKENS = 0.40

REASONING_EFFORT = os.getenv("REASONING_EFFORT", "minimal").strip().lower()
if REASONING_EFFORT in {"", "none", "null"}:
    REASONING_EFFORT = ""

STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "for", "on", "with", "is", "are",
    "was", "were", "by", "at", "from", "as", "this", "that", "it", "be", "has", "have",
    "had", "but", "not", "no", "into", "than", "then", "there", "their", "its", "also",
    "may", "can", "which", "within", "without", "after", "before", "left", "right",
    "breast", "tumor", "carcinoma", "report", "reports", "provided", "tier",
}
