"""Per-case LLM inference with validation and retry."""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

from . import config
from .api import call_securegpt_chat
from .models import Case
from .prompts.templates import build_user_prompt
from .prompts.tokens import make_case_token
from .schema.prediction import extract_json_from_text, validate_prediction_obj


def predict_case(training_block: str, test_case: Case, row_index: int, modality: str) -> Dict[str, Any]:
    user_prompt = build_user_prompt(training_block, test_case, row_index=row_index, modality=modality)
    expected_token = make_case_token(test_case, row_index, modality)
    last_err: Optional[str] = None

    for attempt in range(1, config.MAX_RETRIES + 1):
        try:
            print(f"[PREDICT] case_id={test_case.case_id} modality={modality} attempt={attempt}/{config.MAX_RETRIES}")
            raw = call_securegpt_chat(user_prompt, max_completion_tokens=config.MAX_TOKENS)
            obj = extract_json_from_text(raw)
            ok, msg = validate_prediction_obj(
                obj,
                expected_case_id=test_case.case_id,
                expected_token=expected_token,
            )
            if not ok:
                raise ValueError(f"Validation failed: {msg}. Raw head: {raw[:400]}")
            print(f"[PREDICT] case_id={test_case.case_id} modality={modality} VALID JSON received.")
            return obj
        except Exception as e:
            last_err = str(e)
            print(f"[RETRY] case_id={test_case.case_id} attempt={attempt}/{config.MAX_RETRIES} error={last_err}")
            sleep_s = config.BACKOFF_BASE_S ** (attempt - 1)
            print(f"[RETRY] Sleeping for {sleep_s:.2f}s before retry...")
            time.sleep(sleep_s)

    raise RuntimeError(
        f"Failed to get valid prediction for case_id={test_case.case_id}, modality={modality}. "
        f"Last error: {last_err}"
    )
