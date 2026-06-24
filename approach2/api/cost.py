"""Token/cost tracking and a-priori estimation for extraction API calls."""

from __future__ import annotations

import json
import os
import re
import sys
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

from ..extraction import config as llm_config
from ..extraction.data import Case
from ..io_atomic import atomic_write_json
from ..prompts.builder import build_user_prompt
from ..prompts.extraction import SYSTEM_MSG
from ..text_utils import is_affirmative_response, is_negative_response

# -----------------------------
# Token / cost tracking helpers
# -----------------------------

COST_TRACKER_LOCK = threading.Lock()
COST_PERSIST_PATH: Optional[str] = None
COST_PERSIST_RESUME = True
COST_TRACKER: Dict[str, Any] = {
    "calls": 0,
    "prompt_tokens": 0,
    "cached_tokens": 0,
    "uncached_prompt_tokens": 0,
    "completion_tokens": 0,
    "reasoning_tokens": 0,
    "total_tokens": 0,
    "estimated_cost_usd": 0.0,
    "estimated_cache_savings_usd": 0.0,
}

# Global API gate used when fold-level and modality-level parallelism are enabled.
# Thread pools may create many workers, but only this many active HTTP requests
# can be in flight across the entire Python process.
GLOBAL_API_CONCURRENCY_LOCK = threading.Lock()
GLOBAL_API_MAX_CONCURRENCY = 1
GLOBAL_API_SEMAPHORE = threading.BoundedSemaphore(GLOBAL_API_MAX_CONCURRENCY)


def empty_cost_tracker() -> Dict[str, Any]:
    return {
        "calls": 0,
        "prompt_tokens": 0,
        "cached_tokens": 0,
        "uncached_prompt_tokens": 0,
        "completion_tokens": 0,
        "reasoning_tokens": 0,
        "total_tokens": 0,
        "estimated_cost_usd": 0.0,
        "estimated_cache_savings_usd": 0.0,
    }


def _tracker_fields_from_payload(data: Dict[str, Any]) -> Dict[str, Any]:
    source = data.get("cumulative") if isinstance(data.get("cumulative"), dict) else data
    out = empty_cost_tracker()
    for key in out:
        if key in source:
            out[key] = source[key]
    return out


def load_cost_tracker_snapshot(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    return _tracker_fields_from_payload(data)


def reset_cost_tracker() -> None:
    with COST_TRACKER_LOCK:
        COST_TRACKER.clear()
        COST_TRACKER.update(empty_cost_tracker())


def set_cost_persist_path(path: Optional[str], *, resume: bool = True) -> None:
    global COST_PERSIST_PATH, COST_PERSIST_RESUME
    COST_PERSIST_PATH = path
    COST_PERSIST_RESUME = bool(resume)


def initialize_cost_tracker_for_resume(path: str, *, resume: bool = True) -> None:
    """Load prior cumulative usage from disk so resumed runs keep accurate totals."""
    reset_cost_tracker()
    if not resume or not path:
        return
    prior = load_cost_tracker_snapshot(path)
    if not prior:
        return
    with COST_TRACKER_LOCK:
        COST_TRACKER.update(prior)


def _persist_cost_tracker_if_configured() -> None:
    if not COST_PERSIST_PATH:
        return
    write_cost_tracker_json(
        os.path.dirname(COST_PERSIST_PATH),
        filename=os.path.basename(COST_PERSIST_PATH),
        quiet=True,
    )


def configure_global_api_concurrency(max_concurrent_requests: int) -> None:
    """Set a process-wide cap on simultaneous Stanford AI Sandbox calls."""
    global GLOBAL_API_MAX_CONCURRENCY, GLOBAL_API_SEMAPHORE
    max_concurrent_requests = max(1, int(max_concurrent_requests or 1))
    with GLOBAL_API_CONCURRENCY_LOCK:
        GLOBAL_API_MAX_CONCURRENCY = max_concurrent_requests
        GLOBAL_API_SEMAPHORE = threading.BoundedSemaphore(max_concurrent_requests)
    print(f"[API_CONCURRENCY] Global API request cap set to {max_concurrent_requests}.")


def get_global_api_concurrency() -> int:
    with GLOBAL_API_CONCURRENCY_LOCK:
        return int(GLOBAL_API_MAX_CONCURRENCY)


def estimate_cost_from_usage(usage: Dict[str, Any]) -> Dict[str, Any]:
    """Estimate post-LLM cost from one API response usage object.

    Billing assumption:
      uncached input tokens use input-token price
      cached input tokens use cached-input-token price
      completion tokens use output-token price

    For GPT-5-style models, completion_tokens can include both visible output
    and hidden reasoning tokens, so all completion_tokens are priced as output.
    """
    usage = usage or {}
    prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
    completion_tokens = int(usage.get("completion_tokens", 0) or 0)
    total_tokens = int(usage.get("total_tokens", 0) or 0)

    prompt_details = usage.get("prompt_tokens_details", {}) or {}
    completion_details = usage.get("completion_tokens_details", {}) or {}

    cached_tokens = int(prompt_details.get("cached_tokens", 0) or 0)
    reasoning_tokens = int(completion_details.get("reasoning_tokens", 0) or 0)
    uncached_prompt_tokens = max(prompt_tokens - cached_tokens, 0)

    input_cost = (uncached_prompt_tokens / 1_000_000) * llm_config.PRICE_PER_1M_INPUT_TOKENS
    cached_input_cost = (cached_tokens / 1_000_000) * llm_config.PRICE_PER_1M_CACHED_INPUT_TOKENS
    output_cost = (completion_tokens / 1_000_000) * llm_config.PRICE_PER_1M_OUTPUT_TOKENS
    estimated_cost = input_cost + cached_input_cost + output_cost

    no_cache_input_cost = (prompt_tokens / 1_000_000) * llm_config.PRICE_PER_1M_INPUT_TOKENS
    actual_input_cost = input_cost + cached_input_cost
    cache_savings = max(no_cache_input_cost - actual_input_cost, 0.0)

    return {
        "prompt_tokens": prompt_tokens,
        "cached_tokens": cached_tokens,
        "uncached_prompt_tokens": uncached_prompt_tokens,
        "completion_tokens": completion_tokens,
        "reasoning_tokens": reasoning_tokens,
        "total_tokens": total_tokens,
        "estimated_input_cost_usd": input_cost,
        "estimated_cached_input_cost_usd": cached_input_cost,
        "estimated_output_cost_usd": output_cost,
        "estimated_cost_usd": estimated_cost,
        "estimated_cache_savings_usd": cache_savings,
    }


def estimate_cost_from_token_counts(
    prompt_tokens: int,
    completion_tokens: int,
    cached_tokens: int = 0,
) -> Dict[str, Any]:
    """Estimate cost from token counts before or after an API call."""
    prompt_tokens = int(prompt_tokens or 0)
    completion_tokens = int(completion_tokens or 0)
    cached_tokens = min(max(int(cached_tokens or 0), 0), prompt_tokens)
    usage = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "prompt_tokens_details": {"cached_tokens": cached_tokens},
        "completion_tokens_details": {"reasoning_tokens": 0},
    }
    return estimate_cost_from_usage(usage)


def update_cost_tracker(cost_info: Dict[str, Any]) -> None:
    """Thread-safe cumulative token/cost accounting."""
    with COST_TRACKER_LOCK:
        COST_TRACKER["calls"] += 1
        COST_TRACKER["prompt_tokens"] += int(cost_info["prompt_tokens"])
        COST_TRACKER["cached_tokens"] += int(cost_info["cached_tokens"])
        COST_TRACKER["uncached_prompt_tokens"] += int(cost_info["uncached_prompt_tokens"])
        COST_TRACKER["completion_tokens"] += int(cost_info["completion_tokens"])
        COST_TRACKER["reasoning_tokens"] += int(cost_info["reasoning_tokens"])
        COST_TRACKER["total_tokens"] += int(cost_info["total_tokens"])
        COST_TRACKER["estimated_cost_usd"] += float(cost_info["estimated_cost_usd"])
        COST_TRACKER["estimated_cache_savings_usd"] += float(cost_info["estimated_cache_savings_usd"])
    _persist_cost_tracker_if_configured()


def get_cost_tracker_snapshot() -> Dict[str, Any]:
    with COST_TRACKER_LOCK:
        return dict(COST_TRACKER)


def print_cumulative_report(*, label: str = "current session") -> None:
    snapshot = get_cost_tracker_snapshot()
    print(f"\n[CUMULATIVE TOKEN / COST REPORT] ({label})")
    print(f"model:                    {llm_config.MODEL}")
    print(f"pricing_label:            {llm_config.PRICING_LABEL}")
    print(
        "pricing_per_1M:           "
        f"input=${llm_config.PRICE_PER_1M_INPUT_TOKENS:.4f} "
        f"cached_input=${llm_config.PRICE_PER_1M_CACHED_INPUT_TOKENS:.4f} "
        f"output=${llm_config.PRICE_PER_1M_OUTPUT_TOKENS:.4f}"
    )
    print(f"calls:                    {snapshot['calls']}")
    print(f"prompt_tokens:            {snapshot['prompt_tokens']}")
    print(f"cached_tokens:            {snapshot['cached_tokens']}")
    print(f"uncached_prompt_tokens:   {snapshot['uncached_prompt_tokens']}")
    print(f"completion_tokens:        {snapshot['completion_tokens']}")
    print(f"reasoning_tokens:         {snapshot['reasoning_tokens']}")
    print(f"total_tokens:             {snapshot['total_tokens']}")
    print(f"estimated_total_cost_usd: ${snapshot['estimated_cost_usd']:.8f}")
    print(f"estimated_cache_savings:  ${snapshot['estimated_cache_savings_usd']:.8f}")


def write_cost_tracker_json(
    out_dir: str,
    filename: str = "llm_token_cost_report.json",
    *,
    quiet: bool = False,
) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    cumulative = get_cost_tracker_snapshot()
    payload = {
        "cost_type": "post_run_actual",
        "model": llm_config.MODEL,
        "deployment": llm_config.DEPLOYMENT,
        "pricing_label": llm_config.PRICING_LABEL,
        "api_version": llm_config.API_VERSION,
        "reasoning_effort": llm_config.REASONING_EFFORT or None,
        "price_per_1M_input_tokens": llm_config.PRICE_PER_1M_INPUT_TOKENS,
        "price_per_1M_cached_input_tokens": llm_config.PRICE_PER_1M_CACHED_INPUT_TOKENS,
        "price_per_1M_output_tokens": llm_config.PRICE_PER_1M_OUTPUT_TOKENS,
        "billing_note": (
            "Post-run actuals from API usage metadata. cached_tokens are subtracted from "
            "prompt_tokens for uncached input billing; they are not double-counted."
        ),
        "written_at": datetime.now().isoformat(timespec="seconds"),
        "cumulative": cumulative,
    }
    payload.update(cumulative)
    atomic_write_json(payload, path)
    if not quiet:
        print(f"[SAVE] Wrote LLM token/cost report (post-run actuals): {path}")
    return path


def write_apriori_cost_estimate_json(
    out_dir: str,
    estimate: Dict[str, Any],
    label: str = "planned pipeline",
    filename: str = "llm_cost_estimate_apriori.json",
) -> str:
    """Persist a-priori cost estimate separately from post-run actuals."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    payload = dict(estimate)
    payload.update({
        "cost_type": "apriori_estimate",
        "estimate_scope": label,
        "assumptions": (
            "Uses rendered prompts for scheduled cases, MAX_TOKENS completion cap per call, "
            "and optional static-prefix cache model within modality. Not exact billing."
        ),
        "written_at": datetime.now().isoformat(timespec="seconds"),
    })
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"[SAVE] Wrote a-priori LLM cost estimate: {path}")
    return path


def _rough_token_count(text: str) -> int:
    """Approximate tokens without requiring an external tokenizer.

    If tiktoken is installed locally, use it. Otherwise use a conservative
    mixed character/word heuristic. This is for pre-run estimates only; actual
    billing uses the API response usage object.
    """
    text = str(text or "")
    try:
        import tiktoken  # type: ignore
        try:
            enc = tiktoken.encoding_for_model(llm_config.MODEL)
        except Exception:
            enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        words = len(re.findall(r"\S+", text))
        chars = len(text)
        return max(1, int(max(chars / 4.0, words * 1.33)))


def build_chat_messages(prompt: str) -> List[Dict[str, str]]:
    """Build the exact chat message structure sent to the API.

    Keeping the long, stable instruction block before the case-specific text
    gives repeated calls a shared prompt prefix, improving the chance that the
    platform can cache input tokens across cases.
    """
    return [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user", "content": prompt},
    ]


def estimate_prompt_tokens_from_messages(messages: Sequence[Dict[str, str]]) -> int:
    # Small per-message overhead approximates Chat Completions formatting.
    return sum(_rough_token_count(m.get("content", "")) + 4 for m in messages) + 3


def estimate_prompt_tokens_for_case(case: "Case", report_mode: str) -> int:
    return estimate_prompt_tokens_from_messages(build_chat_messages(build_user_prompt(case, report_mode)))


def estimate_static_cached_prefix_tokens(report_mode: str) -> int:
    """Approximate the stable prefix that can be cached within one modality.

    This excludes case_id, index_side, and report text. It is an optimistic
    estimate; actual cached_tokens are taken only from API usage.
    """
    dummy = Case(case_id="__CASE_ID__", preop_mri="", path_report="", index_side="__INDEX_SIDE__")
    prompt = build_user_prompt(dummy, report_mode)
    marker = "CASE\n"
    if marker in prompt:
        prompt = prompt.split(marker, 1)[0] + marker
    return estimate_prompt_tokens_from_messages(build_chat_messages(prompt))


def summarize_apriori_cost_estimate(
    prompt_token_counts: Sequence[int],
    report_modes: Sequence[str],
    max_completion_tokens: int = llm_config.MAX_TOKENS,
    assume_static_prefix_cache: bool = True,
) -> Dict[str, Any]:
    """Summarize conservative and cache-aware a-priori cost estimates."""
    prompt_counts = [int(x) for x in prompt_token_counts]
    modes = [str(x) for x in report_modes]
    if len(prompt_counts) != len(modes):
        raise ValueError("prompt_token_counts and report_modes must have the same length.")

    total_prompt_tokens = sum(prompt_counts)
    total_completion_cap_tokens = len(prompt_counts) * int(max_completion_tokens)
    no_cache = estimate_cost_from_token_counts(
        prompt_tokens=total_prompt_tokens,
        completion_tokens=total_completion_cap_tokens,
        cached_tokens=0,
    )

    estimated_cached_tokens = 0
    if assume_static_prefix_cache and prompt_counts:
        seen_mode_counts: Dict[str, int] = {}
        static_prefix_by_mode = {mode: estimate_static_cached_prefix_tokens(mode) for mode in sorted(set(modes))}
        for prompt_tokens, mode in zip(prompt_counts, modes):
            seen_mode_counts[mode] = seen_mode_counts.get(mode, 0) + 1
            if seen_mode_counts[mode] > 1:
                estimated_cached_tokens += min(prompt_tokens, static_prefix_by_mode.get(mode, 0))

    cache_aware = estimate_cost_from_token_counts(
        prompt_tokens=total_prompt_tokens,
        completion_tokens=total_completion_cap_tokens,
        cached_tokens=estimated_cached_tokens,
    )

    return {
        "model": llm_config.MODEL,
        "deployment": llm_config.DEPLOYMENT,
        "pricing_label": llm_config.PRICING_LABEL,
        "api_version": llm_config.API_VERSION,
        "n_calls": len(prompt_counts),
        "estimated_prompt_tokens": total_prompt_tokens,
        "estimated_completion_cap_tokens": total_completion_cap_tokens,
        "max_completion_tokens_per_call": int(max_completion_tokens),
        "no_cache_estimated_cost_usd": no_cache["estimated_cost_usd"],
        "cache_aware_estimated_cost_usd": cache_aware["estimated_cost_usd"],
        "cache_aware_estimated_cached_tokens": estimated_cached_tokens,
        "cache_aware_estimated_cache_savings_usd": cache_aware["estimated_cache_savings_usd"],
        "price_per_1M_input_tokens": llm_config.PRICE_PER_1M_INPUT_TOKENS,
        "price_per_1M_cached_input_tokens": llm_config.PRICE_PER_1M_CACHED_INPUT_TOKENS,
        "price_per_1M_output_tokens": llm_config.PRICE_PER_1M_OUTPUT_TOKENS,
    }


def print_apriori_cost_estimate_report(estimate: Dict[str, Any], label: str = "planned pipeline") -> None:
    print("\n[A-PRIORI LLM TOKEN / COST ESTIMATE]")
    print(f"scope:                         {label}")
    print(f"model:                         {estimate['model']}")
    print(f"pricing_per_1M:                input=${estimate['price_per_1M_input_tokens']:.4f} cached_input=${estimate['price_per_1M_cached_input_tokens']:.4f} output=${estimate['price_per_1M_output_tokens']:.4f}")
    print(f"planned_llm_calls:             {estimate['n_calls']}")
    print(f"estimated_prompt_tokens:       {estimate['estimated_prompt_tokens']}")
    print(f"max_completion_tokens_per_call:{estimate['max_completion_tokens_per_call']}")
    print(f"completion_token_cap_total:    {estimate['estimated_completion_cap_tokens']}")
    print(f"estimated_cost_no_cache:       ${estimate['no_cache_estimated_cost_usd']:.8f}")
    print(f"cache_aware_estimated_cost:    ${estimate['cache_aware_estimated_cost_usd']:.8f}")
    print(f"cache_aware_cached_tokens:     {estimate['cache_aware_estimated_cached_tokens']}")
    print(f"cache_aware_cache_savings:     ${estimate['cache_aware_estimated_cache_savings_usd']:.8f}")
    print("[A-PRIORI NOTE] This uses the full prompts that will be sent, plus the configured completion-token cap. Actual cost is recomputed from API usage after each call.")


def confirm_cost_estimate_or_exit(estimate: Dict[str, Any], assume_yes: bool = False) -> None:
    if assume_yes:
        print("[CONFIRM] --yes supplied; continuing without interactive cost confirmation.")
        return
    if not sys.__stdin__.isatty():
        raise RuntimeError(
            "Interactive cost confirmation is required but stdin is not a TTY. "
            "Rerun with --yes after reviewing the printed a-priori estimate."
        )
    prompt = "Continue with LLM extraction calls? [yes/no]: "
    sys.__stdout__.write(prompt)
    sys.__stdout__.flush()
    reply = sys.__stdin__.readline().strip()
    if is_negative_response(reply):
        print("[ABORT] User declined; exiting before LLM extraction calls.")
        raise SystemExit(1)
    if not is_affirmative_response(reply):
        print("[ABORT] Unrecognized response; type yes/y or no/n. Exiting before LLM extraction calls.")
        raise SystemExit(1)
