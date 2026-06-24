"""Token usage cost tracking and a-priori cost estimation."""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .. import config
from ..io_atomic import atomic_write_json
from ..models import Case, RunConfig
from ..prompts.system import SYSTEM_MSG
from ..prompts.templates import build_user_prompt


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


class CostTracker:
    """Session-scoped token and cost accumulator."""

    def __init__(self) -> None:
        self._data = empty_cost_tracker()

    def reset(self) -> None:
        self._data = empty_cost_tracker()

    def to_dict(self) -> Dict[str, Any]:
        return dict(self._data)

    def update_from_usage(self, usage: Dict[str, Any]) -> Dict[str, Any]:
        cost_info = estimate_cost_from_usage(usage)
        self._data["calls"] += 1
        self._data["prompt_tokens"] += int(cost_info["prompt_tokens"])
        self._data["cached_tokens"] += int(cost_info["cached_tokens"])
        self._data["uncached_prompt_tokens"] += int(cost_info["uncached_prompt_tokens"])
        self._data["completion_tokens"] += int(cost_info["completion_tokens"])
        self._data["reasoning_tokens"] += int(cost_info["reasoning_tokens"])
        self._data["total_tokens"] += int(cost_info["total_tokens"])
        self._data["estimated_cost_usd"] += float(cost_info["estimated_cost_usd"])
        self._data["estimated_cache_savings_usd"] += float(cost_info["estimated_cache_savings_usd"])
        return cost_info

    def print_cumulative_report(self) -> None:
        d = self._data
        print("\n[CUMULATIVE TOKEN / COST REPORT]")
        print(f"calls:                    {d['calls']}")
        print(f"prompt_tokens:            {d['prompt_tokens']}")
        print(f"cached_tokens:            {d['cached_tokens']}")
        print(f"uncached_prompt_tokens:   {d['uncached_prompt_tokens']}")
        print(f"completion_tokens:        {d['completion_tokens']}")
        print(f"reasoning_tokens:         {d['reasoning_tokens']}")
        print(f"total_tokens:             {d['total_tokens']}")
        print(f"estimated_total_cost_usd: ${d['estimated_cost_usd']:.8f}")
        print(f"estimated_cache_savings:  ${d['estimated_cache_savings_usd']:.8f}")


def estimate_cost_from_usage(usage: Dict[str, Any]) -> Dict[str, Any]:
    prompt_tokens = usage.get("prompt_tokens", 0) or 0
    completion_tokens = usage.get("completion_tokens", 0) or 0
    total_tokens = usage.get("total_tokens", 0) or 0

    prompt_details = usage.get("prompt_tokens_details", {}) or {}
    completion_details = usage.get("completion_tokens_details", {}) or {}

    cached_tokens = prompt_details.get("cached_tokens", 0) or 0
    reasoning_tokens = completion_details.get("reasoning_tokens", 0) or 0
    uncached_prompt_tokens = max(prompt_tokens - cached_tokens, 0)

    input_cost = (uncached_prompt_tokens / 1_000_000) * config.PRICE_PER_1M_INPUT_TOKENS
    cached_input_cost = (cached_tokens / 1_000_000) * config.PRICE_PER_1M_CACHED_INPUT_TOKENS
    output_cost = (completion_tokens / 1_000_000) * config.PRICE_PER_1M_OUTPUT_TOKENS
    estimated_cost = input_cost + cached_input_cost + output_cost

    no_cache_input_cost = (prompt_tokens / 1_000_000) * config.PRICE_PER_1M_INPUT_TOKENS
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


def merge_cost_trackers(prior: Optional[Dict[str, Any]], session: Dict[str, Any]) -> Dict[str, Any]:
    if not prior:
        return dict(session)
    merged = empty_cost_tracker()
    int_keys = {
        "calls",
        "prompt_tokens",
        "cached_tokens",
        "uncached_prompt_tokens",
        "completion_tokens",
        "reasoning_tokens",
        "total_tokens",
    }
    float_keys = {"estimated_cost_usd", "estimated_cache_savings_usd"}
    for k in int_keys:
        merged[k] = int(prior.get(k, 0)) + int(session.get(k, 0))
    for k in float_keys:
        merged[k] = float(prior.get(k, 0.0)) + float(session.get(k, 0.0))
    return merged


def load_cost_tracker_snapshot(path: str) -> Optional[Dict[str, Any]]:
    import json
    import os

    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None
    if isinstance(data, dict) and isinstance(data.get("cumulative"), dict):
        return data["cumulative"]
    if isinstance(data, dict):
        return data
    return None


def save_cumulative_report_json(
    path: str,
    tracker: CostTracker,
    prior: Optional[Dict[str, Any]] = None,
) -> None:
    session = tracker.to_dict()
    cumulative = merge_cost_trackers(prior, session)
    payload = {
        "resume_script_version": config.RESUME_SCRIPT_VERSION,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "cumulative": cumulative,
        "session": session,
        "prior_sessions": prior or empty_cost_tracker(),
    }
    atomic_write_json(path, payload)


def _estimate_tokens_for_text(text: str) -> int:
    try:
        import tiktoken  # type: ignore

        try:
            enc = tiktoken.encoding_for_model(config.DEPLOYMENT)
        except Exception:
            enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return int(math.ceil(len(text) / 4.0))


def _messages_text_for_token_estimate(user_prompt: str) -> str:
    return f"system:\n{SYSTEM_MSG}\nuser:\n{user_prompt}"


def _common_prefix_len(strings: List[str]) -> int:
    if not strings:
        return 0
    prefix = strings[0]
    for s in strings[1:]:
        max_i = min(len(prefix), len(s))
        i = 0
        while i < max_i and prefix[i] == s[i]:
            i += 1
        prefix = prefix[:i]
        if not prefix:
            break
    return len(prefix)


def estimate_apriori_pipeline_cost(
    training_block: str,
    test_cases_with_idxs: List[Tuple[int, Case]],
    modality: str,
) -> Dict[str, Any]:
    prompt_texts = [
        _messages_text_for_token_estimate(build_user_prompt(training_block, c, row_index=idx, modality=modality))
        for idx, c in test_cases_with_idxs
    ]
    prompt_tokens_by_case = [_estimate_tokens_for_text(t) for t in prompt_texts]
    total_prompt_tokens = int(sum(prompt_tokens_by_case))

    common_prefix_chars = _common_prefix_len(prompt_texts)
    common_prefix_text = prompt_texts[0][:common_prefix_chars] if prompt_texts else ""
    common_prefix_tokens = _estimate_tokens_for_text(common_prefix_text) if prompt_texts else 0

    n_calls = len(prompt_texts)
    estimated_cached_tokens = int(common_prefix_tokens * max(n_calls - 1, 0))
    estimated_cached_tokens = min(estimated_cached_tokens, total_prompt_tokens)
    estimated_uncached_prompt_tokens = max(total_prompt_tokens - estimated_cached_tokens, 0)
    estimated_completion_tokens_upper = int(n_calls * config.MAX_TOKENS)

    no_cache_input_cost = (total_prompt_tokens / 1_000_000) * config.PRICE_PER_1M_INPUT_TOKENS
    cache_adjusted_input_cost = (
        (estimated_uncached_prompt_tokens / 1_000_000) * config.PRICE_PER_1M_INPUT_TOKENS
        + (estimated_cached_tokens / 1_000_000) * config.PRICE_PER_1M_CACHED_INPUT_TOKENS
    )
    output_cost_upper = (estimated_completion_tokens_upper / 1_000_000) * config.PRICE_PER_1M_OUTPUT_TOKENS

    return {
        "n_calls": n_calls,
        "prompt_tokens_estimated_total": total_prompt_tokens,
        "prompt_tokens_estimated_min": min(prompt_tokens_by_case) if prompt_tokens_by_case else 0,
        "prompt_tokens_estimated_max": max(prompt_tokens_by_case) if prompt_tokens_by_case else 0,
        "prompt_tokens_estimated_mean": float(np.mean(prompt_tokens_by_case)) if prompt_tokens_by_case else 0.0,
        "common_prefix_tokens_estimated": common_prefix_tokens,
        "cached_prompt_tokens_estimated": estimated_cached_tokens,
        "uncached_prompt_tokens_estimated": estimated_uncached_prompt_tokens,
        "completion_tokens_upper_estimated": estimated_completion_tokens_upper,
        "estimated_no_cache_cost_usd_upper": no_cache_input_cost + output_cost_upper,
        "estimated_cache_adjusted_cost_usd_upper": cache_adjusted_input_cost + output_cost_upper,
        "estimated_cache_savings_usd": max(no_cache_input_cost - cache_adjusted_input_cost, 0.0),
    }


def summarize_apriori_costs(run_configs: List[RunConfig]) -> Dict[str, Any]:
    total = {
        "n_runs": len(run_configs),
        "n_calls": 0,
        "prompt_tokens_estimated_total": 0,
        "cached_prompt_tokens_estimated": 0,
        "uncached_prompt_tokens_estimated": 0,
        "completion_tokens_upper_estimated": 0,
        "estimated_no_cache_cost_usd_upper": 0.0,
        "estimated_cache_adjusted_cost_usd_upper": 0.0,
        "estimated_cache_savings_usd": 0.0,
    }
    for rc in run_configs:
        r = rc.apriori_cost
        for key in [
            "n_calls",
            "prompt_tokens_estimated_total",
            "cached_prompt_tokens_estimated",
            "uncached_prompt_tokens_estimated",
            "completion_tokens_upper_estimated",
        ]:
            total[key] += int(r[key])
        for key in [
            "estimated_no_cache_cost_usd_upper",
            "estimated_cache_adjusted_cost_usd_upper",
            "estimated_cache_savings_usd",
        ]:
            total[key] += float(r[key])
    return total


def print_apriori_cost_report(run_configs: List[RunConfig]) -> None:
    total = summarize_apriori_costs(run_configs)
    print("\n[A-PRIORI TOKEN / COST ESTIMATE ACROSS ALL REQUESTED RUNS]")
    print(f"model/deployment:                         {config.DEPLOYMENT}")
    print(f"planned shotsettier runs:                {total['n_runs']}")
    print(f"planned prediction calls:                 {total['n_calls']}")
    print(f"estimated prompt tokens total:            {total['prompt_tokens_estimated_total']}")
    print(f"estimated cached prompt tokens:           {total['cached_prompt_tokens_estimated']}")
    print(f"estimated uncached prompt tokens:         {total['uncached_prompt_tokens_estimated']}")
    print(f"completion-token upper bound:             {total['completion_tokens_upper_estimated']} ({config.MAX_TOKENS} max_completion_tokens/call)")
    print(f"estimated no-cache cost upper bound:      ${total['estimated_no_cache_cost_usd_upper']:.8f}")
    print(f"estimated cache-adjusted cost upper bound:${total['estimated_cache_adjusted_cost_usd_upper']:.8f}")
    print(f"estimated cache savings:                  ${total['estimated_cache_savings_usd']:.8f}")
    print("NOTE: This is a local estimate only. Final cost uses API response usage fields.")
    print("\n[RUN BREAKDOWN]")
    for rc in run_configs:
        r = rc.apriori_cost
        print(
            f"  {rc.shotset_name}/{rc.modality}: "
            f"calls={r['n_calls']} "
            f"prompt_mean={r['prompt_tokens_estimated_mean']:.1f} "
            f"prompt_min={r['prompt_tokens_estimated_min']} "
            f"prompt_max={r['prompt_tokens_estimated_max']} "
            f"skipped_missing_mri={len(rc.skipped_missing_mri)} "
            f"est_cost_cache_adj=${r['estimated_cache_adjusted_cost_usd_upper']:.8f}"
        )


def confirm_before_full_run(
    run_configs: List[RunConfig],
    assume_yes: bool = False,
    *,
    resume: bool = True,
    skip_completed_configs: bool = True,
    force_rerun_cases: bool = False,
) -> None:
    from ..checkpoint.resume import print_resume_plan, summarize_resume_plan

    print_apriori_cost_report(run_configs)
    resume_summary = summarize_resume_plan(
        run_configs,
        resume=resume,
        skip_completed_configs=skip_completed_configs,
        force_rerun_cases=force_rerun_cases,
    )
    print_resume_plan(resume_summary)
    if assume_yes:
        print("[CONFIRM] --yes supplied; continuing without interactive prompt.")
        return
    answer = input("\nContinue with the full 3-tier LLM pipeline? Type 'yes' to proceed: ").strip().lower()
    if answer not in {"yes", "y"}:
        print("[ABORT] User did not confirm. No prediction calls were made.")
        raise SystemExit(0)
