"""API module with backward-compatible module-level shims."""

from __future__ import annotations

from typing import Dict, Optional

from .. import config
from .client import SecureGPTClient
from .cost import (
    CostTracker,
    confirm_before_full_run,
    empty_cost_tracker,
    estimate_apriori_pipeline_cost,
    estimate_cost_from_usage,
    load_cost_tracker_snapshot,
    merge_cost_trackers,
    print_apriori_cost_report,
    save_cumulative_report_json,
    summarize_apriori_costs,
)

_default_client: Optional[SecureGPTClient] = None
_default_tracker = CostTracker()

# Backward-compatible global dict view synced from default tracker
COST_TRACKER: Dict = _default_tracker.to_dict()

API_KEY: Optional[str] = None
URL: Optional[str] = None
HEADERS: Dict[str, str] = {}


def _sync_cost_tracker_dict() -> None:
    global COST_TRACKER
    COST_TRACKER = _default_tracker.to_dict()


def get_default_client() -> SecureGPTClient:
    if _default_client is None:
        raise RuntimeError("API is not configured. Call configure_api(...) first.")
    return _default_client


def get_default_tracker() -> CostTracker:
    return _default_tracker


def configure_api(env_path: str, deployment: str, api_version: str) -> None:
    global _default_client, API_KEY, URL, HEADERS
    _default_client = SecureGPTClient(
        env_path=env_path,
        deployment=deployment,
        api_version=api_version,
        cost_tracker=_default_tracker,
    )
    URL = _default_client.url
    HEADERS = _default_client.headers
    API_KEY = _default_client.headers.get("api-key")
    config.DEPLOYMENT = deployment
    config.API_VERSION = api_version


def reset_cost_tracker() -> None:
    _default_tracker.reset()
    _sync_cost_tracker_dict()


def update_cost_tracker(cost_info: Dict) -> None:
    _default_tracker._data["calls"] += 1
    _default_tracker._data["prompt_tokens"] += int(cost_info["prompt_tokens"])
    _default_tracker._data["cached_tokens"] += int(cost_info["cached_tokens"])
    _default_tracker._data["uncached_prompt_tokens"] += int(cost_info["uncached_prompt_tokens"])
    _default_tracker._data["completion_tokens"] += int(cost_info["completion_tokens"])
    _default_tracker._data["reasoning_tokens"] += int(cost_info["reasoning_tokens"])
    _default_tracker._data["total_tokens"] += int(cost_info["total_tokens"])
    _default_tracker._data["estimated_cost_usd"] += float(cost_info["estimated_cost_usd"])
    _default_tracker._data["estimated_cache_savings_usd"] += float(cost_info["estimated_cache_savings_usd"])
    _sync_cost_tracker_dict()


def print_cumulative_report() -> None:
    _default_tracker.print_cumulative_report()


def call_securegpt_chat(prompt: str, max_completion_tokens: int = config.MAX_TOKENS) -> str:
    content = get_default_client().chat(prompt, max_completion_tokens=max_completion_tokens)
    _sync_cost_tracker_dict()
    return content


def preflight_check() -> None:
    get_default_client().preflight_check()
    _sync_cost_tracker_dict()


def save_cumulative_report_json_shim(path: str, prior: Optional[Dict] = None) -> None:
    save_cumulative_report_json(path, _default_tracker, prior=prior)


__all__ = [
    "SecureGPTClient",
    "CostTracker",
    "configure_api",
    "call_securegpt_chat",
    "preflight_check",
    "reset_cost_tracker",
    "update_cost_tracker",
    "print_cumulative_report",
    "estimate_cost_from_usage",
    "merge_cost_trackers",
    "load_cost_tracker_snapshot",
    "estimate_apriori_pipeline_cost",
    "summarize_apriori_costs",
    "print_apriori_cost_report",
    "confirm_before_full_run",
    "empty_cost_tracker",
    "COST_TRACKER",
    "API_KEY",
    "URL",
    "HEADERS",
    "get_default_client",
    "get_default_tracker",
    "save_cumulative_report_json_shim",
]
