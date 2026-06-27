"""Shared helpers for comparing a-priori LLM cost estimates to post-run actuals."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

APPROACH1_APRIORI = "approach1"
APPROACH2_APRIORI = "approach2"

APRIORI_SESSION_FILENAME = "llm_cost_estimate_apriori.json"
APRIORI_INITIAL_FILENAME = "llm_cost_estimate_apriori_initial.json"

COST_TOKENS_PLOT = "cost_estimate_vs_actual_tokens.png"
COST_USD_PLOT = "cost_estimate_vs_actual_usd.png"
COST_BY_CONFIG_PLOT = "cost_estimate_vs_actual_by_config.png"


def extract_actual_cumulative(cost_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Normalize post-run cost JSON to a flat cumulative tracker dict."""
    if not cost_data or not isinstance(cost_data, dict):
        return {}
    source = cost_data.get("cumulative") if isinstance(cost_data.get("cumulative"), dict) else cost_data
    if not isinstance(source, dict):
        return {}
    return {
        "calls": int(source.get("calls", 0) or 0),
        "prompt_tokens": int(source.get("prompt_tokens", 0) or 0),
        "cached_tokens": int(source.get("cached_tokens", 0) or 0),
        "uncached_prompt_tokens": int(source.get("uncached_prompt_tokens", 0) or 0),
        "completion_tokens": int(source.get("completion_tokens", 0) or 0),
        "reasoning_tokens": int(source.get("reasoning_tokens", 0) or 0),
        "total_tokens": int(source.get("total_tokens", 0) or 0),
        "estimated_cost_usd": float(source.get("estimated_cost_usd", 0.0) or 0.0),
        "estimated_cache_savings_usd": float(source.get("estimated_cache_savings_usd", 0.0) or 0.0),
    }


def is_partial_resume_apriori_estimate(apriori: Optional[Dict[str, Any]]) -> bool:
    """True when an Approach 2 a-priori block covers remaining resume work only."""
    if not apriori or not isinstance(apriori, dict):
        return False
    return int(apriori.get("n_completed_splits_skipped_in_estimate", 0) or 0) > 0 or int(
        apriori.get("n_calls_skipped_existing_checkpoints", 0) or 0
    ) > 0


def _read_json_if_exists(path: str) -> Dict[str, Any]:
    if not path or not os.path.isfile(path):
        return {}
    try:
        import json

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def load_approach2_apriori_for_comparison(out_dir: str) -> Dict[str, Any]:
    """Load the immutable initial full-pipeline a-priori snapshot for report comparisons."""
    initial = _read_json_if_exists(os.path.join(out_dir, APRIORI_INITIAL_FILENAME))
    if initial:
        return initial
    session = _read_json_if_exists(os.path.join(out_dir, APRIORI_SESSION_FILENAME))
    if session and not is_partial_resume_apriori_estimate(session):
        return session
    return {}


def backfill_approach2_initial_apriori_if_needed(out_dir: str, csv_path: Optional[str] = None) -> Optional[str]:
    """Create the initial a-priori snapshot from saved manifests when missing."""
    initial_path = os.path.join(out_dir, APRIORI_INITIAL_FILENAME)
    if os.path.isfile(initial_path):
        return initial_path
    if not csv_path:
        return None
    from approach2.api.cost import ensure_initial_apriori_cost_estimate_json
    from approach2.cli import estimate_full_pipeline_apriori_for_out_dir

    estimate = estimate_full_pipeline_apriori_for_out_dir(out_dir, csv_path)
    return ensure_initial_apriori_cost_estimate_json(out_dir, estimate)


def normalize_apriori_estimate(
    apriori: Optional[Dict[str, Any]],
    *,
    flavor: str,
) -> Dict[str, Any]:
    """Map approach-specific a-priori JSON to a common schema."""
    if not apriori or not isinstance(apriori, dict):
        return {}
    if flavor == APPROACH1_APRIORI:
        prompt_tokens = int(apriori.get("prompt_tokens_estimated_total", 0) or 0)
        completion_cap = int(apriori.get("completion_tokens_upper_estimated", 0) or 0)
        return {
            "n_calls": int(apriori.get("n_calls", 0) or 0),
            "prompt_tokens": prompt_tokens,
            "cached_prompt_tokens": int(apriori.get("cached_prompt_tokens_estimated", 0) or 0),
            "uncached_prompt_tokens": int(apriori.get("uncached_prompt_tokens_estimated", 0) or 0),
            "completion_tokens_cap": completion_cap,
            "total_tokens_upper": prompt_tokens + completion_cap,
            "no_cache_cost_usd": float(apriori.get("estimated_no_cache_cost_usd_upper", 0.0) or 0.0),
            "cache_aware_cost_usd": float(apriori.get("estimated_cache_adjusted_cost_usd_upper", 0.0) or 0.0),
            "cache_savings_usd": float(apriori.get("estimated_cache_savings_usd", 0.0) or 0.0),
        }
    if flavor == APPROACH2_APRIORI:
        prompt_tokens = int(apriori.get("estimated_prompt_tokens", 0) or 0)
        completion_cap = int(apriori.get("estimated_completion_cap_tokens", 0) or 0)
        return {
            "n_calls": int(apriori.get("n_calls", 0) or 0),
            "prompt_tokens": prompt_tokens,
            "cached_prompt_tokens": int(apriori.get("cache_aware_estimated_cached_tokens", 0) or 0),
            "uncached_prompt_tokens": max(
                prompt_tokens - int(apriori.get("cache_aware_estimated_cached_tokens", 0) or 0),
                0,
            ),
            "completion_tokens_cap": completion_cap,
            "total_tokens_upper": prompt_tokens + completion_cap,
            "no_cache_cost_usd": float(apriori.get("no_cache_estimated_cost_usd", 0.0) or 0.0),
            "cache_aware_cost_usd": float(apriori.get("cache_aware_estimated_cost_usd", 0.0) or 0.0),
            "cache_savings_usd": float(apriori.get("cache_aware_estimated_cache_savings_usd", 0.0) or 0.0),
        }
    raise ValueError(f"Unknown apriori flavor: {flavor}")


def aggregate_apriori_from_run_configs(run_config_payloads: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Sum Approach 1 per-config apriori_cost blocks from run_config.json payloads."""
    totals = {
        "n_calls": 0,
        "prompt_tokens_estimated_total": 0,
        "cached_prompt_tokens_estimated": 0,
        "uncached_prompt_tokens_estimated": 0,
        "completion_tokens_upper_estimated": 0,
        "estimated_no_cache_cost_usd_upper": 0.0,
        "estimated_cache_adjusted_cost_usd_upper": 0.0,
        "estimated_cache_savings_usd": 0.0,
    }
    for payload in run_config_payloads:
        block = payload.get("apriori_cost") if isinstance(payload, dict) else None
        if not isinstance(block, dict):
            continue
        totals["n_calls"] += int(block.get("n_calls", 0) or 0)
        totals["prompt_tokens_estimated_total"] += int(block.get("prompt_tokens_estimated_total", 0) or 0)
        totals["cached_prompt_tokens_estimated"] += int(block.get("cached_prompt_tokens_estimated", 0) or 0)
        totals["uncached_prompt_tokens_estimated"] += int(block.get("uncached_prompt_tokens_estimated", 0) or 0)
        totals["completion_tokens_upper_estimated"] += int(block.get("completion_tokens_upper_estimated", 0) or 0)
        totals["estimated_no_cache_cost_usd_upper"] += float(block.get("estimated_no_cache_cost_usd_upper", 0.0) or 0.0)
        totals["estimated_cache_adjusted_cost_usd_upper"] += float(
            block.get("estimated_cache_adjusted_cost_usd_upper", 0.0) or 0.0
        )
        totals["estimated_cache_savings_usd"] += float(block.get("estimated_cache_savings_usd", 0.0) or 0.0)
    return totals


def build_cost_comparison_summary_df(
    apriori_norm: Dict[str, Any],
    actual_norm: Dict[str, Any],
) -> pd.DataFrame:
    """Pipeline-level table comparing a-priori estimates to post-run actuals."""
    if not apriori_norm and not actual_norm:
        return pd.DataFrame()
    rows = [
        {
            "metric": "API calls",
            "apriori_no_cache": apriori_norm.get("n_calls"),
            "apriori_cache_aware": apriori_norm.get("n_calls"),
            "actual": actual_norm.get("calls"),
        },
        {
            "metric": "Prompt tokens",
            "apriori_no_cache": apriori_norm.get("prompt_tokens"),
            "apriori_cache_aware": apriori_norm.get("prompt_tokens"),
            "actual": actual_norm.get("prompt_tokens"),
        },
        {
            "metric": "Cached prompt tokens",
            "apriori_no_cache": 0,
            "apriori_cache_aware": apriori_norm.get("cached_prompt_tokens"),
            "actual": actual_norm.get("cached_tokens"),
        },
        {
            "metric": "Uncached prompt tokens",
            "apriori_no_cache": apriori_norm.get("prompt_tokens"),
            "apriori_cache_aware": apriori_norm.get("uncached_prompt_tokens"),
            "actual": actual_norm.get("uncached_prompt_tokens"),
        },
        {
            "metric": "Completion tokens",
            "apriori_no_cache": apriori_norm.get("completion_tokens_cap"),
            "apriori_cache_aware": apriori_norm.get("completion_tokens_cap"),
            "actual": actual_norm.get("completion_tokens"),
        },
        {
            "metric": "Reasoning tokens",
            "apriori_no_cache": None,
            "apriori_cache_aware": None,
            "actual": actual_norm.get("reasoning_tokens"),
        },
        {
            "metric": "Total tokens",
            "apriori_no_cache": apriori_norm.get("total_tokens_upper"),
            "apriori_cache_aware": apriori_norm.get("total_tokens_upper"),
            "actual": actual_norm.get("total_tokens"),
        },
        {
            "metric": "Estimated cost (USD)",
            "apriori_no_cache": apriori_norm.get("no_cache_cost_usd"),
            "apriori_cache_aware": apriori_norm.get("cache_aware_cost_usd"),
            "actual": actual_norm.get("estimated_cost_usd"),
        },
        {
            "metric": "Cache savings (USD)",
            "apriori_no_cache": 0.0,
            "apriori_cache_aware": apriori_norm.get("cache_savings_usd"),
            "actual": actual_norm.get("estimated_cache_savings_usd"),
        },
    ]
    df = pd.DataFrame(rows)
    est = apriori_norm.get("cache_aware_cost_usd")
    act = actual_norm.get("estimated_cost_usd")
    if est is not None and act is not None:
        delta = float(act) - float(est)
        pct = (delta / float(est) * 100.0) if float(est) else None
        df = pd.concat(
            [
                df,
                pd.DataFrame([{
                    "metric": "Actual minus cache-aware estimate (USD)",
                    "apriori_no_cache": None,
                    "apriori_cache_aware": None,
                    "actual": delta,
                }, {
                    "metric": "Actual vs cache-aware estimate (%)",
                    "apriori_no_cache": None,
                    "apriori_cache_aware": None,
                    "actual": pct,
                }]),
            ],
            ignore_index=True,
        )
    return df


def build_per_config_cost_comparison_df(
    config_rows: Sequence[Tuple[str, Dict[str, Any], Dict[str, Any]]],
    *,
    flavor: str,
) -> pd.DataFrame:
    """Per-run breakdown for Approach 1 shotset/modality folders."""
    rows: List[Dict[str, Any]] = []
    for label, apriori_raw, actual_raw in config_rows:
        apriori = normalize_apriori_estimate(apriori_raw, flavor=flavor)
        actual = extract_actual_cumulative(actual_raw)
        if not apriori and not actual:
            continue
        est = apriori.get("cache_aware_cost_usd")
        act = actual.get("estimated_cost_usd")
        delta = (float(act) - float(est)) if est is not None and act is not None else None
        rows.append({
            "configuration": label,
            "apriori_calls": apriori.get("n_calls"),
            "actual_calls": actual.get("calls"),
            "apriori_prompt_tokens": apriori.get("prompt_tokens"),
            "actual_prompt_tokens": actual.get("prompt_tokens"),
            "apriori_completion_cap": apriori.get("completion_tokens_cap"),
            "actual_completion_tokens": actual.get("completion_tokens"),
            "apriori_cache_aware_cost_usd": est,
            "actual_cost_usd": act,
            "cost_delta_usd": delta,
        })
    return pd.DataFrame(rows)


def _grouped_bar_plot(
    out_path: str,
    *,
    title: str,
    ylabel: str,
    categories: Sequence[str],
    series: Sequence[Tuple[str, Sequence[float]]],
) -> bool:
    if not categories or not any(any(v is not None for v in vals) for _, vals in series):
        return False
    x = range(len(categories))
    width = 0.8 / max(len(series), 1)
    plt.figure(figsize=(8, 4.5))
    for i, (label, values) in enumerate(series):
        offsets = [xi + (i - (len(series) - 1) / 2) * width for xi in x]
        plt.bar(offsets, values, width=width, label=label)
    plt.xticks(list(x), list(categories), rotation=15, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()
    return True


def plot_cost_tokens_comparison(
    out_path: str,
    apriori_norm: Dict[str, Any],
    actual_norm: Dict[str, Any],
    *,
    title: str = "Token usage: estimate vs actual",
) -> bool:
    categories = ["Prompt", "Completion", "Total"]
    est_prompt = float(apriori_norm.get("prompt_tokens", 0) or 0)
    est_completion = float(apriori_norm.get("completion_tokens_cap", 0) or 0)
    est_total = float(apriori_norm.get("total_tokens_upper", 0) or 0)
    act_prompt = float(actual_norm.get("prompt_tokens", 0) or 0)
    act_completion = float(actual_norm.get("completion_tokens", 0) or 0)
    act_total = float(actual_norm.get("total_tokens", 0) or 0)
    return _grouped_bar_plot(
        out_path,
        title=title,
        ylabel="Tokens",
        categories=categories,
        series=[
            ("A priori (cache-aware upper bound)", [est_prompt, est_completion, est_total]),
            ("Post-run actual", [act_prompt, act_completion, act_total]),
        ],
    )


def plot_cost_usd_comparison(
    out_path: str,
    apriori_norm: Dict[str, Any],
    actual_norm: Dict[str, Any],
    *,
    title: str = "Cost (USD): estimate vs actual",
) -> bool:
    categories = ["No-cache upper bound", "Cache-aware upper bound", "Post-run actual"]
    values_est = [
        float(apriori_norm.get("no_cache_cost_usd", 0.0) or 0.0),
        float(apriori_norm.get("cache_aware_cost_usd", 0.0) or 0.0),
        float(actual_norm.get("estimated_cost_usd", 0.0) or 0.0),
    ]
    if not any(values_est):
        return False
    plt.figure(figsize=(7.5, 4.5))
    colors = ["#94a3b8", "#60a5fa", "#16a34a"]
    plt.bar(categories, values_est, color=colors[: len(categories)])
    plt.ylabel("USD")
    plt.title(title)
    plt.xticks(rotation=12, ha="right")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()
    return True


def plot_cost_by_config_comparison(
    out_path: str,
    per_config_df: pd.DataFrame,
    *,
    title: str = "Cache-aware cost estimate vs actual by configuration",
) -> bool:
    if per_config_df is None or len(per_config_df) == 0:
        return False
    labels = per_config_df["configuration"].astype(str).tolist()
    est = per_config_df["apriori_cache_aware_cost_usd"].astype(float).tolist()
    act = per_config_df["actual_cost_usd"].astype(float).tolist()
    x = range(len(labels))
    width = 0.35
    plt.figure(figsize=(max(8, len(labels) * 1.2), 4.5))
    plt.bar([i - width / 2 for i in x], est, width=width, label="A priori (cache-aware)")
    plt.bar([i + width / 2 for i in x], act, width=width, label="Post-run actual")
    plt.xticks(list(x), labels, rotation=25, ha="right")
    plt.ylabel("USD")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()
    return True


def generate_cost_comparison_plots(
    out_dir: str,
    apriori_norm: Dict[str, Any],
    actual_norm: Dict[str, Any],
    *,
    per_config_df: Optional[pd.DataFrame] = None,
    plot_subdir: str = "report_plots",
) -> List[str]:
    """Write comparison PNGs and return absolute paths."""
    if not apriori_norm and not actual_norm:
        return []
    plot_dir = os.path.join(out_dir, plot_subdir)
    paths: List[str] = []
    tokens_path = os.path.join(plot_dir, COST_TOKENS_PLOT)
    if plot_cost_tokens_comparison(tokens_path, apriori_norm, actual_norm):
        paths.append(tokens_path)
    usd_path = os.path.join(plot_dir, COST_USD_PLOT)
    if plot_cost_usd_comparison(usd_path, apriori_norm, actual_norm):
        paths.append(usd_path)
    if per_config_df is not None and len(per_config_df):
        by_cfg_path = os.path.join(plot_dir, COST_BY_CONFIG_PLOT)
        if plot_cost_by_config_comparison(by_cfg_path, per_config_df):
            paths.append(by_cfg_path)
    return paths
