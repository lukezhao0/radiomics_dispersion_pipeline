"""Generate comparison plots for experiment runs."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PLOT_CAPTIONS: Dict[str, str] = {
    "best_regression_spearman": (
        "Best regression Spearman ρ per run (max across shotsets/modalities/models). "
        "Addresses: which run achieves strongest dispersion score ranking?"
    ),
    "best_high_low_classification": (
        "Best high/low dispersion classification performance per run. "
        "Uses AUROC when available; otherwise accuracy (clearly labeled). "
        "Addresses: which run best separates high vs low dispersion?"
    ),
    "best_relapse_classification": (
        "Best relapse classification performance per run (AUROC preferred, else F1/accuracy). "
        "Addresses: which run best predicts relapse?"
    ),
    "approach1_shotset": (
        "Approach 1 shotset comparison within each run. "
        "Addresses: which few-shot exemplar set performs best?"
    ),
    "modality_comparison": (
        "Modality comparison (MRI-only, pathology-only, combined). "
        "Addresses: which input modality combination is most informative?"
    ),
    "reasoning_comparison": (
        "Reasoning-effort comparison across runs, faceted by approach. "
        "Addresses: do minimal, low, and medium/default reasoning differ meaningfully?"
    ),
    "approach2_model_family": (
        "Approach 2 model-family comparison for regression Spearman and high/low AUROC. "
        "Addresses: which supervised model family performs best?"
    ),
    "approach2_representation": (
        "Approach 2 feature-representation comparison. "
        "Addresses: which lexical representation is most predictive?"
    ),
    "cost_comparison": (
        "Estimated a priori vs actual LLM cost per run. "
        "Addresses: how do cost estimates compare to billed usage?"
    ),
    "performance_vs_cost": (
        "Performance vs actual LLM cost scatter. "
        "Addresses: what is the performance–cost trade-off?"
    ),
    "performance_vs_runtime": (
        "Performance vs runtime or API calls. "
        "Addresses: what is the performance–compute trade-off?"
    ),
    "api_token_usage": (
        "API calls and token usage comparison across runs. "
        "Addresses: how do runs differ in LLM resource consumption?"
    ),
    "summary_heatmap": (
        "Normalized heatmap of key metrics across runs. "
        "Addresses: which runs are strongest across multiple dimensions?"
    ),
}


def _save_fig(fig: plt.Figure, out_dir: Path, name: str) -> Path:
    png_path = out_dir / f"{name}.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return png_path


def _run_meta_cols(df: pd.DataFrame) -> List[str]:
    cols = ["run_id", "run_label", "approach", "model", "reasoning", "pipeline_version"]
    return [c for c in cols if c in df.columns]


def _filter_target_metric(df: pd.DataFrame, target: str, metric: str) -> pd.DataFrame:
    return df[(df["target"] == target) & (df["metric"] == metric)].copy()


def _operational_best_per_run(df: pd.DataFrame, metric: str, higher_is_better: bool = False) -> pd.DataFrame:
    """Prefer run-level aggregate cost/usage rows over per-config duplicates."""
    sub = _filter_target_metric(df, "operational", metric)
    if sub.empty:
        return sub
    priority = []
    for _, row in sub.iterrows():
        src = str(row.get("source_file", ""))
        if "llm_token_cost_report.json" in src or "aggregated_from_per_config" in src:
            p = 0
        elif "llm_cost_estimate_apriori" in src:
            p = 1
        elif "token_cost_report.json" in src:
            p = 3
        else:
            p = 2
        priority.append(p)
    sub = sub.copy()
    sub["_priority"] = priority
    group_cols = _run_meta_cols(sub)
    rows = []
    for _, grp in sub.groupby(group_cols, dropna=False):
        if higher_is_better:
            row = grp.loc[grp["value"].idxmax()]
        else:
            row = grp.sort_values(["_priority", "value"]).iloc[0]
        rows.append(row)
    out = pd.DataFrame(rows)
    return out.drop(columns=["_priority"], errors="ignore")


def _best_per_run(
    df: pd.DataFrame, target: str, metric: str, higher_is_better: bool = True
) -> pd.DataFrame:
    if target == "operational":
        return _operational_best_per_run(df, metric, higher_is_better=higher_is_better)
    sub = _filter_target_metric(df, target, metric)
    if sub.empty:
        return sub
    group_cols = _run_meta_cols(sub)
    if higher_is_better:
        idx = sub.groupby(group_cols, dropna=False)["value"].idxmax()
    else:
        idx = sub.groupby(group_cols, dropna=False)["value"].idxmin()
    return sub.loc[idx].copy()


def _plot_bar_best_metric(
    df: pd.DataFrame,
    target: str,
    metric: str,
    title: str,
    ylabel: str,
    out_dir: Path,
    filename: str,
    fallback: Optional[Tuple[str, str]] = None,
) -> Optional[Path]:
    best = _best_per_run(df, target, metric)
    metric_used = metric
    if best.empty and fallback:
        metric_used = fallback[1]
        best = _best_per_run(df, fallback[0], fallback[1])
        title = f"{title} (using {fallback[1]} — {metric} unavailable)"
        ylabel = fallback[1]

    if best.empty:
        logger.warning("No data for plot %s (%s/%s)", filename, target, metric)
        return None

    best = best.sort_values("run_label")
    fig, ax = plt.subplots(figsize=(max(8, len(best) * 0.6), 5))
    colors = best["approach"].map({"approach1": "#4C78A8", "approach2": "#F58518"}).fillna("#72B7B2")
    ax.bar(best["run_label"], best["value"], color=colors)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Run")
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    for i, row in best.iterrows():
        ax.text(
            list(best.index).index(i), row["value"], f"{row['value']:.3f}",
            ha="center", va="bottom", fontsize=7,
        )
    fig.tight_layout()
    return _save_fig(fig, out_dir, filename)


def plot_best_regression_spearman(df: pd.DataFrame, out_dir: Path) -> Optional[Path]:
    return _plot_bar_best_metric(
        df, "dispersion_regression", "spearman_rho",
        "Best Regression Spearman ρ by Run", "Spearman ρ", out_dir, "best_regression_spearman",
    )


def plot_best_high_low_classification(df: pd.DataFrame, out_dir: Path) -> Optional[Path]:
    return _plot_bar_best_metric(
        df, "dispersion_high_low", "auroc",
        "Best High/Low Dispersion Classification by Run", "AUROC", out_dir,
        "best_high_low_classification",
        fallback=("dispersion_high_low", "accuracy"),
    )


def plot_best_relapse_classification(df: pd.DataFrame, out_dir: Path) -> Optional[Path]:
    for metric in ("auroc", "auprc", "f1", "accuracy"):
        path = _plot_bar_best_metric(
            df, "relapse", metric,
            f"Best Relapse Classification ({metric.upper()}) by Run", metric.upper(),
            out_dir, "best_relapse_classification",
        )
        if path:
            return path
    return None


def plot_approach1_shotset(df: pd.DataFrame, out_dir: Path) -> Optional[Path]:
    sub = df[
        (df["approach"] == "approach1")
        & (df["shotset"].astype(str).str.len() > 0)
        & (
            ((df["target"] == "dispersion_regression") & (df["metric"] == "spearman_rho"))
            | ((df["target"] == "dispersion_high_low") & (df["metric"] == "accuracy"))
        )
    ].copy()
    if sub.empty:
        logger.warning("No Approach 1 shotset data")
        return None

    runs = sub["run_label"].unique()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (target, metric, title) in zip(
        axes,
        [
            ("dispersion_regression", "spearman_rho", "Spearman ρ"),
            ("dispersion_high_low", "accuracy", "Accuracy"),
        ],
    ):
        part = sub[(sub["target"] == target) & (sub["metric"] == metric)]
        if part.empty:
            ax.set_title(f"{title} — no data")
            continue
        for run_label in runs:
            rp = part[part["run_label"] == run_label]
            ax.plot(rp["shotset"], rp["value"], marker="o", label=run_label)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=30, labelsize=7)
        ax.legend(fontsize=6)
    fig.suptitle("Approach 1 Shotset Comparison")
    fig.tight_layout()
    return _save_fig(fig, out_dir, "approach1_shotset")


def plot_modality_comparison(df: pd.DataFrame, out_dir: Path) -> Optional[Path]:
    sub = df[
        (df["modality"].astype(str).str.len() > 0)
        & (df["target"] == "dispersion_regression")
        & (df["metric"] == "spearman_rho")
    ].copy()
    if sub.empty:
        logger.warning("No modality comparison data")
        return None

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, approach in zip(axes, ["approach1", "approach2"]):
        part = sub[sub["approach"] == approach]
        if part.empty:
            ax.set_title(f"{approach} — no data")
            continue
        pivot = part.groupby(["run_label", "modality"])["value"].max().reset_index()
        modalities = sorted(pivot["modality"].unique())
        x = np.arange(len(pivot["run_label"].unique()))
        width = 0.8 / max(len(modalities), 1)
        run_labels = sorted(pivot["run_label"].unique())
        for i, mod in enumerate(modalities):
            vals = [
                pivot[(pivot["run_label"] == rl) & (pivot["modality"] == mod)]["value"].max()
                if len(pivot[(pivot["run_label"] == rl) & (pivot["modality"] == mod)]) else np.nan
                for rl in run_labels
            ]
            ax.bar(x + i * width, vals, width=width, label=mod)
        ax.set_xticks(x + width * (len(modalities) - 1) / 2)
        ax.set_xticklabels(run_labels, rotation=30, ha="right", fontsize=7)
        ax.set_title(f"{approach}: Spearman ρ by Modality")
        ax.legend(fontsize=6)
    fig.suptitle("Modality Comparison")
    fig.tight_layout()
    return _save_fig(fig, out_dir, "modality_comparison")


def plot_reasoning_comparison(df: pd.DataFrame, out_dir: Path) -> Optional[Path]:
    best = _best_per_run(df, "dispersion_regression", "spearman_rho")
    if best.empty:
        return None
    fig, ax = plt.subplots(figsize=(10, 5))
    for approach, grp in best.groupby("approach"):
        ax.scatter(grp["reasoning"], grp["value"], label=approach, s=80)
        for _, row in grp.iterrows():
            ax.annotate(row["run_label"], (row["reasoning"], row["value"]), fontsize=6, alpha=0.7)
    ax.set_title("Reasoning Level vs Best Spearman ρ")
    ax.set_xlabel("Reasoning")
    ax.set_ylabel("Best Spearman ρ")
    ax.legend()
    fig.tight_layout()
    return _save_fig(fig, out_dir, "reasoning_comparison")


def plot_approach2_model_family(df: pd.DataFrame, out_dir: Path) -> Optional[Path]:
    sub = df[
        (df["approach"] == "approach2")
        & (df["model_key"].astype(str).str.len() > 0)
        & (
            ((df["target"] == "dispersion_regression") & (df["metric"] == "spearman_rho"))
            | ((df["target"] == "dispersion_high_low") & (df["metric"] == "auroc"))
        )
    ].copy()
    if sub.empty:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (target, metric, title) in zip(
        axes,
        [
            ("dispersion_regression", "spearman_rho", "Regression Spearman"),
            ("dispersion_high_low", "auroc", "High/Low AUROC"),
        ],
    ):
        part = sub[(sub["target"] == target) & (sub["metric"] == metric)]
        if part.empty:
            ax.set_title(f"{title} — no data")
            continue
        agg = part.groupby("model_key")["value"].max().sort_values(ascending=False)
        ax.barh(agg.index.astype(str), agg.values)
        ax.set_title(title)
        ax.invert_yaxis()
    fig.suptitle("Approach 2 Model Family Comparison (best across runs)")
    fig.tight_layout()
    return _save_fig(fig, out_dir, "approach2_model_family")


def plot_approach2_representation(df: pd.DataFrame, out_dir: Path) -> Optional[Path]:
    sub = df[
        (df["approach"] == "approach2")
        & (df["representation"].astype(str).str.len() > 0)
        & (df["target"] == "dispersion_regression")
        & (df["metric"] == "spearman_rho")
    ].copy()
    if sub.empty:
        return None
    agg = sub.groupby("representation")["value"].max().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(agg.index.astype(str), agg.values, color="#54A24B")
    ax.set_title("Approach 2 Feature Representation (max Spearman ρ)")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    return _save_fig(fig, out_dir, "approach2_representation")


def plot_cost_comparison(df: pd.DataFrame, out_dir: Path) -> Optional[Path]:
    actual = _best_per_run(df, "operational", "cost_usd_actual", higher_is_better=False)
    apriori = _best_per_run(df, "operational", "cost_usd_apriori", higher_is_better=False)
    if actual.empty and apriori.empty:
        return None

    runs = sorted(set(actual["run_label"].tolist()) | set(apriori["run_label"].tolist()))
    x = np.arange(len(runs))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(8, len(runs) * 0.6), 5))

    def _vals(frame: pd.DataFrame) -> List[float]:
        return [
            frame[frame["run_label"] == rl]["value"].iloc[0]
            if len(frame[frame["run_label"] == rl]) else np.nan
            for rl in runs
        ]

    if not apriori.empty:
        ax.bar(x - width / 2, _vals(apriori), width, label="A priori estimate")
    if not actual.empty:
        ax.bar(x + width / 2, _vals(actual), width, label="Actual cost")
    ax.set_xticks(x)
    ax.set_xticklabels(runs, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("USD")
    ax.set_title("LLM Cost: A Priori vs Actual")
    ax.legend()
    fig.tight_layout()
    return _save_fig(fig, out_dir, "cost_comparison")


def plot_performance_vs_cost(df: pd.DataFrame, out_dir: Path) -> Optional[Path]:
    perf = _best_per_run(df, "dispersion_regression", "spearman_rho")
    cost = _best_per_run(df, "operational", "cost_usd_actual", higher_is_better=False)
    if perf.empty or cost.empty:
        return None
    merged = perf.merge(
        cost[_run_meta_cols(cost) + ["value"]],
        on=_run_meta_cols(perf),
        suffixes=("_perf", "_cost"),
    )
    if merged.empty:
        return None
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(merged["value_cost"], merged["value_perf"], s=80)
    for _, row in merged.iterrows():
        ax.annotate(row["run_label"], (row["value_cost"], row["value_perf"]), fontsize=7)
    ax.set_xlabel("Actual LLM Cost (USD)")
    ax.set_ylabel("Best Spearman ρ")
    ax.set_title("Performance vs Cost")
    fig.tight_layout()
    return _save_fig(fig, out_dir, "performance_vs_cost")


def plot_performance_vs_runtime(df: pd.DataFrame, out_dir: Path) -> Optional[Path]:
    perf = _best_per_run(df, "dispersion_regression", "spearman_rho")
    runtime = _filter_target_metric(df, "operational", "runtime_seconds")
    calls = _filter_target_metric(df, "operational", "api_calls")

    x_df = runtime if not runtime.empty else calls
    x_label = "Runtime (seconds)" if not runtime.empty else "API Calls"
    if perf.empty or x_df.empty:
        # Try API calls at run level
        if perf.empty or calls.empty:
            logger.warning("No performance vs runtime data")
            return None
        x_df = _best_per_run(df, "operational", "api_calls", higher_is_better=False)
        x_label = "API Calls"

    x_best = x_df.groupby(_run_meta_cols(x_df), dropna=False)["value"].max().reset_index()
    merged = perf.merge(x_best, on=_run_meta_cols(perf), suffixes=("_perf", "_x"))
    if merged.empty:
        return None
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(merged["value_x"], merged["value_perf"], s=80)
    for _, row in merged.iterrows():
        ax.annotate(row["run_label"], (row["value_x"], row["value_perf"]), fontsize=7)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Best Spearman ρ")
    ax.set_title("Performance vs Runtime / API Calls")
    fig.tight_layout()
    return _save_fig(fig, out_dir, "performance_vs_runtime")


def plot_api_token_usage(df: pd.DataFrame, out_dir: Path) -> Optional[Path]:
    metrics = ["api_calls", "prompt_tokens", "cached_tokens", "completion_tokens", "reasoning_tokens"]
    sub = df[(df["target"] == "operational") & (df["metric"].isin(metrics))].copy()
    if sub.empty:
        return None

    # Use run-level max (aggregate cost reports)
    run_best = sub.groupby(_run_meta_cols(sub) + ["metric"], dropna=False)["value"].max().reset_index()
    pivot = run_best.pivot_table(
        index="run_label", columns="metric", values="value", aggfunc="max"
    )
    if pivot.empty:
        return None

    fig, ax = plt.subplots(figsize=(12, 5))
    pivot.plot(kind="bar", ax=ax)
    ax.set_title("API Calls and Token Usage by Run")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    return _save_fig(fig, out_dir, "api_token_usage")


def plot_summary_heatmap(df: pd.DataFrame, out_dir: Path) -> Optional[Path]:
    targets_metrics = [
        ("dispersion_regression", "spearman_rho"),
        ("dispersion_high_low", "auroc"),
        ("dispersion_high_low", "accuracy"),
        ("relapse", "auroc"),
        ("operational", "cost_usd_actual"),
    ]
    rows = []
    for target, metric in targets_metrics:
        best = _best_per_run(
            df, target, metric,
            higher_is_better=metric not in ("cost_usd_actual", "mae", "rmse"),
        )
        if best.empty and target == "dispersion_high_low" and metric == "auroc":
            best = _best_per_run(df, target, "accuracy")
            metric = "accuracy"
        if best.empty:
            continue
        for _, r in best.iterrows():
            rows.append(
                {
                    "run_label": r["run_label"],
                    "column": f"{target}:{metric}",
                    "value": r["value"],
                }
            )
    if not rows:
        return None

    heat = pd.DataFrame(rows).pivot(index="run_label", columns="column", values="value")
    # Normalize columns 0-1 for display (higher better; invert cost)
    norm = heat.copy()
    for col in norm.columns:
        col_vals = norm[col].astype(float)
        if "cost" in col:
            col_vals = col_vals.max() - col_vals
        vmin, vmax = col_vals.min(), col_vals.max()
        if vmax > vmin:
            norm[col] = (col_vals - vmin) / (vmax - vmin)
        else:
            norm[col] = 0.5

    fig, ax = plt.subplots(figsize=(max(8, len(heat.columns) * 1.2), max(4, len(heat) * 0.4)))
    im = ax.imshow(norm.values, aspect="auto", cmap="YlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(heat.columns)))
    ax.set_xticklabels(heat.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(heat.index)))
    ax.set_yticklabels(heat.index, fontsize=8)
    for i in range(len(heat.index)):
        for j in range(len(heat.columns)):
            raw = heat.iloc[i, j]
            if not pd.isna(raw):
                ax.text(j, i, f"{raw:.3f}", ha="center", va="center", fontsize=7)
    ax.set_title("Summary Heatmap (columns normalized; greener = better within column)")
    fig.colorbar(im, ax=ax, fraction=0.02)
    fig.tight_layout()
    return _save_fig(fig, out_dir, "summary_heatmap")


PLOT_FUNCTIONS = {
    "best_regression_spearman": plot_best_regression_spearman,
    "best_high_low_classification": plot_best_high_low_classification,
    "best_relapse_classification": plot_best_relapse_classification,
    "approach1_shotset": plot_approach1_shotset,
    "modality_comparison": plot_modality_comparison,
    "reasoning_comparison": plot_reasoning_comparison,
    "approach2_model_family": plot_approach2_model_family,
    "approach2_representation": plot_approach2_representation,
    "cost_comparison": plot_cost_comparison,
    "performance_vs_cost": plot_performance_vs_cost,
    "performance_vs_runtime": plot_performance_vs_runtime,
    "api_token_usage": plot_api_token_usage,
    "summary_heatmap": plot_summary_heatmap,
}


def generate_all_plots(
    normalized_df: pd.DataFrame,
    plots_dir: Path,
    plot_names: List[str],
) -> Dict[str, Optional[Path]]:
    results: Dict[str, Optional[Path]] = {}
    for name in plot_names:
        fn = PLOT_FUNCTIONS.get(name)
        if fn is None:
            logger.warning("Unknown plot: %s", name)
            results[name] = None
            continue
        try:
            results[name] = fn(normalized_df, plots_dir)
        except Exception as exc:
            logger.exception("Plot %s failed: %s", name, exc)
            results[name] = None
    return results
