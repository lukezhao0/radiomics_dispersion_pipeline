#!/usr/bin/env python3

import os
import textwrap
import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = "/Users/lukezhao/projects/onc/out/llm_token_audit_260504/llm_token_cost_estimates.csv"

# Choose one:
#   "no_cache"
#   "common_prefix_cache_estimate"
#   "custom_cached_fraction_0.00"
CACHE_SCENARIO = "no_cache"

OUT_DIR = os.path.dirname(CSV_PATH)
OUT_PNG_TOTAL = os.path.join(OUT_DIR, f"llm_token_cost_bar_chart_{CACHE_SCENARIO}.png")
OUT_PNG_STACKED = os.path.join(OUT_DIR, f"llm_token_cost_stacked_bar_chart_{CACHE_SCENARIO}.png")


def wrap_label(label: str, width: int = 28) -> str:
    return "\n".join(textwrap.wrap(str(label), width=width))


def main() -> None:
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Could not find CSV file: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)

    required_cols = {
        "model",
        "processing",
        "context",
        "cache_scenario",
        "estimated_input_cost_usd",
        "estimated_cached_input_cost_usd",
        "estimated_output_cost_usd",
        "estimated_total_cost_usd",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    plot_df = df[df["cache_scenario"] == CACHE_SCENARIO].copy()

    if plot_df.empty:
        available = sorted(df["cache_scenario"].dropna().unique())
        raise ValueError(
            f"No rows found for CACHE_SCENARIO={CACHE_SCENARIO!r}. "
            f"Available scenarios: {available}"
        )

    plot_df["label"] = (
        plot_df["model"].astype(str)
        + " | "
        + plot_df["processing"].astype(str)
        + " | "
        + plot_df["context"].astype(str)
    )

    plot_df = plot_df.sort_values("estimated_total_cost_usd", ascending=True).reset_index(drop=True)
    plot_df["wrapped_label"] = plot_df["label"].apply(wrap_label)

    # -------------------------
    # Plot 1: Total estimated cost
    # -------------------------
    plt.figure(figsize=(14, max(6, 0.55 * len(plot_df))))
    plt.barh(plot_df["wrapped_label"], plot_df["estimated_total_cost_usd"])
    plt.xlabel("Estimated total cost, USD")
    plt.ylabel("Model / processing / context")
    plt.title(f"Estimated LLM Token Cost by Model ({CACHE_SCENARIO})")

    for i, value in enumerate(plot_df["estimated_total_cost_usd"]):
        plt.text(value, i, f" ${value:,.2f}", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(OUT_PNG_TOTAL, dpi=300)
    plt.close()

    # -------------------------
    # Plot 2: Stacked cost components
    # -------------------------
    input_cost = plot_df["estimated_input_cost_usd"].astype(float)
    cached_input_cost = plot_df["estimated_cached_input_cost_usd"].astype(float)
    output_cost = plot_df["estimated_output_cost_usd"].astype(float)

    plt.figure(figsize=(14, max(6, 0.55 * len(plot_df))))
    plt.barh(plot_df["wrapped_label"], input_cost, label="Input cost")
    plt.barh(
        plot_df["wrapped_label"],
        cached_input_cost,
        left=input_cost,
        label="Cached input cost",
    )
    plt.barh(
        plot_df["wrapped_label"],
        output_cost,
        left=input_cost + cached_input_cost,
        label="Output cost",
    )

    plt.xlabel("Estimated cost, USD")
    plt.ylabel("Model / processing / context")
    plt.title(f"Estimated LLM Token Cost Components ({CACHE_SCENARIO})")
    plt.legend()

    total = plot_df["estimated_total_cost_usd"].astype(float)
    for i, value in enumerate(total):
        plt.text(value, i, f" ${value:,.2f}", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(OUT_PNG_STACKED, dpi=300)
    plt.close()

    print(f"[DONE] Wrote total cost bar chart: {OUT_PNG_TOTAL}")
    print(f"[DONE] Wrote stacked cost bar chart: {OUT_PNG_STACKED}")

    print("\nTop 10 most expensive rows:")
    cols = [
        "model",
        "processing",
        "context",
        "cache_scenario",
        "estimated_input_cost_usd",
        "estimated_cached_input_cost_usd",
        "estimated_output_cost_usd",
        "estimated_total_cost_usd",
    ]
    print(
        plot_df.sort_values("estimated_total_cost_usd", ascending=False)[cols]
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()