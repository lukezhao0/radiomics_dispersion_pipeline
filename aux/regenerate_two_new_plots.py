#!/usr/bin/env python3
"""
Regenerate AUROC and Spearman comparison plots from nested_outer_metrics_summary.csv.

Outputs:
  top-bottom mode:
    nested_classification_auroc_comparison_NEW_top_bottom.png
    nested_regression_spearman_comparison_NEW_top_bottom.png

  all mode:
    nested_classification_auroc_comparison_NEW_all.png
    nested_regression_spearman_comparison_NEW_all.png

Formatting:
  - Vertical bar chart with labels on x-axis
  - Slanted x-axis labels
  - Metric value above each bar
  - No confidence intervals
  - Compact labels: "<model type> | <model_key>"
  - Optional top/bottom mode with visible gap between top N and bottom N
  - Optional all mode plotting all configurations
"""

from __future__ import annotations

import argparse
import os
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def compact_model_type(dataset_key: Any) -> str:
    dk = str(dataset_key)

    if dk.startswith("mri_pathcal_weighted_teacher_student"):
        return "wMRI+TS"

    if dk.startswith("mri_pathcal_weighted"):
        return "wMRI"

    if dk == "mri_teacher_student":
        return "mri"

    if dk == "mri":
        return "mri"

    if dk == "path":
        return "path"

    if dk == "combined":
        return "combined"

    return dk


def make_compact_label(row: pd.Series) -> str:
    model_type = compact_model_type(row.get("dataset_key", ""))
    model_key = str(row.get("model_key", ""))
    return f"{model_type} | {model_key}"


def select_rows_for_plot(
    metrics_df: pd.DataFrame,
    task_type: str,
    metric: str,
    plot_mode: str,
    top_n: int,
    higher_is_better: bool = True,
) -> pd.DataFrame:
    """
    Select either:
      - all rows, sorted best-to-worst
      - top N and bottom N rows, with rank_bucket labels
    """
    if "task_type" not in metrics_df.columns:
        raise ValueError("metrics_df must contain a 'task_type' column.")

    if metric not in metrics_df.columns:
        raise ValueError(f"metrics_df does not contain requested metric column: {metric}")

    df = metrics_df[metrics_df["task_type"] == task_type].copy()
    df[metric] = pd.to_numeric(df[metric], errors="coerce")
    df = df[df[metric].notna()].copy()

    if len(df) == 0:
        return df

    df["_original_row_index"] = df.index

    ranked = df.sort_values(metric, ascending=not higher_is_better).reset_index(drop=True)

    if plot_mode == "all":
        selected = ranked.copy()
        selected["rank_bucket"] = "All models"

    elif plot_mode == "top-bottom":
        top_df = ranked.head(top_n).copy()
        bottom_df = ranked.tail(top_n).copy()

        top_df["rank_bucket"] = f"Top {top_n}"
        bottom_df["rank_bucket"] = f"Bottom {top_n}"

        selected = pd.concat([top_df, bottom_df], ignore_index=True)

        # Avoid duplicate rows if total rows < top_n + bottom_n.
        selected = selected.drop_duplicates(subset=["_original_row_index"], keep="first").copy()

        # Keep top group first, bottom group second.
        top_part = selected[selected["rank_bucket"] == f"Top {top_n}"].copy()
        bottom_part = selected[selected["rank_bucket"] == f"Bottom {top_n}"].copy()

        top_part = top_part.sort_values(metric, ascending=not higher_is_better)
        bottom_part = bottom_part.sort_values(metric, ascending=not higher_is_better)

        selected = pd.concat([top_part, bottom_part], ignore_index=True)

    else:
        raise ValueError(f"Unsupported plot_mode: {plot_mode}")

    selected["plot_label"] = selected.apply(make_compact_label, axis=1)

    return selected


def save_figure(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

    try:
        plt.tight_layout()
    except Exception as exc:
        print(f"[PLOT_WARN] tight_layout failed for {path}: {exc}")

    plt.savefig(path, dpi=260, bbox_inches="tight")
    plt.close()


def plot_vertical_bar_chart(
    metrics_df: pd.DataFrame,
    task_type: str,
    metric: str,
    out_png: str,
    title: str,
    ylabel: str,
    plot_mode: str,
    top_n: int = 40,
    higher_is_better: bool = True,
    label_fontsize: int = 10,
    gap_width: float = 5.0,
) -> None:
    plot_df = select_rows_for_plot(
        metrics_df=metrics_df,
        task_type=task_type,
        metric=metric,
        plot_mode=plot_mode,
        top_n=top_n,
        higher_is_better=higher_is_better,
    )

    if len(plot_df) == 0:
        print(f"[PLOT_SKIP] No valid rows for task_type={task_type}, metric={metric}")
        return

    plot_data_csv = out_png.replace(".png", "_plot_data.csv")
    plot_df.drop(columns=["_original_row_index"], errors="ignore").to_csv(plot_data_csv, index=False)
    print(f"[SAVE] Plot data: {plot_data_csv}")

    values = plot_df[metric].astype(float).values
    labels = plot_df["plot_label"].tolist()
    n_total = len(plot_df)

    x_positions = np.arange(n_total, dtype=float)

    top_count = 0
    if plot_mode == "top-bottom":
        top_count = int((plot_df["rank_bucket"] == f"Top {top_n}").sum())
        if top_count < n_total:
            x_positions[top_count:] += gap_width

    fig_width = max(20.0, 0.34 * n_total + 6.0)
    fig_height = 10.0

    plt.figure(figsize=(fig_width, fig_height))
    bars = plt.bar(x_positions, values)

    ax = plt.gca()

    finite_values = values[np.isfinite(values)]
    if len(finite_values):
        y_min = float(np.nanmin(finite_values))
        y_max = float(np.nanmax(finite_values))
        span = y_max - y_min if y_max > y_min else 1.0
        value_offset = 0.025 * span

        for bar, val in zip(bars, values):
            if not np.isfinite(val):
                continue

            x = bar.get_x() + bar.get_width() / 2.0

            if val >= 0:
                y = val + value_offset
                va = "bottom"
            else:
                y = val - value_offset
                va = "top"

            ax.text(
                x,
                y,
                f"{val:.3f}",
                ha="center",
                va=va,
                fontsize=12,
                rotation=90,
            )

        if y_min < 0 < y_max:
            ax.axhline(0, linestyle="--", linewidth=1.0)

        lower_pad = 0.14 * span
        upper_pad = 0.24 * span
        plt.ylim(y_min - lower_pad, y_max + upper_pad)

    if plot_mode == "top-bottom" and top_count < n_total:
        left_last = x_positions[top_count - 1]
        right_first = x_positions[top_count]
        gap_mid = (left_last + right_first) / 2.0

        ax.axvline(gap_mid, linestyle="--", linewidth=1.0)

        y_top = ax.get_ylim()[1]
        ax.text(
            gap_mid,
            y_top,
            f"Top {top_n} | Bottom {top_n}",
            ha="center",
            va="bottom",
            fontsize=12,
        )

    plt.xticks(
        x_positions,
        labels,
        rotation=55,
        ha="right",
        fontsize=label_fontsize,
    )

    plt.ylabel(ylabel, fontsize=14)
    plt.title(title, fontsize=16)
    ax.tick_params(axis="y", labelsize=12)

    save_figure(out_png)
    print(f"[SAVE] Plot: {out_png}")


def regenerate_two_new_plots(
    metrics_csv: str,
    out_dir: str,
    plot_mode: str = "top-bottom",
    top_n: int = 40,
    label_fontsize: int = 10,
) -> None:
    metrics_df = pd.read_csv(metrics_csv)

    mode_suffix = "all" if plot_mode == "all" else "top_bottom"

    auroc_png = os.path.join(
        out_dir,
        f"nested_classification_auroc_comparison_NEW_{mode_suffix}.png",
    )
    spearman_png = os.path.join(
        out_dir,
        f"nested_regression_spearman_comparison_NEW_{mode_suffix}.png",
    )

    if plot_mode == "all":
        auroc_title = "Aggregate held-out classification AUROC: all configurations"
        spearman_title = "Aggregate held-out regression Spearman rho: all configurations"
    else:
        auroc_title = f"Aggregate held-out classification AUROC: top {top_n} and bottom {top_n}"
        spearman_title = f"Aggregate held-out regression Spearman rho: top {top_n} and bottom {top_n}"

    plot_vertical_bar_chart(
        metrics_df=metrics_df,
        task_type="classification",
        metric="auroc",
        out_png=auroc_png,
        title=auroc_title,
        ylabel="AUROC",
        plot_mode=plot_mode,
        top_n=top_n,
        higher_is_better=True,
        label_fontsize=label_fontsize,
    )

    plot_vertical_bar_chart(
        metrics_df=metrics_df,
        task_type="regression",
        metric="spearman_rho",
        out_png=spearman_png,
        title=spearman_title,
        ylabel="Spearman rho",
        plot_mode=plot_mode,
        top_n=top_n,
        higher_is_better=True,
        label_fontsize=label_fontsize,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate AUROC and Spearman comparison plots from an existing "
            "nested_outer_metrics_summary.csv."
        )
    )

    parser.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help=(
            "Existing output directory containing nested_outer_metrics_summary.csv. "
            "The new plots will also be saved here."
        ),
    )

    parser.add_argument(
        "--metrics-csv",
        type=str,
        default=None,
        help=(
            "Optional explicit path to nested_outer_metrics_summary.csv. "
            "If omitted, uses <out-dir>/nested_outer_metrics_summary.csv."
        ),
    )

    parser.add_argument(
        "--plot-mode",
        type=str,
        default="top-bottom",
        choices=["top-bottom", "all"],
        help=(
            "Plot mode. Use 'top-bottom' for top N and bottom N with a gap, "
            "or 'all' to plot all configurations. Default: top-bottom."
        ),
    )

    parser.add_argument(
        "--top-n",
        type=int,
        default=40,
        help="Number of top and bottom configurations to plot in top-bottom mode. Default: 40.",
    )

    parser.add_argument(
        "--label-fontsize",
        type=int,
        default=14,
        help="Font size for x-axis model labels. Default: 10.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    out_dir = os.path.abspath(args.out_dir)
    metrics_csv = (
        os.path.abspath(args.metrics_csv)
        if args.metrics_csv
        else os.path.join(out_dir, "nested_outer_metrics_summary.csv")
    )

    if not os.path.exists(metrics_csv):
        raise FileNotFoundError(f"Could not find metrics CSV: {metrics_csv}")

    os.makedirs(out_dir, exist_ok=True)

    print(f"[INPUT] metrics_csv={metrics_csv}")
    print(f"[OUTPUT] out_dir={out_dir}")
    print(f"[CONFIG] plot_mode={args.plot_mode}")
    print(f"[CONFIG] top_n={args.top_n}")
    print(f"[CONFIG] label_fontsize={args.label_fontsize}")

    regenerate_two_new_plots(
        metrics_csv=metrics_csv,
        out_dir=out_dir,
        plot_mode=args.plot_mode,
        top_n=args.top_n,
        label_fontsize=args.label_fontsize,
    )

    print("[DONE] Regenerated AUROC and Spearman comparison plots.")


if __name__ == "__main__":
    main()