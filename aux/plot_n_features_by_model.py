#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        required=True,
        help="Path to nested_outer_hyperparameters_all.csv",
    )
    parser.add_argument(
        "--out",
        default="n_features_by_model.png",
        help="Output plot filename",
    )
    parser.add_argument(
        "--title",
        default="# Input Features by Model",
        help="Plot title",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    # Keep only standard model rows with n_features_input.
    # Teacher-student rows may instead use n_mri_features.
    df = df[df["n_features_input"].notna()].copy()
    df["n_features_input"] = pd.to_numeric(df["n_features_input"], errors="coerce")
    df = df[df["n_features_input"].notna()].copy()

    # Create a slightly more informative x-axis label.
    # You can simplify this to just model_key if desired.
    df["model_label"] = (
        df["dataset_key"].astype(str)
        + " | "
        + df["representation"].astype(str)
        + " | "
        + df["model_key"].astype(str)
    )

    # Sort models by median feature count.
    order = (
        df.groupby("model_label")["n_features_input"]
        .median()
        .sort_values()
        .index
        .tolist()
    )

    x_positions = {label: i for i, label in enumerate(order)}
    df["x"] = df["model_label"].map(x_positions)

    # Small jitter so repeated split/model points are visible.
    rng = np.random.default_rng(17)
    df["x_jitter"] = df["x"] + rng.normal(0, 0.06, size=len(df))

    plt.figure(figsize=(max(10, 0.35 * len(order)), 6))

    # Plot every row as one point.
    plt.scatter(df["x_jitter"], df["n_features_input"], alpha=0.55)

    # Add median line/marker for each model label.
    medians = df.groupby("model_label")["n_features_input"].median().reindex(order)
    plt.scatter(
        range(len(order)),
        medians.values,
        marker="D",
        s=45,
        label="Median",
    )

    plt.xticks(range(len(order)), order, rotation=90)
    plt.ylabel("# features input to ML model")
    plt.xlabel("Dataset | Representation | Model")
    plt.title(args.title)
    plt.legend()
    plt.tight_layout()

    out_path = Path(args.out)
    plt.savefig(out_path, dpi=300)
    print(f"Saved plot to: {out_path.resolve()}")

    # Also save a summary table.
    summary = (
        df.groupby("model_label")["n_features_input"]
        .agg(["count", "mean", "median", "min", "max"])
        .reset_index()
    )
    summary_path = out_path.with_suffix(".summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"Saved summary to: {summary_path.resolve()}")


if __name__ == "__main__":
    main()