#!/usr/bin/env python3
"""
Plot MRI audit metrics by low vs high tumor dispersion.

This script reads mri_audit_case_table_all_resumed_splits.csv and compares
MRI report/audit metrics between low-dispersion and high-dispersion cases.

Outputs:
  - One PNG plot per metric
  - audit_metric_high_low_statistics.csv
  - audit_metric_high_low_statistics_fdr.csv
  - audit_metric_high_low_manifest.csv

Default behavior:
  - Uses all rows unless --split-role is set to train/test
  - Aggregates repeated case_id,row_index rows to one case-level row by default,
    which avoids treating repeated split appearances of the same case as fully
    independent observations.
"""

from __future__ import annotations

import argparse
import math
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


DEFAULT_INPUT_CSV = (
    "/Users/lukezhao/projects/onc/out/"
    "securegpt_dispersion_nested_path_informed_eval7_resumed3/"
    "mri_audit_case_table_all_resumed_splits.csv"
)

METRICS = [
    "mri_report_chars",
    "mri_report_words",
    "has_findings_section",
    "has_impression_section",
    "has_comparison_section",
    "has_response_language",
    "has_measurement_language",
    "n_phrase_features",
    "n_phrase_present",
    "n_phrase_uncertain",
    "n_phrase_negated",
    "n_group_features",
    "n_group_present",
    "n_spatial_morph_response_phrases",
    "n_distribution_phrases",
    "n_ambiguity_phrases",
    "n_uncertain_group_mentions",
    "n_negated_group_mentions",
    "mri_extraction_density",
    "distribution_density",
    "uncertainty_density",
    "negation_density",
]

BINARY_METRICS = {
    "has_findings_section",
    "has_impression_section",
    "has_comparison_section",
    "has_response_language",
    "has_measurement_language",
}


def sanitize_filename(s: str) -> str:
    """Make a safe filename stem."""
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s))
    return s.strip("_")


def parse_dispersion_label(x) -> Optional[int]:
    """
    Normalize dispersion labels to:
      0 = low dispersion
      1 = high dispersion
    """
    if pd.isna(x):
        return None

    if isinstance(x, str):
        t = x.strip().lower()
        if t in {"1", "high", "high_dispersion", "true", "yes"}:
            return 1
        if t in {"0", "low", "low_dispersion", "false", "no"}:
            return 0

    try:
        val = float(x)
        if val == 1:
            return 1
        if val == 0:
            return 0
    except Exception:
        return None

    return None


def benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    """
    Benjamini-Hochberg FDR correction.
    Returns q-values aligned to the input index.
    """
    p = pd.to_numeric(p_values, errors="coerce")
    q = pd.Series(np.nan, index=p.index, dtype=float)

    valid = p.dropna()
    if valid.empty:
        return q

    order = valid.sort_values().index
    ranked_p = valid.loc[order].values
    m = len(ranked_p)

    raw_q = ranked_p * m / np.arange(1, m + 1)
    monotone_q = np.minimum.accumulate(raw_q[::-1])[::-1]
    monotone_q = np.clip(monotone_q, 0, 1)

    q.loc[order] = monotone_q
    return q


def cliffs_delta(high: np.ndarray, low: np.ndarray) -> float:
    """
    Cliff's delta effect size.

    Positive value means high-dispersion cases tend to have larger values
    than low-dispersion cases.
    """
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    high = high[np.isfinite(high)]
    low = low[np.isfinite(low)]

    if len(high) == 0 or len(low) == 0:
        return np.nan

    greater = 0
    lesser = 0

    # This is fine for small/medium clinical datasets.
    # If you have millions of rows, replace with a rank-based implementation.
    for h in high:
        greater += np.sum(h > low)
        lesser += np.sum(h < low)

    return float((greater - lesser) / (len(high) * len(low)))


def bootstrap_diff_ci(
    high: np.ndarray,
    low: np.ndarray,
    statistic: str = "mean",
    n_boot: int = 5000,
    seed: int = 17,
) -> Tuple[float, float]:
    """
    Bootstrap 95% CI for high-minus-low difference in mean or median.
    """
    rng = np.random.default_rng(seed)

    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    high = high[np.isfinite(high)]
    low = low[np.isfinite(low)]

    if len(high) < 2 or len(low) < 2:
        return np.nan, np.nan

    diffs = []
    for _ in range(n_boot):
        h = rng.choice(high, size=len(high), replace=True)
        l = rng.choice(low, size=len(low), replace=True)

        if statistic == "mean":
            diffs.append(np.mean(h) - np.mean(l))
        elif statistic == "median":
            diffs.append(np.median(h) - np.median(l))
        else:
            raise ValueError(f"Unsupported statistic: {statistic}")

    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """
    Wilson 95% CI for a binomial proportion.
    """
    if n <= 0:
        return np.nan, np.nan

    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half_width = z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom

    return float(max(0, center - half_width)), float(min(1, center + half_width))


def aggregate_to_case_level(df: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
    """
    Collapse repeated split-level rows to one row per case_id,row_index.

    Why:
      mri_audit_case_table_all_resumed_splits.csv can contain repeated rows for
      the same case across split artifacts. Treating all repeated rows as
      independent can inflate statistical significance.

    Strategy:
      - dispersion labels/outcomes: first non-null value
      - split_role: prefer test if a case appears as test in any split
      - metric columns: mean across appearances
    """
    required = {"case_id", "row_index"}
    if not required.issubset(df.columns):
        print("[WARN] case_id/row_index missing; cannot aggregate to case level.")
        return df.copy()

    df = df.copy()

    # Prefer test rows when preserving one representative split_role.
    role_priority = {"test": 0, "train": 1}
    if "split_role" in df.columns:
        df["_role_priority"] = (
            df["split_role"]
            .astype(str)
            .str.lower()
            .map(role_priority)
            .fillna(2)
        )
    else:
        df["_role_priority"] = 2

    df = df.sort_values(["case_id", "row_index", "_role_priority"])

    agg = {}

    for col in df.columns:
        if col in {"case_id", "row_index"}:
            continue
        if col in metrics:
            agg[col] = "mean"
        elif col == "dispersion_true_high_low_norm":
            agg[col] = "first"
        elif col in {
            "dispersion_true",
            "dispersion_true_high_low",
            "relapse_true",
            "split_role",
            "split_id",
            "source_split_id",
        }:
            agg[col] = "first"

    out = (
        df.groupby(["case_id", "row_index"], as_index=False)
        .agg(agg)
        .drop(columns=["_role_priority"], errors="ignore")
    )

    return out


def compute_continuous_stats(
    metric: str,
    high: np.ndarray,
    low: np.ndarray,
    n_boot: int,
    seed: int,
) -> Dict[str, float]:
    """
    Statistics for continuous/count/density metrics.
    """
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)

    high = high[np.isfinite(high)]
    low = low[np.isfinite(low)]

    rec = {
        "metric": metric,
        "metric_type": "continuous_or_count",
        "n_low": len(low),
        "n_high": len(high),
        "low_mean": np.nan,
        "high_mean": np.nan,
        "high_minus_low_mean": np.nan,
        "mean_diff_ci95_low": np.nan,
        "mean_diff_ci95_high": np.nan,
        "low_median": np.nan,
        "high_median": np.nan,
        "high_minus_low_median": np.nan,
        "median_diff_ci95_low": np.nan,
        "median_diff_ci95_high": np.nan,
        "low_std": np.nan,
        "high_std": np.nan,
        "welch_t_p": np.nan,
        "mannwhitney_p": np.nan,
        "cliffs_delta_high_vs_low": np.nan,
        "main_p": np.nan,
        "main_test": "Mann-Whitney U",
    }

    if len(low) == 0 or len(high) == 0:
        return rec

    rec["low_mean"] = float(np.mean(low))
    rec["high_mean"] = float(np.mean(high))
    rec["high_minus_low_mean"] = rec["high_mean"] - rec["low_mean"]

    rec["low_median"] = float(np.median(low))
    rec["high_median"] = float(np.median(high))
    rec["high_minus_low_median"] = rec["high_median"] - rec["low_median"]

    rec["low_std"] = float(np.std(low, ddof=1)) if len(low) > 1 else np.nan
    rec["high_std"] = float(np.std(high, ddof=1)) if len(high) > 1 else np.nan

    if len(low) >= 2 and len(high) >= 2:
        rec["welch_t_p"] = float(
            stats.ttest_ind(high, low, equal_var=False, nan_policy="omit").pvalue
        )

        rec["mannwhitney_p"] = float(
            stats.mannwhitneyu(high, low, alternative="two-sided").pvalue
        )

        rec["main_p"] = rec["mannwhitney_p"]
        rec["cliffs_delta_high_vs_low"] = cliffs_delta(high, low)

        lo, hi = bootstrap_diff_ci(
            high,
            low,
            statistic="mean",
            n_boot=n_boot,
            seed=seed,
        )
        rec["mean_diff_ci95_low"] = lo
        rec["mean_diff_ci95_high"] = hi

        lo, hi = bootstrap_diff_ci(
            high,
            low,
            statistic="median",
            n_boot=n_boot,
            seed=seed + 1,
        )
        rec["median_diff_ci95_low"] = lo
        rec["median_diff_ci95_high"] = hi

    return rec


def compute_binary_stats(metric: str, high: np.ndarray, low: np.ndarray) -> Dict[str, float]:
    """
    Statistics for binary flag metrics.
    """
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)

    high = high[np.isfinite(high)]
    low = low[np.isfinite(low)]

    # Force binary interpretation after numeric coercion.
    high = (high > 0).astype(int)
    low = (low > 0).astype(int)

    n_high = len(high)
    n_low = len(low)
    high_pos = int(high.sum())
    low_pos = int(low.sum())
    high_neg = int(n_high - high_pos)
    low_neg = int(n_low - low_pos)

    rec = {
        "metric": metric,
        "metric_type": "binary_flag",
        "n_low": n_low,
        "n_high": n_high,
        "low_positive": low_pos,
        "high_positive": high_pos,
        "low_proportion": np.nan,
        "high_proportion": np.nan,
        "high_minus_low_proportion": np.nan,
        "fisher_exact_p": np.nan,
        "odds_ratio_high_vs_low": np.nan,
        "main_p": np.nan,
        "main_test": "Fisher exact",
    }

    if n_low == 0 or n_high == 0:
        return rec

    low_prop = low_pos / n_low
    high_prop = high_pos / n_high

    rec["low_proportion"] = float(low_prop)
    rec["high_proportion"] = float(high_prop)
    rec["high_minus_low_proportion"] = float(high_prop - low_prop)

    # Table rows are high/low, columns are positive/negative.
    table = np.array(
        [
            [high_pos, high_neg],
            [low_pos, low_neg],
        ]
    )

    try:
        odds_ratio, p = stats.fisher_exact(table, alternative="two-sided")
        rec["odds_ratio_high_vs_low"] = float(odds_ratio)
        rec["fisher_exact_p"] = float(p)
        rec["main_p"] = float(p)
    except Exception:
        pass

    return rec


def plot_continuous_metric(
    metric: str,
    high: np.ndarray,
    low: np.ndarray,
    stat_row: Dict,
    output_path: Path,
    seed: int,
) -> None:
    """
    Boxplot + jittered points for continuous/count/density metrics.
    """
    rng = np.random.default_rng(seed)

    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    high = high[np.isfinite(high)]
    low = low[np.isfinite(low)]

    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    data = [low, high]
    labels = [f"Low dispersion\nn={len(low)}", f"High dispersion\nn={len(high)}"]

    ax.boxplot(
        data,
        labels=labels,
        showfliers=False,
        widths=0.55,
    )

    for i, vals in enumerate(data, start=1):
        if len(vals) == 0:
            continue
        x = rng.normal(loc=i, scale=0.055, size=len(vals))
        ax.scatter(x, vals, alpha=0.65, s=26)

    ax.set_title(f"{metric}: high vs low dispersion")
    ax.set_ylabel(metric)
    ax.grid(axis="y", alpha=0.3)

    annotation = (
        f"High - Low mean = {stat_row.get('high_minus_low_mean', np.nan):.4g}\n"
        f"High - Low median = {stat_row.get('high_minus_low_median', np.nan):.4g}\n"
        f"Mann-Whitney p = {stat_row.get('mannwhitney_p', np.nan):.3g}\n"
        f"Cliff's delta = {stat_row.get('cliffs_delta_high_vs_low', np.nan):.3g}"
    )

    ax.text(
        0.02,
        0.98,
        annotation,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round", alpha=0.12),
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_binary_metric(
    metric: str,
    high: np.ndarray,
    low: np.ndarray,
    stat_row: Dict,
    output_path: Path,
) -> None:
    """
    Bar plot of binary flag prevalence by low/high dispersion.
    """
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    high = (high[np.isfinite(high)] > 0).astype(int)
    low = (low[np.isfinite(low)] > 0).astype(int)

    n_low = len(low)
    n_high = len(high)
    low_pos = int(low.sum())
    high_pos = int(high.sum())

    low_prop = low_pos / n_low if n_low else np.nan
    high_prop = high_pos / n_high if n_high else np.nan

    low_ci = wilson_ci(low_pos, n_low)
    high_ci = wilson_ci(high_pos, n_high)

    props = [low_prop, high_prop]
    lower_err = [
        low_prop - low_ci[0] if np.isfinite(low_prop) else 0,
        high_prop - high_ci[0] if np.isfinite(high_prop) else 0,
    ]
    upper_err = [
        low_ci[1] - low_prop if np.isfinite(low_prop) else 0,
        high_ci[1] - high_prop if np.isfinite(high_prop) else 0,
    ]

    fig, ax = plt.subplots(figsize=(7.0, 5.3))

    ax.bar(
        [0, 1],
        props,
        yerr=[lower_err, upper_err],
        capsize=6,
        tick_label=[f"Low dispersion\nn={n_low}", f"High dispersion\nn={n_high}"],
    )

    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Proportion of reports with flag present")
    ax.set_title(f"{metric}: high vs low dispersion")
    ax.grid(axis="y", alpha=0.3)

    annotation = (
        f"High proportion = {high_prop:.3f}\n"
        f"Low proportion = {low_prop:.3f}\n"
        f"Risk diff = {stat_row.get('high_minus_low_proportion', np.nan):.3f}\n"
        f"Fisher exact p = {stat_row.get('fisher_exact_p', np.nan):.3g}\n"
        f"OR = {stat_row.get('odds_ratio_high_vs_low', np.nan):.3g}"
    )

    ax.text(
        0.02,
        0.98,
        annotation,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round", alpha=0.12),
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def prepare_dataframe(
    input_csv: Path,
    metrics: List[str],
    split_role: str,
    aggregate_cases: bool,
) -> pd.DataFrame:
    """
    Read, clean, optionally filter by split_role, and optionally aggregate to case level.
    """
    df = pd.read_csv(input_csv)

    if "dispersion_true_high_low" not in df.columns:
        raise ValueError("Input CSV must contain dispersion_true_high_low.")

    df["dispersion_true_high_low_norm"] = df["dispersion_true_high_low"].apply(parse_dispersion_label)
    df = df[df["dispersion_true_high_low_norm"].isin([0, 1])].copy()

    if split_role != "all":
        if "split_role" not in df.columns:
            raise ValueError("--split-role was set, but input CSV has no split_role column.")
        df = df[df["split_role"].astype(str).str.lower() == split_role].copy()

    for metric in metrics:
        if metric in df.columns:
            df[metric] = pd.to_numeric(df[metric], errors="coerce")

    if aggregate_cases:
        df = aggregate_to_case_level(df, metrics=metrics)

    return df


def run_analysis(args: argparse.Namespace) -> None:
    input_csv = Path(args.input_csv).expanduser().resolve()
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    parent = input_csv.parent
    output_dir = parent / args.output_folder
    output_dir.mkdir(parents=True, exist_ok=True)

    df = prepare_dataframe(
        input_csv=input_csv,
        metrics=METRICS,
        split_role=args.split_role,
        aggregate_cases=not args.no_case_aggregation,
    )

    available_metrics = [m for m in METRICS if m in df.columns]
    missing_metrics = [m for m in METRICS if m not in df.columns]

    if missing_metrics:
        print("[WARN] Missing metrics skipped:")
        for m in missing_metrics:
            print(f"  - {m}")

    manifest_rows = []
    stat_rows = []

    for metric in available_metrics:
        sub = df[["dispersion_true_high_low_norm", metric]].copy()
        sub = sub.dropna(subset=["dispersion_true_high_low_norm", metric])

        low = sub.loc[sub["dispersion_true_high_low_norm"] == 0, metric].to_numpy(dtype=float)
        high = sub.loc[sub["dispersion_true_high_low_norm"] == 1, metric].to_numpy(dtype=float)

        if len(low) == 0 or len(high) == 0:
            print(f"[WARN] Skipping {metric}: missing low or high group.")
            continue

        metric_type = "binary_flag" if metric in BINARY_METRICS else "continuous_or_count"

        if metric_type == "binary_flag":
            stat_row = compute_binary_stats(metric, high=high, low=low)
            plot_path = output_dir / f"{sanitize_filename(metric)}_high_vs_low_bar.png"
            plot_binary_metric(metric, high=high, low=low, stat_row=stat_row, output_path=plot_path)
        else:
            stat_row = compute_continuous_stats(
                metric,
                high=high,
                low=low,
                n_boot=args.n_boot,
                seed=args.seed,
            )
            plot_path = output_dir / f"{sanitize_filename(metric)}_high_vs_low_boxplot.png"
            plot_continuous_metric(
                metric,
                high=high,
                low=low,
                stat_row=stat_row,
                output_path=plot_path,
                seed=args.seed,
            )

        stat_rows.append(stat_row)

        manifest_rows.append(
            {
                "metric": metric,
                "metric_type": metric_type,
                "plot_path": str(plot_path),
                "n_low_nonmissing": len(low),
                "n_high_nonmissing": len(high),
            }
        )

        print(f"[OK] {metric} -> {plot_path.name}")

    stats_df = pd.DataFrame(stat_rows)
    if len(stats_df):
        stats_df["main_p_fdr_bh"] = benjamini_hochberg(stats_df["main_p"])

        # Sort strongest findings first, while keeping NaNs at bottom.
        stats_df = stats_df.sort_values(
            ["main_p_fdr_bh", "main_p", "metric"],
            na_position="last",
        )

    manifest_df = pd.DataFrame(manifest_rows)

    stats_path = output_dir / "audit_metric_high_low_statistics.csv"
    fdr_path = output_dir / "audit_metric_high_low_statistics_fdr.csv"
    manifest_path = output_dir / "audit_metric_high_low_manifest.csv"

    stats_df.to_csv(stats_path, index=False)
    stats_df.to_csv(fdr_path, index=False)
    manifest_df.to_csv(manifest_path, index=False)

    readme_path = output_dir / "README.txt"
    with open(readme_path, "w") as f:
        f.write(
            "MRI audit high-vs-low dispersion metric comparison\n"
            "=================================================\n\n"
            f"Input CSV: {input_csv}\n"
            f"Rows analyzed after filtering/aggregation: {len(df)}\n"
            f"Split role filter: {args.split_role}\n"
            f"Case-level aggregation: {not args.no_case_aggregation}\n"
            f"Bootstrap iterations: {args.n_boot}\n\n"
            "Interpretation notes:\n"
            "- For continuous/count/density metrics, the main test is Mann-Whitney U.\n"
            "- Welch t-test p-values are also reported as a mean-difference sensitivity check.\n"
            "- Cliff's delta > 0 means high-dispersion cases tend to have larger values.\n"
            "- For binary has_* flags, the main test is Fisher exact.\n"
            "- main_p_fdr_bh is Benjamini-Hochberg corrected across all tested metrics.\n"
            "- By default, repeated split-level appearances are aggregated to one case-level row.\n"
        )

    print("\nDone.")
    print(f"Output folder: {output_dir}")
    print(f"Statistics:    {stats_path}")
    print(f"Manifest:      {manifest_path}")
    print(f"README:        {readme_path}")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare MRI audit metrics between low- and high-dispersion cases."
    )

    parser.add_argument(
        "--input-csv",
        default=DEFAULT_INPUT_CSV,
        help="Path to mri_audit_case_table_all_resumed_splits.csv.",
    )

    parser.add_argument(
        "--output-folder",
        default="mri_audit_high_low_metric_plots",
        help="Name of the new folder to create inside the input CSV parent directory.",
    )

    parser.add_argument(
        "--split-role",
        choices=["all", "train", "test"],
        default="all",
        help=(
            "Analyze all rows, only train rows, or only test rows. "
            "Use 'test' for held-out-only descriptive diagnostics."
        ),
    )

    parser.add_argument(
        "--no-case-aggregation",
        action="store_true",
        help=(
            "Do not aggregate repeated case_id,row_index rows. "
            "Use this only if you intentionally want split-row-level statistics."
        ),
    )

    parser.add_argument(
        "--n-boot",
        type=int,
        default=5000,
        help="Bootstrap iterations for continuous metric difference CIs.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=17,
        help="Random seed for jitter and bootstrap resampling.",
    )

    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    run_analysis(args)


if __name__ == "__main__":
    main()