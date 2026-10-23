#!/usr/bin/env python3
"""
Compare pathology-report audit metrics against MRI-report audit metrics.

This script is designed for the downstream leakage-aware audit outputs produced
from completed outer_split_* artifacts. It does NOT rerun SecureGPT, does NOT
rediscover lexicons, does NOT relearn pathology-informed weights, and does NOT
touch any held-out pathology for MRI-only model inference. It simply compares
already-generated per-case/per-split audit metrics between two modalities:

  1) MRI audit case table, for example:
     mri_audit_case_table_all_resumed_splits.csv

  2) Pathology audit case table, for example:
     pathology_audit_case_table_all_resumed_splits.csv

The comparison is paired by split/case/row whenever possible. For each intended
metric, the script maps the MRI metric name to the corresponding pathology metric
name, merges the two tables, computes paired statistics, and generates plots.

Default metric list mirrors the MRI underperformance audit:
  - report length
  - section / response / measurement flags
  - phrase feature counts
  - canonical group/concept counts
  - spatial / morphology / response concept burden
  - distribution concept burden
  - ambiguity concept burden
  - uncertainty / negation burden
  - density-normalized extraction metrics

Outputs
-------
out_dir/
  metric_mapping.csv
  paired_metric_table_split_rows.csv
  paired_metric_table_case_level.csv
  path_vs_mri_metric_stats_split_rows.csv
  path_vs_mri_metric_stats_case_level.csv
  path_vs_mri_metric_stats_by_dispersion_case_level.csv
  path_vs_mri_metric_stats_by_dispersion_split_rows.csv
  path_vs_mri_audit_comparison_summary.txt
  plots/
    <metric>_paired_boxplot_case_level.png
    <metric>_difference_hist_case_level.png
    <metric>_scatter_case_level.png
    <metric>_by_dispersion_case_level.png
    primary_pvalue_overview_case_level.png

Typical usage
-------------
python compare.py \
  --mri-audit-path /Users/lukezhao/projects/onc/out/securegpt_dispersion_nested_path_informed_eval7_resumed3/mri_audit_case_table_all_resumed_splits.csv \
  --path-audit-path /Users/lukezhao/projects/onc/out/pathology_underperformance_audit/pathology_audit_case_table_all_resumed_splits.csv

If you want to force split-level paired rows rather than case-level summaries
for plots:
python compare_pathology_mri_audit_metrics.py \
  --mri-audit-path .../mri_audit_case_table_all_resumed_splits.csv \
  --path-audit-path .../pathology_audit_case_table_all_resumed_splits.csv \
  --plot-level split_rows

Author: generated for leakage-aware breast cancer MRI/pathology audit analysis.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

try:
    from scipy.stats import binomtest as _binomtest
except ImportError:  # pragma: no cover
    _binomtest = None


DEFAULT_MRI_AUDIT_PATH = (
    "/Users/lukezhao/projects/onc/out/"
    "securegpt_dispersion_nested_path_informed_eval7_resumed3/"
    "mri_audit_case_table_all_resumed_splits.csv"
)

DEFAULT_PATH_AUDIT_PATH = (
    "/Users/lukezhao/projects/onc/out/"
    "pathology_audit_case_table_all_resumed_splits.csv"
)

DEFAULT_INTENDED_METRICS = [
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

DEFAULT_LABEL_COLS = [
    "dispersion_true",
    "dispersion_true_high_low",
    "relapse_true",
    "split_role",
    "source_split_id",
]

EPS = 1e-12


@dataclass
class MetricMapping:
    metric: str
    mri_col: Optional[str]
    path_col: Optional[str]
    status: str
    notes: str = ""


# -----------------------------------------------------------------------------
# General utilities
# -----------------------------------------------------------------------------


def ordered_unique(xs: Iterable[str]) -> List[str]:
    seen = set()
    out = []
    for x in xs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def sanitize_filename(s: str) -> str:
    s = str(s)
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s)
    return s.strip("_")[:180] or "metric"


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def read_csv_required(path: str, label: str) -> pd.DataFrame:
    if not path:
        raise ValueError(f"Missing path for {label}.")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{label} does not exist: {path}")
    df = pd.read_csv(path)
    if len(df) == 0:
        print(f"[WARN] {label} is empty: {path}")
    return df


def infer_default_out_dir(mri_audit_path: str, path_audit_path: str) -> str:
    # Put results next to the pathology audit by default, because this is the
    # newer artifact in this workflow.
    parent = os.path.dirname(os.path.abspath(path_audit_path)) or os.getcwd()
    return os.path.join(parent, "path_vs_mri_audit_metric_comparison")


def available_key_cols(mri_df: pd.DataFrame, path_df: pd.DataFrame, requested: Sequence[str]) -> List[str]:
    common = [c for c in requested if c in mri_df.columns and c in path_df.columns]
    if "case_id" not in common and "row_index" not in common:
        raise ValueError(
            "Could not find usable merge keys. Expected at least case_id or row_index "
            "in both MRI and pathology audit tables."
        )
    return common


def first_nonnull(values: Sequence[Any]) -> Any:
    for v in values:
        if pd.notna(v):
            return v
    return np.nan


def coalesce_label_columns(df: pd.DataFrame, label_cols: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for label in label_cols:
        candidates = [c for c in [label, f"{label}_mri", f"{label}_path"] if c in out.columns]
        if not candidates:
            continue
        if label not in out.columns or len(candidates) > 1:
            out[label] = out[candidates].bfill(axis=1).iloc[:, 0]
        drop_cols = [c for c in candidates if c != label]
        out = out.drop(columns=drop_cols, errors="ignore")
    return out


def is_binary_pair(x: pd.Series, y: pd.Series, metric: str) -> bool:
    vals = pd.concat([safe_numeric(x), safe_numeric(y)], ignore_index=True).dropna().unique()
    if len(vals) == 0:
        return False
    vals_set = set(float(v) for v in vals)
    return vals_set.issubset({0.0, 1.0}) and (
        metric.startswith("has_") or metric.endswith("_missing") or metric.endswith("_true")
    )


def exact_binomial_two_sided(k: int, n: int, p: float = 0.5) -> float:
    if n <= 0:
        return np.nan
    if _binomtest is not None:
        return float(_binomtest(k=k, n=n, p=p, alternative="two-sided").pvalue)
    # Compatibility fallback for older SciPy.
    return float(stats.binom_test(k, n=n, p=p, alternative="two-sided"))  # type: ignore[attr-defined]


def benjamini_hochberg(p_values: Sequence[float]) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    q = np.full_like(p, np.nan, dtype=float)
    finite = np.isfinite(p)
    if finite.sum() == 0:
        return q
    idx = np.where(finite)[0]
    p_finite = p[idx]
    order = np.argsort(p_finite)
    ranked = p_finite[order]
    m = len(ranked)
    q_ranked = ranked * m / np.arange(1, m + 1)
    q_ranked = np.minimum.accumulate(q_ranked[::-1])[::-1]
    q_ranked = np.clip(q_ranked, 0.0, 1.0)
    q[idx[order]] = q_ranked
    return q


# -----------------------------------------------------------------------------
# Metric mapping
# -----------------------------------------------------------------------------


def candidate_pathology_cols_for_metric(metric: str) -> List[str]:
    """
    Given an MRI audit metric name, return plausible pathology-side column names.

    The pathology audit script may preserve generic names like n_phrase_present
    while changing MRI-specific names like mri_report_words -> path_report_words.
    This resolver intentionally accepts several naming conventions.
    """
    candidates: List[str] = []

    explicit: Dict[str, List[str]] = {
        "mri_report_missing": [
            "path_report_missing",
            "pathology_report_missing",
            "path_report_is_missing",
            "pathology_report_is_missing",
            "mri_report_missing",
            "report_missing",
        ],
        "mri_report_chars": [
            "path_report_chars",
            "pathology_report_chars",
            "path_chars",
            "report_chars",
            "mri_report_chars",
        ],
        "mri_report_words": [
            "path_report_words",
            "pathology_report_words",
            "path_words",
            "report_words",
            "mri_report_words",
        ],
        "mri_extraction_density": [
            "path_extraction_density",
            "pathology_extraction_density",
            "extraction_density",
            "mri_extraction_density",
        ],
    }
    candidates.extend(explicit.get(metric, []))

    if metric.startswith("mri_"):
        candidates.append(metric.replace("mri_", "path_", 1))
        candidates.append(metric.replace("mri_", "pathology_", 1))
        candidates.append(metric[4:])  # e.g. report_words
    else:
        # Most phrase/concept/flag metrics are intentionally modality-generic.
        candidates.append(metric)
        candidates.append(f"path_{metric}")
        candidates.append(f"pathology_{metric}")

    # Some pathology audits may use final-diagnosis or synoptic sections rather
    # than radiology-style findings/impression. These are fallbacks only; exact
    # same-named columns are preferred when present.
    pathology_section_fallbacks = {
        "has_findings_section": [
            "has_findings_section",
            "has_final_diagnosis_section",
            "has_diagnosis_section",
            "has_synoptic_section",
        ],
        "has_impression_section": [
            "has_impression_section",
            "has_final_diagnosis_section",
            "has_diagnosis_section",
            "has_synoptic_section",
        ],
        "has_comparison_section": [
            "has_comparison_section",
            "has_prior_language",
            "has_history_section",
            "has_clinical_history_section",
        ],
        "has_response_language": [
            "has_response_language",
            "has_treatment_effect_language",
            "has_residual_language",
        ],
        "has_measurement_language": [
            "has_measurement_language",
            "has_size_language",
            "has_dimension_language",
        ],
    }
    candidates.extend(pathology_section_fallbacks.get(metric, []))
    return ordered_unique(candidates)


def find_existing_col(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    colset = set(df.columns)
    for c in candidates:
        if c in colset:
            return c
    # Case-insensitive fallback.
    lower_to_col = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        hit = lower_to_col.get(str(c).lower())
        if hit is not None:
            return hit
    return None


def build_metric_mapping(
    mri_df: pd.DataFrame,
    path_df: pd.DataFrame,
    intended_metrics: Sequence[str],
) -> pd.DataFrame:
    rows: List[MetricMapping] = []
    for metric in intended_metrics:
        mri_col = find_existing_col(mri_df, [metric])
        path_col = find_existing_col(path_df, candidate_pathology_cols_for_metric(metric))

        status = "ok" if (mri_col is not None and path_col is not None) else "missing"
        notes = ""
        if mri_col is None:
            notes += "MRI column missing. "
        if path_col is None:
            notes += "Pathology equivalent column missing. "
        if status == "ok" and path_col != metric:
            notes += f"Pathology column mapped from {metric} to {path_col}."

        rows.append(MetricMapping(metric=metric, mri_col=mri_col, path_col=path_col, status=status, notes=notes.strip()))

    return pd.DataFrame([r.__dict__ for r in rows])


# -----------------------------------------------------------------------------
# Paired table construction
# -----------------------------------------------------------------------------


def make_paired_metric_table(
    mri_df: pd.DataFrame,
    path_df: pd.DataFrame,
    metric_map: pd.DataFrame,
    merge_keys: Sequence[str],
    label_cols: Sequence[str],
    join: str = "inner",
) -> pd.DataFrame:
    ok_map = metric_map[metric_map["status"] == "ok"].copy()
    if len(ok_map) == 0:
        raise ValueError("No metrics could be mapped between MRI and pathology audit tables.")

    mri_metric_rename = {
        str(row["mri_col"]): f"mri__{row['metric']}"
        for _, row in ok_map.iterrows()
    }
    path_metric_rename = {
        str(row["path_col"]): f"path__{row['metric']}"
        for _, row in ok_map.iterrows()
    }

    mri_keep = ordered_unique(
        list(merge_keys)
        + [c for c in label_cols if c in mri_df.columns]
        + list(mri_metric_rename.keys())
    )
    path_keep = ordered_unique(
        list(merge_keys)
        + [c for c in label_cols if c in path_df.columns]
        + list(path_metric_rename.keys())
    )

    mri_part = mri_df[mri_keep].copy().rename(columns=mri_metric_rename)
    path_part = path_df[path_keep].copy().rename(columns=path_metric_rename)

    for key in merge_keys:
        # Use string keys for stable matching across tables, except leave numeric
        # values untouched if both are already numeric-compatible.
        if key in ["case_id", "split_id", "source_split_id", "split_role"]:
            mri_part[key] = mri_part[key].astype(str)
            path_part[key] = path_part[key].astype(str)

    merged = mri_part.merge(
        path_part,
        on=list(merge_keys),
        how=join,
        suffixes=("_mri", "_path"),
        indicator=True,
    )
    merged = coalesce_label_columns(merged, label_cols)

    for metric in ok_map["metric"].tolist():
        for prefix in ["mri__", "path__"]:
            col = f"{prefix}{metric}"
            if col in merged.columns:
                merged[col] = safe_numeric(merged[col])

    return merged


def make_case_level_table(
    paired_split_rows: pd.DataFrame,
    metrics: Sequence[str],
    label_cols: Sequence[str],
) -> pd.DataFrame:
    group_keys = [c for c in ["case_id", "row_index"] if c in paired_split_rows.columns]
    if not group_keys:
        raise ValueError("Cannot make case-level table without case_id or row_index.")

    metric_cols = []
    for metric in metrics:
        metric_cols.extend([f"mri__{metric}", f"path__{metric}"])
    metric_cols = [c for c in metric_cols if c in paired_split_rows.columns]

    agg_spec: Dict[str, Any] = {c: "mean" for c in metric_cols}
    for label in label_cols:
        if label in paired_split_rows.columns and label not in group_keys:
            agg_spec[label] = lambda s: first_nonnull(list(s))

    # Keep a small provenance count to show how many split rows contributed.
    out = paired_split_rows.groupby(group_keys, dropna=False).agg(agg_spec).reset_index()
    n_rows = paired_split_rows.groupby(group_keys, dropna=False).size().reset_index(name="n_split_rows_represented")
    out = out.merge(n_rows, on=group_keys, how="left")
    return out


# -----------------------------------------------------------------------------
# Statistics
# -----------------------------------------------------------------------------


def paired_metric_stats_one(df: pd.DataFrame, metric: str, group_name: str = "overall", group_value: Any = "all") -> Dict[str, Any]:
    mri_col = f"mri__{metric}"
    path_col = f"path__{metric}"

    rec: Dict[str, Any] = {
        "group_name": group_name,
        "group_value": group_value,
        "metric": metric,
        "mri_col": mri_col,
        "path_col": path_col,
        "n_pairs": 0,
        "n_missing_mri": np.nan,
        "n_missing_path": np.nan,
        "mri_mean": np.nan,
        "path_mean": np.nan,
        "mri_median": np.nan,
        "path_median": np.nan,
        "mean_path_minus_mri": np.nan,
        "median_path_minus_mri": np.nan,
        "paired_cohen_dz": np.nan,
        "pearson_r": np.nan,
        "pearson_p": np.nan,
        "spearman_rho": np.nan,
        "spearman_p": np.nan,
        "binary_b_mri1_path0": np.nan,
        "binary_c_mri0_path1": np.nan,
        "primary_test": "",
        "primary_p": np.nan,
        "paired_t_p": np.nan,
        "wilcoxon_p": np.nan,
        "sign_test_p": np.nan,
        "mcnemar_exact_p": np.nan,
    }

    if mri_col not in df.columns or path_col not in df.columns:
        rec["primary_test"] = "missing_columns"
        return rec

    raw_x = safe_numeric(df[mri_col])
    raw_y = safe_numeric(df[path_col])
    rec["n_missing_mri"] = int(raw_x.isna().sum())
    rec["n_missing_path"] = int(raw_y.isna().sum())

    mask = raw_x.notna() & raw_y.notna()
    x = raw_x[mask].astype(float)
    y = raw_y[mask].astype(float)
    rec["n_pairs"] = int(mask.sum())

    if rec["n_pairs"] == 0:
        rec["primary_test"] = "no_complete_pairs"
        return rec

    diff = y - x
    rec["mri_mean"] = float(x.mean())
    rec["path_mean"] = float(y.mean())
    rec["mri_median"] = float(x.median())
    rec["path_median"] = float(y.median())
    rec["mean_path_minus_mri"] = float(diff.mean())
    rec["median_path_minus_mri"] = float(diff.median())

    diff_sd = float(diff.std(ddof=1)) if len(diff) >= 2 else np.nan
    if np.isfinite(diff_sd) and diff_sd > EPS:
        rec["paired_cohen_dz"] = float(diff.mean() / diff_sd)

    if len(x) >= 3 and x.nunique(dropna=True) > 1 and y.nunique(dropna=True) > 1:
        try:
            r, p = stats.pearsonr(x, y)
            rec["pearson_r"] = float(r)
            rec["pearson_p"] = float(p)
        except Exception:
            pass
        try:
            rho, p = stats.spearmanr(x, y)
            rec["spearman_rho"] = float(rho)
            rec["spearman_p"] = float(p)
        except Exception:
            pass

    if is_binary_pair(x, y, metric):
        xi = x.round().astype(int)
        yi = y.round().astype(int)
        b = int(((xi == 1) & (yi == 0)).sum())  # MRI positive, pathology negative
        c = int(((xi == 0) & (yi == 1)).sum())  # MRI negative, pathology positive
        rec["binary_b_mri1_path0"] = b
        rec["binary_c_mri0_path1"] = c
        if b + c == 0:
            rec["mcnemar_exact_p"] = 1.0
        else:
            rec["mcnemar_exact_p"] = exact_binomial_two_sided(k=min(b, c), n=b + c, p=0.5)
        rec["primary_test"] = "exact_mcnemar_binomial"
        rec["primary_p"] = rec["mcnemar_exact_p"]
        return rec

    # Continuous/count paired tests.
    if len(diff) >= 2:
        try:
            t_res = stats.ttest_rel(y, x, nan_policy="omit")
            rec["paired_t_p"] = float(t_res.pvalue)
        except Exception:
            pass

    nonzero_diff = diff[np.abs(diff) > EPS]
    if len(diff) >= 3:
        if len(nonzero_diff) == 0:
            rec["wilcoxon_p"] = 1.0
        else:
            try:
                w_res = stats.wilcoxon(y, x, zero_method="wilcox", alternative="two-sided")
                rec["wilcoxon_p"] = float(w_res.pvalue)
            except Exception:
                pass

    n_pos = int((diff > EPS).sum())
    n_neg = int((diff < -EPS).sum())
    if n_pos + n_neg > 0:
        rec["sign_test_p"] = exact_binomial_two_sided(k=min(n_pos, n_neg), n=n_pos + n_neg, p=0.5)
    elif len(diff) > 0:
        rec["sign_test_p"] = 1.0

    if np.isfinite(rec["wilcoxon_p"]):
        rec["primary_test"] = "paired_wilcoxon"
        rec["primary_p"] = rec["wilcoxon_p"]
    elif np.isfinite(rec["sign_test_p"]):
        rec["primary_test"] = "paired_sign_test"
        rec["primary_p"] = rec["sign_test_p"]
    elif np.isfinite(rec["paired_t_p"]):
        rec["primary_test"] = "paired_t_test"
        rec["primary_p"] = rec["paired_t_p"]
    else:
        rec["primary_test"] = "insufficient_variation"

    return rec


def compute_paired_stats(
    df: pd.DataFrame,
    metrics: Sequence[str],
    group_name: str = "overall",
    group_value: Any = "all",
) -> pd.DataFrame:
    rows = [paired_metric_stats_one(df, metric, group_name=group_name, group_value=group_value) for metric in metrics]
    out = pd.DataFrame(rows)
    if "primary_p" in out.columns:
        out["primary_q_bh"] = benjamini_hochberg(out["primary_p"].values)
    return out


def compute_stratified_stats(
    df: pd.DataFrame,
    metrics: Sequence[str],
    stratify_cols: Sequence[str],
) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for col in stratify_cols:
        if col not in df.columns:
            continue
        sub = df[df[col].notna()].copy()
        if len(sub) == 0:
            continue
        for value, part in sub.groupby(col, dropna=True):
            stats_df = compute_paired_stats(part, metrics, group_name=col, group_value=value)
            rows.append(stats_df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# -----------------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------------


def sample_for_plot(df: pd.DataFrame, n: int, seed: int = 17) -> pd.DataFrame:
    if len(df) <= n:
        return df
    return df.sample(n=n, random_state=seed)


def label_for_dispersion(v: Any) -> str:
    try:
        fv = float(v)
        if fv == 0:
            return "Low dispersion"
        if fv == 1:
            return "High dispersion"
    except Exception:
        pass
    return str(v)


def plot_paired_boxplot(
    df: pd.DataFrame,
    metric: str,
    out_path: str,
    title_suffix: str,
    max_paired_lines: int = 250,
) -> None:
    mri_col = f"mri__{metric}"
    path_col = f"path__{metric}"
    if mri_col not in df.columns or path_col not in df.columns:
        return

    sub = df[[mri_col, path_col]].dropna().copy()
    if len(sub) == 0:
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.boxplot([sub[mri_col].values, sub[path_col].values], labels=["MRI", "Pathology"], showfliers=False)

    line_sub = sample_for_plot(sub, max_paired_lines)
    for _, row in line_sub.iterrows():
        ax.plot([1, 2], [row[mri_col], row[path_col]], alpha=0.12, linewidth=0.8)

    # Add points after lines.
    rng = np.random.default_rng(17)
    x1 = 1 + rng.normal(0, 0.035, size=len(line_sub))
    x2 = 2 + rng.normal(0, 0.035, size=len(line_sub))
    ax.scatter(x1, line_sub[mri_col], s=12, alpha=0.45)
    ax.scatter(x2, line_sub[path_col], s=12, alpha=0.45)

    ax.set_title(f"{metric}: MRI vs pathology\n{title_suffix}")
    ax.set_ylabel(metric)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_difference_hist(df: pd.DataFrame, metric: str, out_path: str, title_suffix: str) -> None:
    mri_col = f"mri__{metric}"
    path_col = f"path__{metric}"
    if mri_col not in df.columns or path_col not in df.columns:
        return
    sub = df[[mri_col, path_col]].dropna().copy()
    if len(sub) == 0:
        return

    diff = sub[path_col] - sub[mri_col]
    fig, ax = plt.subplots(figsize=(7, 5))
    bins = min(30, max(8, int(np.sqrt(len(diff)))))
    ax.hist(diff.values, bins=bins, alpha=0.85)
    ax.axvline(0, linestyle="--", linewidth=1.2)
    ax.set_title(f"{metric}: pathology minus MRI\n{title_suffix}")
    ax.set_xlabel("Pathology value - MRI value")
    ax.set_ylabel("Number of paired cases")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_scatter(df: pd.DataFrame, metric: str, out_path: str, title_suffix: str) -> None:
    mri_col = f"mri__{metric}"
    path_col = f"path__{metric}"
    if mri_col not in df.columns or path_col not in df.columns:
        return

    sub = df[[mri_col, path_col]].dropna().copy()
    if len(sub) < 3:
        return

    # For binary variables, a scatter is not very informative; the paired
    # boxplot/difference plot and McNemar table are more useful.
    if is_binary_pair(sub[mri_col], sub[path_col], metric):
        return

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(sub[mri_col], sub[path_col], s=18, alpha=0.55)
    lo = float(min(sub[mri_col].min(), sub[path_col].min()))
    hi = float(max(sub[mri_col].max(), sub[path_col].max()))
    if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
        ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.0)
    ax.set_title(f"{metric}: paired case scatter\n{title_suffix}")
    ax.set_xlabel("MRI")
    ax.set_ylabel("Pathology")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_by_dispersion(df: pd.DataFrame, metric: str, out_path: str, title_suffix: str) -> None:
    if "dispersion_true_high_low" not in df.columns:
        return

    mri_col = f"mri__{metric}"
    path_col = f"path__{metric}"
    if mri_col not in df.columns or path_col not in df.columns:
        return

    sub = df[["dispersion_true_high_low", mri_col, path_col]].dropna().copy()
    if len(sub) == 0 or sub["dispersion_true_high_low"].nunique() < 2:
        return

    groups = []
    labels = []
    for group_value, part in sorted(sub.groupby("dispersion_true_high_low"), key=lambda kv: str(kv[0])):
        groups.append(part[mri_col].values)
        labels.append(f"{label_for_dispersion(group_value)}\nMRI")
        groups.append(part[path_col].values)
        labels.append(f"{label_for_dispersion(group_value)}\nPath")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.boxplot(groups, labels=labels, showfliers=False)
    ax.set_title(f"{metric}: MRI vs pathology within dispersion groups\n{title_suffix}")
    ax.set_ylabel(metric)
    ax.grid(axis="y", alpha=0.25)
    fig.autofmt_xdate(rotation=20)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_pvalue_overview(stats_df: pd.DataFrame, out_path: str, title: str, top_n: int = 30) -> None:
    if stats_df is None or len(stats_df) == 0 or "primary_p" not in stats_df.columns:
        return
    sub = stats_df.copy()
    sub = sub[np.isfinite(pd.to_numeric(sub["primary_p"], errors="coerce"))].copy()
    if len(sub) == 0:
        return
    sub["primary_p"] = safe_numeric(sub["primary_p"])
    sub["neglog10_p"] = -np.log10(np.clip(sub["primary_p"], 1e-300, 1.0))
    sub = sub.sort_values("primary_p", ascending=True).head(top_n)
    sub = sub.sort_values("neglog10_p", ascending=True)

    fig_h = max(5, 0.28 * len(sub) + 1.5)
    fig, ax = plt.subplots(figsize=(9, fig_h))
    ax.barh(sub["metric"], sub["neglog10_p"])
    ax.axvline(-math.log10(0.05), linestyle="--", linewidth=1.0)
    ax.set_xlabel("-log10(primary paired-test p-value)")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def generate_metric_plots(
    df: pd.DataFrame,
    metrics: Sequence[str],
    out_dir: str,
    level_name: str,
    max_paired_lines: int,
) -> None:
    ensure_dir(out_dir)
    title_suffix = f"{level_name}; n={len(df)} paired rows"

    for metric in metrics:
        safe = sanitize_filename(metric)
        plot_paired_boxplot(
            df,
            metric,
            os.path.join(out_dir, f"{safe}_paired_boxplot_{level_name}.png"),
            title_suffix=title_suffix,
            max_paired_lines=max_paired_lines,
        )
        plot_difference_hist(
            df,
            metric,
            os.path.join(out_dir, f"{safe}_difference_hist_{level_name}.png"),
            title_suffix=title_suffix,
        )
        plot_scatter(
            df,
            metric,
            os.path.join(out_dir, f"{safe}_scatter_{level_name}.png"),
            title_suffix=title_suffix,
        )
        plot_by_dispersion(
            df,
            metric,
            os.path.join(out_dir, f"{safe}_by_dispersion_{level_name}.png"),
            title_suffix=title_suffix,
        )


# -----------------------------------------------------------------------------
# Summary writing
# -----------------------------------------------------------------------------


def summarize_top_findings(stats_df: pd.DataFrame, n: int = 12) -> str:
    if stats_df is None or len(stats_df) == 0:
        return "No statistics were computed.\n"
    sub = stats_df.copy()
    sub["primary_p"] = safe_numeric(sub["primary_p"])
    sub = sub[np.isfinite(sub["primary_p"])].copy()
    if len(sub) == 0:
        return "No finite primary p-values were computed.\n"
    sub = sub.sort_values("primary_p", ascending=True).head(n)

    lines = []
    for _, row in sub.iterrows():
        q = row.get("primary_q_bh", np.nan)
        q_text = "NA" if not np.isfinite(q) else f"{q:.3g}"
        lines.append(
            f"- {row['metric']}: mean(path-MRI)={row.get('mean_path_minus_mri', np.nan):.4g}, "
            f"primary_test={row.get('primary_test', '')}, p={row.get('primary_p', np.nan):.3g}, q={q_text}, "
            f"n_pairs={int(row.get('n_pairs', 0))}"
        )
    return "\n".join(lines) + "\n"


def write_summary(
    out_path: str,
    args: argparse.Namespace,
    metric_map: pd.DataFrame,
    split_stats: pd.DataFrame,
    case_stats: pd.DataFrame,
    paired_split_rows: pd.DataFrame,
    paired_case_level: pd.DataFrame,
) -> None:
    ok_n = int((metric_map["status"] == "ok").sum())
    missing_n = int((metric_map["status"] != "ok").sum())

    with open(out_path, "w") as f:
        f.write("Pathology vs MRI audit metric comparison\n")
        f.write("=" * 48 + "\n\n")
        f.write(f"MRI audit path: {args.mri_audit_path}\n")
        f.write(f"Pathology audit path: {args.path_audit_path}\n")
        f.write(f"Output directory: {args.out_dir}\n")
        f.write(f"Merge keys: {', '.join(args.merge_keys)}\n")
        f.write(f"Join mode: {args.join}\n\n")

        f.write("Metric mapping\n")
        f.write("-" * 48 + "\n")
        f.write(f"Mapped metrics: {ok_n}\n")
        f.write(f"Missing metrics: {missing_n}\n")
        if missing_n:
            f.write("\nMissing/unmapped metrics:\n")
            for _, row in metric_map[metric_map["status"] != "ok"].iterrows():
                f.write(f"- {row['metric']}: {row.get('notes', '')}\n")
        f.write("\n")

        f.write("Paired table sizes\n")
        f.write("-" * 48 + "\n")
        f.write(f"Split-row paired rows: {len(paired_split_rows)}\n")
        f.write(f"Case-level paired rows: {len(paired_case_level)}\n\n")

        f.write("Top case-level paired differences\n")
        f.write("-" * 48 + "\n")
        f.write(summarize_top_findings(case_stats))
        f.write("\n")

        f.write("Top split-row paired differences\n")
        f.write("-" * 48 + "\n")
        f.write(summarize_top_findings(split_stats))
        f.write("\n")

        f.write("Interpretation notes\n")
        f.write("-" * 48 + "\n")
        f.write(
            "These are paired descriptive audit comparisons, not predictive model results. "
            "They quantify whether the pathology audit tends to contain more or less of the "
            "same lexical/concept evidence than the MRI audit for the same case/split rows. "
            "Case-level statistics are usually preferable for inference because resumed split "
            "tables may contain repeated appearances of the same patient/case across outer "
            "splits. Split-row statistics are retained for transparency and debugging.\n"
        )


# -----------------------------------------------------------------------------
# CLI / main
# -----------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare pathology audit metrics against MRI audit metrics with paired statistics and plots."
    )
    parser.add_argument("--mri-audit-path", default=DEFAULT_MRI_AUDIT_PATH, help="Path to mri_audit_case_table_all_resumed_splits.csv.")
    parser.add_argument("--path-audit-path", default=DEFAULT_PATH_AUDIT_PATH, help="Path to pathology_audit_case_table_all_resumed_splits.csv.")
    parser.add_argument("--out-dir", default=None, help="Output directory. Defaults to a folder next to the pathology audit CSV.")
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=DEFAULT_INTENDED_METRICS,
        help="MRI-side metric names to compare. Pathology equivalents are auto-mapped.",
    )
    parser.add_argument(
        "--merge-keys",
        nargs="+",
        default=["split_id", "case_id", "row_index"],
        help="Columns used to pair MRI and pathology rows. Defaults to split_id case_id row_index.",
    )
    parser.add_argument(
        "--label-cols",
        nargs="+",
        default=DEFAULT_LABEL_COLS,
        help="Metadata/label columns to carry through when available.",
    )
    parser.add_argument(
        "--stratify-cols",
        nargs="*",
        default=["dispersion_true_high_low"],
        help="Columns for optional stratified paired statistics.",
    )
    parser.add_argument(
        "--join",
        default="inner",
        choices=["inner", "left", "right", "outer"],
        help="How to join MRI and pathology tables on merge keys. Inner is safest for paired comparisons.",
    )
    parser.add_argument(
        "--plot-level",
        default="case_level",
        choices=["case_level", "split_rows", "both"],
        help="Which paired table to use for plots.",
    )
    parser.add_argument(
        "--max-paired-lines",
        type=int,
        default=250,
        help="Maximum number of paired line segments drawn per metric plot to avoid overplotting.",
    )
    parser.add_argument("--no-plots", action="store_true", help="Write CSV/stat outputs but skip plot generation.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.out_dir is None:
        args.out_dir = infer_default_out_dir(args.mri_audit_path, args.path_audit_path)
    ensure_dir(args.out_dir)
    plots_dir = ensure_dir(os.path.join(args.out_dir, "plots"))

    print("=" * 100)
    print("[COMPARE] Pathology vs MRI audit metric comparison")
    print(f"[COMPARE] MRI audit:       {args.mri_audit_path}")
    print(f"[COMPARE] Pathology audit: {args.path_audit_path}")
    print(f"[COMPARE] Output dir:      {args.out_dir}")

    mri_df = read_csv_required(args.mri_audit_path, "MRI audit table")
    path_df = read_csv_required(args.path_audit_path, "Pathology audit table")
    print(f"[COMPARE] MRI rows={len(mri_df)} cols={len(mri_df.columns)}")
    print(f"[COMPARE] Path rows={len(path_df)} cols={len(path_df.columns)}")

    # Use only merge keys that exist in both tables. Usually this will be:
    # split_id, case_id, row_index.
    keys = available_key_cols(mri_df, path_df, args.merge_keys)
    args.merge_keys = keys
    print(f"[COMPARE] Using merge keys: {keys}")

    metric_map = build_metric_mapping(mri_df, path_df, args.metrics)
    metric_map_path = os.path.join(args.out_dir, "metric_mapping.csv")
    metric_map.to_csv(metric_map_path, index=False)
    print(f"[COMPARE] Metric mapping written: {metric_map_path}")

    mapped_metrics = metric_map.loc[metric_map["status"] == "ok", "metric"].astype(str).tolist()
    if not mapped_metrics:
        print("[ERROR] No intended metrics could be mapped between MRI and pathology tables.")
        print("[ERROR] See metric_mapping.csv for missing columns.")
        return 2
    print(f"[COMPARE] Mapped {len(mapped_metrics)} / {len(args.metrics)} intended metrics.")

    paired_split_rows = make_paired_metric_table(
        mri_df=mri_df,
        path_df=path_df,
        metric_map=metric_map,
        merge_keys=keys,
        label_cols=args.label_cols,
        join=args.join,
    )
    split_rows_path = os.path.join(args.out_dir, "paired_metric_table_split_rows.csv")
    paired_split_rows.to_csv(split_rows_path, index=False)
    print(f"[COMPARE] Paired split-row table written: {split_rows_path} rows={len(paired_split_rows)}")

    paired_case_level = make_case_level_table(paired_split_rows, mapped_metrics, args.label_cols)
    case_level_path = os.path.join(args.out_dir, "paired_metric_table_case_level.csv")
    paired_case_level.to_csv(case_level_path, index=False)
    print(f"[COMPARE] Paired case-level table written: {case_level_path} rows={len(paired_case_level)}")

    split_stats = compute_paired_stats(paired_split_rows, mapped_metrics, group_name="overall", group_value="all")
    case_stats = compute_paired_stats(paired_case_level, mapped_metrics, group_name="overall", group_value="all")

    split_stats_path = os.path.join(args.out_dir, "path_vs_mri_metric_stats_split_rows.csv")
    case_stats_path = os.path.join(args.out_dir, "path_vs_mri_metric_stats_case_level.csv")
    split_stats.to_csv(split_stats_path, index=False)
    case_stats.to_csv(case_stats_path, index=False)
    print(f"[COMPARE] Split-row stats written: {split_stats_path}")
    print(f"[COMPARE] Case-level stats written: {case_stats_path}")

    split_strat = compute_stratified_stats(paired_split_rows, mapped_metrics, args.stratify_cols)
    case_strat = compute_stratified_stats(paired_case_level, mapped_metrics, args.stratify_cols)

    if len(split_strat):
        p = os.path.join(args.out_dir, "path_vs_mri_metric_stats_by_dispersion_split_rows.csv")
        split_strat.to_csv(p, index=False)
        print(f"[COMPARE] Stratified split-row stats written: {p}")
    if len(case_strat):
        p = os.path.join(args.out_dir, "path_vs_mri_metric_stats_by_dispersion_case_level.csv")
        case_strat.to_csv(p, index=False)
        print(f"[COMPARE] Stratified case-level stats written: {p}")

    if not args.no_plots:
        if args.plot_level in {"case_level", "both"}:
            generate_metric_plots(
                paired_case_level,
                mapped_metrics,
                out_dir=plots_dir,
                level_name="case_level",
                max_paired_lines=args.max_paired_lines,
            )
            plot_pvalue_overview(
                case_stats,
                os.path.join(plots_dir, "primary_pvalue_overview_case_level.png"),
                title="Strongest paired pathology-vs-MRI audit differences: case level",
            )
        if args.plot_level in {"split_rows", "both"}:
            generate_metric_plots(
                paired_split_rows,
                mapped_metrics,
                out_dir=plots_dir,
                level_name="split_rows",
                max_paired_lines=args.max_paired_lines,
            )
            plot_pvalue_overview(
                split_stats,
                os.path.join(plots_dir, "primary_pvalue_overview_split_rows.png"),
                title="Strongest paired pathology-vs-MRI audit differences: split rows",
            )
        print(f"[COMPARE] Plots written under: {plots_dir}")

    summary_path = os.path.join(args.out_dir, "path_vs_mri_audit_comparison_summary.txt")
    write_summary(
        out_path=summary_path,
        args=args,
        metric_map=metric_map,
        split_stats=split_stats,
        case_stats=case_stats,
        paired_split_rows=paired_split_rows,
        paired_case_level=paired_case_level,
    )
    print(f"[COMPARE] Summary written: {summary_path}")

    print("[COMPARE] Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())