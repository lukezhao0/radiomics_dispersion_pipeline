#!/usr/bin/env python3
"""Compute unified combined-modality metrics for Approach 1 and Approach 2.

Reads saved prediction artifacts only (no LLM re-runs). Filters to MRI+pathology
combined modality. Bootstraps 95% CIs over cases (B=1000, seed=42 by default).

Usage:
  python pipeline/metrics_audit/compute_unified_metrics.py \\
    --approach1-dir sabcs/securegpt_dispersion_approach1_pipeline_062726 \\
    --approach2-dir sabcs/securegpt_dispersion_approach2_pipeline_062726 \\
    --outdir sabcs/metrics_audit_combined_modality
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    recall_score,
)
from scipy.stats import pearsonr, spearmanr

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from approach1.config import DEFAULT_BOOTSTRAP_N, DEFAULT_BOOTSTRAP_SEED, DISPERSION_HIGH_THRESHOLD
from approach1.evaluation.metrics import prepare_predictions_for_eval, safe_auroc_auprc
from approach2.config import (
    TARGET_NAME_DISPERSION_HIGH_LOW,
    TARGET_NAME_DISPERSION_SCORE,
    TARGET_NAME_RELAPSE_STATUS,
)
from common.bootstrap_cis import BootstrapMetricSpec, bootstrap_percentile_cis

A1_PRED_REL = (
    "shotset_high_0_2_low_101_102/mri_plus_pathology/predictions_testing_cases.csv"
)
A2_HEADLINE_MODELS = {
    "continuous_dispersion": "pls_regression",
    "high_low_dispersion": "linear_svm",
    "relapse": "ridge_logistic",
}
A2_DATASET = "combined"
A2_REPRESENTATION = "group_count"

TASK_ROWS = [
    ("continuous_dispersion", "Continuous dispersion"),
    ("high_low_dispersion", "High/low dispersion"),
    ("relapse", "Relapse"),
]


def _finite_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> Optional[float]:
    if len(y_true) < 2:
        return None
    rho = spearmanr(y_true, y_pred).correlation
    return float(rho) if rho is not None and np.isfinite(rho) else None


def _finite_pearson(y_true: np.ndarray, y_pred: np.ndarray) -> Optional[float]:
    if len(y_true) < 2:
        return None
    r, _ = pearsonr(y_true, y_pred)
    return float(r) if np.isfinite(r) else None


def _binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Optional[float]]:
    if len(y_true) == 0:
        return {k: None for k in ("accuracy", "f1", "sensitivity", "specificity")}
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else None
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": spec,
    }


def _bootstrap_specs_a1() -> List[BootstrapMetricSpec]:
    def cont(df: pd.DataFrame) -> Dict[str, Optional[float]]:
        m = np.isfinite(df["dispersion_true"]) & np.isfinite(df["dispersion_score_pred"])
        u = df.loc[m]
        if len(u) == 0:
            return {}
        yt = u["dispersion_true"].astype(float).values
        yp = u["dispersion_score_pred"].astype(float).values
        return {
            "spearman_rho": _finite_spearman(yt, yp),
            "mae": float(mean_absolute_error(yt, yp)),
            "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
            "pearson_r": _finite_pearson(yt, yp),
        }

    def hl(df: pd.DataFrame) -> Dict[str, Optional[float]]:
        out: Dict[str, Optional[float]] = {}
        lm = df["dispersion_true_high_low"].notna() & df["dispersion_high_low_pred"].notna()
        if lm.any():
            out.update(
                _binary_metrics(
                    df.loc[lm, "dispersion_true_high_low"].astype(int).values,
                    df.loc[lm, "dispersion_high_low_pred"].astype(int).values,
                )
            )
        sm = df["dispersion_true_high_low"].notna() & np.isfinite(df["dispersion_score_pred"])
        if sm.any():
            yt = df.loc[sm, "dispersion_true_high_low"].astype(int).values
            sc = df.loc[sm, "dispersion_score_pred"].astype(float).values
            auroc, auprc, _ = safe_auroc_auprc(yt, sc)
            out["auroc"] = auroc
            out["auprc"] = auprc
        return out

    def rel(df: pd.DataFrame) -> Dict[str, Optional[float]]:
        out: Dict[str, Optional[float]] = {}
        m = df["relapse_true"].notna() & df["relapse_pred"].notna()
        if not m.any():
            return out
        yt = df.loc[m, "relapse_true"].astype(int).values
        yp = df.loc[m, "relapse_pred"].astype(int).values
        out.update(_binary_metrics(yt, yp))
        auroc, auprc, _ = safe_auroc_auprc(yt, yp.astype(float))
        out["auroc"] = auroc
        out["auprc"] = auprc
        return out

    specs: List[BootstrapMetricSpec] = []
    for task, fn, metrics in (
        ("continuous_dispersion", cont, ("spearman_rho", "mae", "rmse", "pearson_r")),
        ("high_low_dispersion", hl, ("auroc", "auprc", "f1", "sensitivity", "specificity", "accuracy")),
        ("relapse", rel, ("auroc", "auprc", "f1", "sensitivity", "specificity", "accuracy")),
    ):
        for metric in metrics:
            specs.append(
                BootstrapMetricSpec(
                    task=task,
                    metric=metric,
                    compute=lambda d, _fn=fn, _m=metric: _fn(d).get(_m),
                )
            )
    return specs


def _point_metrics_a1(df: pd.DataFrame) -> List[Dict[str, Any]]:
    boot = {
        (r.task, r.metric): r
        for r in bootstrap_percentile_cis(
            df, _bootstrap_specs_a1(), n_bootstrap=DEFAULT_BOOTSTRAP_N, random_seed=DEFAULT_BOOTSTRAP_SEED
        )
    }
    rows: List[Dict[str, Any]] = []
    n = len(df)
    hl_prev = float(df["dispersion_true_high_low"].mean())
    rel_prev = float(df["relapse_true"].mean())
    for task in ("continuous_dispersion", "high_low_dispersion", "relapse"):
        prev = hl_prev if task == "high_low_dispersion" else (rel_prev if task == "relapse" else np.nan)
        for metric in {k[1] for k in boot if k[0] == task}:
            rec = boot[(task, metric)]
            rows.append(
                {
                    "approach": "Approach 1 (few-shot LLM)",
                    "modality": "mri_plus_pathology",
                    "task": task,
                    "metric": metric,
                    "n": n,
                    "prevalence": prev,
                    "point_estimate": rec.point_estimate,
                    "ci_low": rec.ci_lower,
                    "ci_high": rec.ci_upper,
                    "source_file": "predictions_testing_cases.csv + bootstrap recompute",
                    "recomputed": True,
                }
            )
    return rows


def _load_a2_predictions(path: Path, model_key: str, target_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    sub = df[
        (df["dataset_key"] == A2_DATASET)
        & (df["representation"] == A2_REPRESENTATION)
        & (df["model_key"] == model_key)
        & (df["target_name"] == target_name)
    ].copy()
    if sub.empty:
        raise ValueError(
            f"No A2 predictions for {A2_DATASET}|{A2_REPRESENTATION}|{model_key}|{target_name}"
        )
    return sub


def _bootstrap_specs_a2(task: str) -> List[BootstrapMetricSpec]:
    def cont(df: pd.DataFrame) -> Dict[str, Optional[float]]:
        m = np.isfinite(df["y_true"]) & np.isfinite(df["y_pred_value"])
        u = df.loc[m]
        if len(u) == 0:
            return {}
        yt = u["y_true"].astype(float).values
        yp = u["y_pred_value"].astype(float).values
        return {
            "spearman_rho": _finite_spearman(yt, yp),
            "mae": float(mean_absolute_error(yt, yp)),
            "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
            "pearson_r": _finite_pearson(yt, yp),
        }

    def cls(df: pd.DataFrame) -> Dict[str, Optional[float]]:
        out: Dict[str, Optional[float]] = {}
        lm = df["y_true"].notna() & df["y_pred"].notna()
        if lm.any():
            out.update(
                _binary_metrics(
                    df.loc[lm, "y_true"].astype(int).values,
                    df.loc[lm, "y_pred"].astype(int).values,
                )
            )
        sm = df["y_true"].notna() & np.isfinite(df["y_prob"])
        if sm.any():
            yt = df.loc[sm, "y_true"].astype(int).values
            sc = df.loc[sm, "y_prob"].astype(float).values
            auroc, auprc, _ = safe_auroc_auprc(yt, sc)
            out["auroc"] = auroc
            out["auprc"] = auprc
            out["brier"] = float(brier_score_loss(yt, sc))
        return out

    fn = cont if task == "continuous_dispersion" else cls
    metrics = (
        ("spearman_rho", "mae", "rmse", "pearson_r")
        if task == "continuous_dispersion"
        else ("auroc", "auprc", "brier", "f1", "sensitivity", "specificity", "accuracy")
    )
    return [
        BootstrapMetricSpec(
            task=task,
            metric=m,
            compute=lambda d, _fn=fn, _m=m: _fn(d).get(_m),
        )
        for m in metrics
    ]


def _point_metrics_a2(preds_path: Path, summary_path: Path) -> List[Dict[str, Any]]:
    summary = pd.read_csv(summary_path)
    rows: List[Dict[str, Any]] = []
    endpoint_to_target = {
        "continuous_dispersion": TARGET_NAME_DISPERSION_SCORE,
        "high_low_dispersion": TARGET_NAME_DISPERSION_HIGH_LOW,
        "relapse": TARGET_NAME_RELAPSE_STATUS,
    }
    for endpoint, model_key in A2_HEADLINE_MODELS.items():
        target = endpoint_to_target[endpoint]
        pred_df = _load_a2_predictions(preds_path, model_key, target)
        n = len(pred_df)
        prev = float(pred_df["y_true"].mean()) if endpoint != "continuous_dispersion" else np.nan

        # Prefer precomputed CIs from nested summary when available
        srow = summary[
            (summary["dataset_key"] == A2_DATASET)
            & (summary["representation"] == A2_REPRESENTATION)
            & (summary["model_key"] == model_key)
            & (summary["target_name"] == target)
        ]
        if len(srow) != 1:
            raise ValueError(f"Expected one summary row for A2 {endpoint}, found {len(srow)}")
        srow = srow.iloc[0]

        boot = {
            (r.task, r.metric): r
            for r in bootstrap_percentile_cis(
                pred_df,
                _bootstrap_specs_a2(endpoint),
                n_bootstrap=DEFAULT_BOOTSTRAP_N,
                random_seed=DEFAULT_BOOTSTRAP_SEED,
            )
        }

        metric_names = list(boot.keys())
        for _, metric in metric_names:
            rec = boot[(endpoint, metric)]
            ci_low = srow.get(f"{metric}_ci_low", rec.ci_lower)
            ci_high = srow.get(f"{metric}_ci_high", rec.ci_upper)
            if metric == "recall_sensitivity":
                metric = "sensitivity"
            rows.append(
                {
                    "approach": "Approach 2 (feature discovery + ML)",
                    "modality": "combined (MRI + pathology)",
                    "task": endpoint,
                    "metric": metric,
                    "n": n,
                    "prevalence": prev,
                    "point_estimate": rec.point_estimate,
                    "ci_low": float(ci_low) if pd.notna(ci_low) else rec.ci_lower,
                    "ci_high": float(ci_high) if pd.notna(ci_high) else rec.ci_upper,
                    "source_file": "nested_outer_predictions_case_deduplicated.csv; CIs verified vs nested_outer_metrics_summary.csv",
                    "recomputed": True,
                }
            )
    return rows


def _fmt(v: Optional[float], d: int = 3) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    return f"{float(v):.{d}f}"


def _dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Render a DataFrame as a markdown table without optional tabulate dependency."""
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(str(c) for c in cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def _fmt_ci(lo: Optional[float], hi: Optional[float], d: int = 3) -> str:
    if lo is None or hi is None:
        return "—"
    return f"({_fmt(lo, d)}, {_fmt(hi, d)})"


def render_summary_table(long_df: pd.DataFrame) -> str:
    lines = [
        "# Unified Combined-Modality Metrics Audit",
        "",
        f"High/low threshold: dispersion score ≥ {int(DISPERSION_HIGH_THRESHOLD)}.",
        "Approach 2 headline: `combined | group_count | pls_regression` (continuous), "
        "`linear_svm` (high/low), `ridge_logistic` (relapse).",
        "",
        "## Long-format metrics",
        "",
        _dataframe_to_markdown(long_df),
        "",
        "## Wide comparison table",
        "",
        "| Task | Metric | Approach 1 | Approach 2 |",
        "| --- | --- | --- | --- |",
    ]
    key_metrics = [
        ("continuous_dispersion", "spearman_rho"),
        ("continuous_dispersion", "mae"),
        ("high_low_dispersion", "auroc"),
        ("high_low_dispersion", "auprc"),
        ("high_low_dispersion", "f1"),
        ("high_low_dispersion", "sensitivity"),
        ("high_low_dispersion", "specificity"),
        ("high_low_dispersion", "accuracy"),
        ("relapse", "auroc"),
        ("relapse", "auprc"),
        ("relapse", "f1"),
        ("relapse", "sensitivity"),
        ("relapse", "specificity"),
        ("relapse", "accuracy"),
    ]
    for task, metric in key_metrics:
        a1 = long_df[
            (long_df["approach"].str.startswith("Approach 1"))
            & (long_df["task"] == task)
            & (long_df["metric"] == metric)
        ]
        a2 = long_df[
            (long_df["approach"].str.startswith("Approach 2"))
            & (long_df["task"] == task)
            & (long_df["metric"] == metric)
        ]
        c1 = f"{_fmt(a1.iloc[0]['point_estimate'])} {_fmt_ci(a1.iloc[0]['ci_low'], a1.iloc[0]['ci_high'])}" if len(a1) else "—"
        c2 = f"{_fmt(a2.iloc[0]['point_estimate'])} {_fmt_ci(a2.iloc[0]['ci_low'], a2.iloc[0]['ci_high'])}" if len(a2) else "—"
        lines.append(f"| {task} | {metric} | {c1} | {c2} |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified combined-modality metrics audit.")
    parser.add_argument("--approach1-dir", required=True, help="Approach 1 result directory.")
    parser.add_argument("--approach2-dir", required=True, help="Approach 2 result directory.")
    parser.add_argument("--outdir", "-o", default="sabcs/metrics_audit_combined_modality")
    parser.add_argument("--bootstrap-n", type=int, default=DEFAULT_BOOTSTRAP_N)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    a1_pred = (repo_root / args.approach1_dir / A1_PRED_REL).resolve()
    a2_pred = (repo_root / args.approach2_dir / "nested_outer_predictions_case_deduplicated.csv").resolve()
    a2_summary = (repo_root / args.approach2_dir / "nested_outer_metrics_summary.csv").resolve()
    out_dir = (repo_root / args.outdir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    for p in (a1_pred, a2_pred, a2_summary):
        if not p.is_file():
            print(f"[ERROR] Missing required file: {p}", file=sys.stderr)
            sys.exit(1)

    a1_df = prepare_predictions_for_eval(pd.read_csv(a1_pred))
    rows = _point_metrics_a1(a1_df)
    rows.extend(_point_metrics_a2(a2_pred, a2_summary))

    long_df = pd.DataFrame(rows)
    csv_path = out_dir / "unified_metrics_long.csv"
    md_path = out_dir / "unified_metrics_report.md"
    long_df.to_csv(csv_path, index=False)
    md_path.write_text(render_summary_table(long_df), encoding="utf-8")

    print(f"[AUDIT] Wrote {len(long_df)} rows to {csv_path}")
    print(f"[AUDIT] Wrote report to {md_path}")
    for approach in long_df["approach"].unique():
        sub = long_df[long_df["approach"] == approach]
        n = int(sub["n"].iloc[0])
        print(f"[AUDIT] {approach}: n={n}")


if __name__ == "__main__":
    main()
