#!/usr/bin/env python3
"""Unified combined-modality performance evaluation: Approach 1 vs Approach 2.

Computes definitionally aligned metrics from saved pipeline outputs without
re-running LLM inference. Writes a long-format metrics CSV and a publication-
ready Markdown report with a side-by-side main table.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from approach1.config import DEFAULT_BOOTSTRAP_N, DEFAULT_BOOTSTRAP_SEED, DISPERSION_HIGH_THRESHOLD
from approach1.evaluation.bootstrap import compute_approach1_bootstrap_cis
from approach1.evaluation.metrics import prepare_predictions_for_eval, safe_auroc_auprc
from approach2.config import TARGET_NAME_DISPERSION_HIGH_LOW, TARGET_NAME_DISPERSION_SCORE, TARGET_NAME_RELAPSE_STATUS
from common.bootstrap_cis import BootstrapMetricSpec, bootstrap_percentile_cis

# Pre-specified Approach 2 headline models (combined | group_count).
A2_HEADLINE_MODELS: Dict[str, str] = {
    "continuous_dispersion": "pls_regression",
    "high_low_dispersion": "linear_svm",
    "relapse": "ridge_logistic",
}

A2_DATASET = "combined"
A2_REPRESENTATION = "group_count"

UNIFIED_COLUMNS = [
    "approach",
    "n",
    "endpoint",
    "metric",
    "point_estimate",
    "ci_low",
    "ci_high",
    "p_value",
    "notes",
]


def _fmt_metric(value: Optional[float], digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "—"
    return f"{float(value):.{digits}f}"


def _fmt_ci(low: Optional[float], high: Optional[float], digits: int = 3) -> str:
    if low is None or high is None or not (np.isfinite(low) and np.isfinite(high)):
        return "—"
    return f"({_fmt_metric(low, digits)}, {_fmt_metric(high, digits)})"


def _relapse_dispersion_mediated_metrics(df: pd.DataFrame) -> Dict[str, Optional[float]]:
    mask = df["relapse_true"].notna() & np.isfinite(df["dispersion_score_pred"])
    if not mask.any():
        return {"auroc": None, "auprc": None}
    y_true = df.loc[mask, "relapse_true"].astype(int).values
    scores = df.loc[mask, "dispersion_score_pred"].astype(float).values
    auroc, auprc, _ = safe_auroc_auprc(y_true, scores)
    return {"auroc": auroc, "auprc": auprc}


def _bootstrap_dispersion_mediated_relapse(
    df: pd.DataFrame,
    *,
    n_bootstrap: int,
    random_seed: int,
) -> List[Dict[str, Any]]:
    specs = [
        BootstrapMetricSpec(
            task="relapse_dispersion_mediated",
            metric="auroc",
            compute=lambda d: _relapse_dispersion_mediated_metrics(d).get("auroc"),
        ),
        BootstrapMetricSpec(
            task="relapse_dispersion_mediated",
            metric="auprc",
            compute=lambda d: _relapse_dispersion_mediated_metrics(d).get("auprc"),
        ),
    ]
    results = bootstrap_percentile_cis(df, specs, n_bootstrap=n_bootstrap, random_seed=random_seed)
    return [r.to_dict() for r in results]


def _a1_bootstrap_lookup(
    pred_df: pd.DataFrame,
    *,
    n_bootstrap: int,
    random_seed: int,
) -> Dict[Tuple[str, str], Dict[str, Any]]:
    base = {
        (row.task, row.metric): row.to_dict()
        for row in compute_approach1_bootstrap_cis(
            pred_df,
            n_bootstrap=n_bootstrap,
            random_seed=random_seed,
        )
    }
    for row in _bootstrap_dispersion_mediated_relapse(
        pred_df,
        n_bootstrap=n_bootstrap,
        random_seed=random_seed + 1,
    ):
        base[(row["task"], row["metric"])] = {
            "task": row["task"],
            "metric": row["metric"],
            "point_estimate": row["point_estimate"],
            "ci_lower": row["ci_lower"],
            "ci_upper": row["ci_upper"],
            "notes": row.get("notes", ""),
        }
    return base


def _row(
    approach: str,
    n: int,
    endpoint: str,
    metric: str,
    point: Optional[float],
    ci_low: Optional[float] = None,
    ci_high: Optional[float] = None,
    p_value: Optional[float] = None,
    notes: str = "",
) -> Dict[str, Any]:
    return {
        "approach": approach,
        "n": int(n),
        "endpoint": endpoint,
        "metric": metric,
        "point_estimate": point,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_value": p_value,
        "notes": notes,
    }


def load_approach1_rows(
    predictions_csv: str,
    *,
    n_bootstrap: int,
    random_seed: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    pred_df = prepare_predictions_for_eval(pd.read_csv(predictions_csv))
    n = len(pred_df)
    relapse_events = int(pred_df["relapse_true"].sum())
    high_low_events = int(pred_df["dispersion_true_high_low"].sum())
    boot = _a1_bootstrap_lookup(pred_df, n_bootstrap=n_bootstrap, random_seed=random_seed)

    def b(task: str, metric: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        rec = boot.get((task, metric), {})
        return (
            rec.get("point_estimate"),
            rec.get("ci_lower"),
            rec.get("ci_upper"),
        )

    rows: List[Dict[str, Any]] = []

    for metric in ("spearman_rho", "mae"):
        pt, lo, hi = b("continuous_dispersion", metric)
        rows.append(
            _row(
                "A1_few_shot_e2e",
                n,
                "continuous_dispersion",
                metric,
                pt,
                lo,
                hi,
                notes="Held-out test split; MRI+pathology combined modality",
            )
        )

    for metric in ("auroc", "sensitivity", "specificity"):
        pt, lo, hi = b("high_low_dispersion", metric)
        note = (
            f"True high = dispersion score >= {int(DISPERSION_HIGH_THRESHOLD)}; "
            "AUROC ranks dispersion_score_pred; sens/spec at LLM binary label (0.5-equivalent)"
        )
        rows.append(
            _row("A1_few_shot_e2e", n, "high_low_dispersion", metric, pt, lo, hi, notes=note)
        )

    for metric in ("auroc", "auprc", "sensitivity", "specificity"):
        pt, lo, hi = b("relapse_prediction", metric)
        rows.append(
            _row(
                "A1_few_shot_e2e",
                n,
                "relapse_direct",
                metric,
                pt,
                lo,
                hi,
                notes=(
                    f"Direct LLM relapse_pred binary label (0/1); events {relapse_events}/{n}; "
                    "AUROC ranks binary relapse_pred, not a calibrated probability"
                ),
            )
        )

    for metric in ("auroc", "auprc"):
        pt, lo, hi = b("relapse_dispersion_mediated", metric)
        rows.append(
            _row(
                "A1_few_shot_e2e",
                n,
                "relapse_dispersion_mediated",
                metric,
                pt,
                lo,
                hi,
                notes=(
                    "Supplementary: relapse risk ranked by predicted dispersion score "
                    "(dispersion_score_pred), not direct relapse labeling"
                ),
            )
        )

    meta = {
        "n": n,
        "relapse_events": relapse_events,
        "high_low_events": high_low_events,
        "validation": "single fixed held-out test split (n=82 MRI-complete with both reports)",
    }
    return rows, meta


def _a2_metric_value(summary_row: pd.Series, metric: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    point = summary_row.get(metric, np.nan)
    lo = summary_row.get(f"{metric}_ci_low", np.nan)
    hi = summary_row.get(f"{metric}_ci_high", np.nan)
    if pd.isna(point):
        return None, None, None
    return float(point), (float(lo) if pd.notna(lo) else None), (float(hi) if pd.notna(hi) else None)


def load_approach2_rows(
    metrics_summary_csv: str,
    permutation_tests_csv: str,
    predictions_dedup_csv: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    summary = pd.read_csv(metrics_summary_csv)
    perm = pd.read_csv(permutation_tests_csv)
    preds = pd.read_csv(predictions_dedup_csv)

    endpoint_to_target = {
        "continuous_dispersion": TARGET_NAME_DISPERSION_SCORE,
        "high_low_dispersion": TARGET_NAME_DISPERSION_HIGH_LOW,
        "relapse": TARGET_NAME_RELAPSE_STATUS,
    }

    relapse_sub = preds[
        (preds["dataset_key"] == A2_DATASET)
        & (preds["representation"] == A2_REPRESENTATION)
        & (preds["model_key"] == A2_HEADLINE_MODELS["relapse"])
        & (preds["target_name"] == TARGET_NAME_RELAPSE_STATUS)
    ]
    n = int(len(relapse_sub))
    relapse_events = int(relapse_sub["y_true"].sum()) if n else 0

    rows: List[Dict[str, Any]] = []

    for endpoint, model_key in A2_HEADLINE_MODELS.items():
        target_name = endpoint_to_target[endpoint]
        sub = summary[
            (summary["dataset_key"] == A2_DATASET)
            & (summary["representation"] == A2_REPRESENTATION)
            & (summary["model_key"] == model_key)
            & (summary["target_name"] == target_name)
        ]
        if len(sub) != 1:
            raise ValueError(
                f"Expected exactly one A2 summary row for {endpoint} "
                f"({A2_DATASET}|{A2_REPRESENTATION}|{model_key}|{target_name}), found {len(sub)}"
            )
        srow = sub.iloc[0]
        n_row = int(srow["n"])

        if endpoint == "continuous_dispersion":
            metrics = ("spearman_rho", "mae")
            note = (
                "Nested 5× repeated Monte Carlo outer CV; case-level deduplication by averaging "
                f"repeated outer-test predictions; headline model {model_key}"
            )
            p_auroc = p_auprc = None
        elif endpoint == "high_low_dispersion":
            metrics = ("auroc", "recall_sensitivity", "specificity")
            note = (
                f"True high = dispersion score >= {int(DISPERSION_HIGH_THRESHOLD)}; "
                f"AUROC from y_prob; sens/spec at 0.5 threshold (y_pred); headline model {model_key}"
            )
            p_auroc = p_auprc = None
        else:
            metrics = ("auroc", "auprc", "recall_sensitivity", "specificity")
            note = (
                f"Nested outer CV + deduplication; AUROC/AUPRC from predicted probabilities (y_prob); "
                f"sens/spec at 0.5; events {relapse_events}/{n_row}; headline model {model_key}"
            )
            perm_row = perm[
                (perm["dataset_key"] == A2_DATASET)
                & (perm["representation"] == A2_REPRESENTATION)
                & (perm["model_key"] == model_key)
            ]
            if len(perm_row) != 1:
                raise ValueError(f"Expected one permutation row for A2 relapse headline model, found {len(perm_row)}")
            p_auroc = float(perm_row.iloc[0]["auroc_empirical_p"])
            p_auprc = float(perm_row.iloc[0]["auprc_empirical_p"])

        for metric in metrics:
            out_metric = "sensitivity" if metric == "recall_sensitivity" else metric
            pt, lo, hi = _a2_metric_value(srow, metric)
            pval = None
            if endpoint == "relapse":
                if out_metric == "auroc":
                    pval = p_auroc
                elif out_metric == "auprc":
                    pval = p_auprc
            rows.append(
                _row(
                    "A2_feature_discovery_ml",
                    n_row,
                    endpoint if endpoint != "relapse" else "relapse_direct",
                    out_metric,
                    pt,
                    lo,
                    hi,
                    p_value=pval,
                    notes=note,
                )
            )

    meta = {
        "n": n,
        "relapse_events": relapse_events,
        "mri_complete_cohort": 86,
        "full_cohort": 104,
        "validation": (
            "nested 5× repeated Monte Carlo outer CV with case-level deduplication "
            f"(n={n} unique MRI-complete patients with ≥1 outer-held-out prediction; "
            "subset of 86 MRI-complete / 104 total)"
        ),
        "headline_models": A2_HEADLINE_MODELS,
    }
    return rows, meta


def build_unified_table(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=UNIFIED_COLUMNS)


def _side_by_side_cell(df: pd.DataFrame, approach: str, endpoint: str, metric: str) -> str:
    sub = df[(df["approach"] == approach) & (df["endpoint"] == endpoint) & (df["metric"] == metric)]
    if len(sub) == 0:
        return "—"
    row = sub.iloc[0]
    return f"{_fmt_metric(row['point_estimate'])} {_fmt_ci(row['ci_low'], row['ci_high'])}"


def _side_by_side_p(df: pd.DataFrame, approach: str, endpoint: str, metric: str) -> str:
    sub = df[(df["approach"] == approach) & (df["endpoint"] == endpoint) & (df["metric"] == metric)]
    if len(sub) == 0 or pd.isna(sub.iloc[0]["p_value"]):
        return ""
    return f" (p={_fmt_metric(sub.iloc[0]['p_value'])})"


def render_markdown_report(
    unified_df: pd.DataFrame,
    a1_meta: Dict[str, Any],
    a2_meta: Dict[str, Any],
) -> str:
    lines: List[str] = [
        "# Unified Combined-Modality Performance Evaluation (MRI + Pathology)",
        "",
        "Side-by-side comparison of **Approach 1** (few-shot end-to-end LLM) and "
        "**Approach 2** (LLM feature discovery → supervised ML) on the fused MRI+pathology "
        "tier only. Metrics are definitionally aligned but **not directly comparable** "
        "because validation schemes, sample sizes, and relapse score sources differ (see Methods).",
        "",
        "## Headline findings",
        "",
        f"- **Approach 1 (dispersion):** Strong continuous dispersion ranking "
        f"(Spearman ρ = {_fmt_metric(_side_by_side_cell(unified_df, 'A1_few_shot_e2e', 'continuous_dispersion', 'spearman_rho').split()[0])}, "
        f"n = {a1_meta['n']}) and high/low AUROC from predicted dispersion scores, with wide bootstrap "
        "intervals and moderate sensitivity at the binary operating point.",
        f"- **Approach 2 (relapse):** Strong relapse discrimination "
        f"(AUROC = {_fmt_metric(_side_by_side_cell(unified_df, 'A2_feature_discovery_ml', 'relapse_direct', 'auroc').split()[0])}, "
        f"AUPRC = {_fmt_metric(_side_by_side_cell(unified_df, 'A2_feature_discovery_ml', 'relapse_direct', 'auprc').split()[0])}, "
        f"n = {a2_meta['n']}; empirical permutation p < 0.001) using pre-specified "
        f"`{A2_DATASET} | {A2_REPRESENTATION} | {A2_HEADLINE_MODELS['relapse']}`.",
        f"- **Approach 1 direct relapse labeling** (supplementary): modest discrimination "
        f"(AUROC ≈ {_fmt_metric(_side_by_side_cell(unified_df, 'A1_few_shot_e2e', 'relapse_direct', 'auroc').split()[0])}, "
        f"events {a1_meta['relapse_events']}/{a1_meta['n']}) when ranking the LLM's binary `relapse_pred`.",
        "",
        "## Main comparison table",
        "",
        "| Endpoint | Metric | Approach 1 (n = "
        f"{a1_meta['n']}) | Approach 2 (n = {a2_meta['n']}) |",
        "| --- | --- | --- | --- |",
    ]

    primary_blocks = [
        ("continuous_dispersion", "Continuous dispersion", [
            ("spearman_rho", "Spearman ρ (true vs predicted score)"),
            ("mae", "MAE"),
        ]),
        ("high_low_dispersion", f"High/low dispersion (true high ≥ {int(DISPERSION_HIGH_THRESHOLD)})", [
            ("auroc", "AUROC (score-ranked)"),
            ("sensitivity", "Sensitivity @ operating threshold"),
            ("specificity", "Specificity @ operating threshold"),
        ]),
        ("relapse_direct", "Relapse (direct classification)", [
            ("auroc", "AUROC"),
            ("auprc", "AUPRC"),
            ("sensitivity", "Sensitivity @ 0.5"),
            ("specificity", "Specificity @ 0.5"),
        ]),
    ]

    for endpoint, endpoint_label, metrics in primary_blocks:
        for i, (metric, metric_label) in enumerate(metrics):
            ep_label = endpoint_label if i == 0 else ""
            a1_cell = _side_by_side_cell(unified_df, "A1_few_shot_e2e", endpoint, metric)
            a2_cell = _side_by_side_cell(unified_df, "A2_feature_discovery_ml", endpoint, metric)
            if endpoint == "relapse_direct" and metric in ("auroc", "auprc"):
                a2_cell += _side_by_side_p(unified_df, "A2_feature_discovery_ml", endpoint, metric)
            lines.append(f"| {ep_label} | {metric_label} | {a1_cell} | {a2_cell} |")

    lines.extend([
        "",
        "### Supplementary: Approach 1 relapse risk ranked by predicted dispersion score",
        "",
        "Not direct relapse labeling — ranks `dispersion_score_pred` against true relapse.",
        "",
        "| Metric | Approach 1 |",
        "| --- | --- |",
    ])
    for metric, label in (("auroc", "AUROC"), ("auprc", "AUPRC")):
        cell = _side_by_side_cell(unified_df, "A1_few_shot_e2e", "relapse_dispersion_mediated", metric)
        lines.append(f"| {label} | {cell} |")

    lines.extend([
        "",
        "## Event counts and cohort notes",
        "",
        f"- **Approach 1:** n = {a1_meta['n']} held-out test cases with both MRI and pathology reports "
        "(18 MRI-missing cases excluded from this tier). "
        f"Relapse events ≈ {a1_meta['relapse_events']}/{a1_meta['n']}; "
        f"high-dispersion prevalence ≈ {a1_meta['high_low_events']}/{a1_meta['n']}.",
        f"- **Approach 2:** n = {a2_meta['n']} unique MRI-complete patients with ≥1 outer-held-out "
        f"prediction after 5× repeated Monte Carlo outer CV and case-level deduplication "
        f"(full cohort = {a2_meta['full_cohort']}; MRI-complete = {a2_meta['mri_complete_cohort']}; "
        f"evaluated n is a subset). Relapse events ≈ {a2_meta['relapse_events']}/{a2_meta['n']}.",
        "",
        "## Methods note",
        "",
        "### Metric alignment",
        "- **Continuous dispersion:** Spearman ρ between true and predicted dispersion score; MAE.",
        f"- **High/low dispersion:** True high defined as score ≥ {int(DISPERSION_HIGH_THRESHOLD)}. "
        "AUROC uses a continuous score (`dispersion_score_pred` in A1; `y_prob` in A2). "
        "Sensitivity and specificity use each model's operating binary label at threshold 0.5 "
        "(A1: `dispersion_high_low_pred`; A2: `y_pred` from `y_prob ≥ 0.5`).",
        "- **Relapse:** AUROC and AUPRC plus sensitivity/specificity at 0.5. "
        "A1 direct relapse ranks the LLM binary `relapse_pred` (0/1), not a calibrated probability; "
        "A2 ranks predicted probabilities (`y_prob`).",
        "",
        "### Why results are not directly comparable",
        f"1. **Validation:** A1 = {a1_meta['validation']}. A2 = {a2_meta['validation']}.",
        f"2. **Sample size:** A1 n = {a1_meta['n']} vs A2 n = {a2_meta['n']}.",
        "3. **Relapse score source:** A1 direct relapse AUROC ranks binary `relapse_pred`; "
        "A2 ranks `y_prob` from supervised logistic regression.",
        "",
        "### Uncertainty quantification",
        f"- **Approach 1:** Post-hoc case-level bootstrap 95% CIs (B = {DEFAULT_BOOTSTRAP_N}, "
        f"seed = {DEFAULT_BOOTSTRAP_SEED}) from `predictions_testing_cases.csv`, matching A2 metric definitions.",
        "- **Approach 2:** Precomputed bootstrap 95% CIs from `nested_outer_metrics_summary.csv` "
        f"(B = 1000 case-level resamples on deduplicated predictions). Relapse permutation p-values "
        "from `relapse_permutation_tests.csv` (1000 label permutations).",
        "",
        "### Approach 2 headline models (pre-specified)",
        f"- Continuous dispersion: `{A2_DATASET} | {A2_REPRESENTATION} | {A2_HEADLINE_MODELS['continuous_dispersion']}`",
        f"- High/low dispersion: `{A2_DATASET} | {A2_REPRESENTATION} | {A2_HEADLINE_MODELS['high_low_dispersion']}`",
        f"- Relapse: `{A2_DATASET} | {A2_REPRESENTATION} | {A2_HEADLINE_MODELS['relapse']}`",
        "",
        "### Interpretation caveats",
        "- Bootstrap intervals are wide, especially for relapse and high/low sensitivity, reflecting "
        "limited events and held-out sample size.",
        "- A1 high/low sensitivity is moderate despite strong score-ranked AUROC, consistent with "
        "miscalibration of the LLM binary high/low label relative to the continuous score.",
        "- A2 relapse performance should be interpreted in the context of nested CV and the smaller "
        f"deduplicated evaluation subset (n = {a2_meta['n']}) rather than the full MRI-complete cohort.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified combined-modality evaluation (A1 vs A2).")
    parser.add_argument(
        "--a1-predictions",
        default="sabcs/securegpt_dispersion_approach1_pipeline_062726/"
        "shotset_high_0_2_low_101_102/mri_plus_pathology/predictions_testing_cases.csv",
        help="Approach 1 predictions_testing_cases.csv (MRI+pathology).",
    )
    parser.add_argument(
        "--a2-dir",
        default="sabcs/securegpt_dispersion_approach2_pipeline_062726",
        help="Approach 2 pipeline output directory.",
    )
    parser.add_argument(
        "--outdir",
        "-o",
        default="sabcs/unified_combined_modality_evaluation",
        help="Output directory for unified CSV and Markdown report.",
    )
    parser.add_argument(
        "--bootstrap-n",
        type=int,
        default=DEFAULT_BOOTSTRAP_N,
        help=f"A1 bootstrap replicates (default: {DEFAULT_BOOTSTRAP_N}).",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=DEFAULT_BOOTSTRAP_SEED,
        help=f"A1 bootstrap seed (default: {DEFAULT_BOOTSTRAP_SEED}).",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    a1_csv = (repo_root / args.a1_predictions).resolve()
    a2_dir = (repo_root / args.a2_dir).resolve()
    out_dir = (repo_root / args.outdir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not a1_csv.is_file():
        print(f"[ERROR] A1 predictions not found: {a1_csv}", file=sys.stderr)
        sys.exit(1)

    a2_metrics = a2_dir / "nested_outer_metrics_summary.csv"
    a2_perm = a2_dir / "relapse_permutation_tests.csv"
    a2_preds = a2_dir / "nested_outer_predictions_case_deduplicated.csv"
    for path in (a2_metrics, a2_perm, a2_preds):
        if not path.is_file():
            print(f"[ERROR] A2 artifact not found: {path}", file=sys.stderr)
            sys.exit(1)

    a1_rows, a1_meta = load_approach1_rows(
        str(a1_csv),
        n_bootstrap=args.bootstrap_n,
        random_seed=args.bootstrap_seed,
    )
    a2_rows, a2_meta = load_approach2_rows(str(a2_metrics), str(a2_perm), str(a2_preds))
    unified_df = build_unified_table(a1_rows + a2_rows)

    csv_path = out_dir / "unified_combined_modality_metrics.csv"
    md_path = out_dir / "unified_combined_modality_report.md"
    unified_df.to_csv(csv_path, index=False)
    md_path.write_text(
        render_markdown_report(unified_df, a1_meta, a2_meta),
        encoding="utf-8",
    )

    print(f"[UNIFIED] Wrote {len(unified_df)} metric rows to {csv_path}")
    print(f"[UNIFIED] Wrote report to {md_path}")
    print(f"[UNIFIED] A1 n={a1_meta['n']} relapse={a1_meta['relapse_events']}/{a1_meta['n']}")
    print(f"[UNIFIED] A2 n={a2_meta['n']} relapse={a2_meta['relapse_events']}/{a2_meta['n']}")


if __name__ == "__main__":
    main()
