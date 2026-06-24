"""Build a consolidated HTML results review page from Approach 1 output artifacts."""

from __future__ import annotations

import html
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .. import config
from ..text_utils import modality_display_name
from .html_report import (
    build_html_report,
    df_to_html_table,
    html_metric_cards,
    html_paragraph,
    html_plot_block,
    html_preblock,
    html_section,
)
from .runner import explanation_text

REPORT_FILENAME = "approach1_results_report.html"

PLOT_FILES_IN_ORDER = [
    "dispersion_true_vs_pred_scatter.png",
    "dispersion_residuals_hist.png",
    "dispersion_high_low_confusion_matrix.png",
    "relapse_confusion_matrix.png",
    "predicted_dispersion_by_true_relapse.png",
    "relapse_predictor_comparison.png",
    "needle_retrieval_rates.png",
    "evidence_features_dispersion_high.png",
    "evidence_features_dispersion_low.png",
    "evidence_features_relapse_yes.png",
    "evidence_features_relapse_no.png",
]


def _read_json(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _read_text(path: str) -> str:
    if not os.path.isfile(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def _fmt_float(value: Any, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def discover_config_dirs(root_out_dir: str) -> List[Tuple[str, str, str]]:
    """Return sorted (shotset_name, modality, absolute_dir) for completed config folders."""
    found: List[Tuple[str, str, str]] = []
    if not os.path.isdir(root_out_dir):
        return found
    for shotset in sorted(os.listdir(root_out_dir)):
        shot_path = os.path.join(root_out_dir, shotset)
        if not os.path.isdir(shot_path) or shotset.startswith("."):
            continue
        for modality in sorted(os.listdir(shot_path)):
            mod_path = os.path.join(shot_path, modality)
            if not os.path.isdir(mod_path):
                continue
            if os.path.isfile(os.path.join(mod_path, "run_config.json")):
                found.append((shotset, modality, mod_path))
    return found


def _relative_path(from_dir: str, target_path: str) -> str:
    return os.path.relpath(target_path, from_dir).replace(os.sep, "/")


def _metrics_cards_from_summary(metrics: Dict[str, Any]) -> Dict[str, tuple[str, str]]:
    disp = metrics.get("dispersion_regression", {}) or {}
    dhl = metrics.get("dispersion_high_low", {}) or {}
    rel = metrics.get("relapse_label", {}) or {}
    needle = metrics.get("needle_retrieval", {}) or {}
    return {
        "Cases evaluated": (str(metrics.get("n_rows", "—")), "Held-out predictions with saved outputs"),
        "Dispersion MAE": (_fmt_float(disp.get("mae")), "Lower is better"),
        "Dispersion RMSE": (_fmt_float(disp.get("rmse")), "Penalizes large errors"),
        "Dispersion Spearman ρ": (_fmt_float(disp.get("spearman_rho")), "Rank correlation; higher is better"),
        "High/low accuracy": (_fmt_float(dhl.get("accuracy")), f"Cutoff >= {int(config.DISPERSION_HIGH_THRESHOLD)}"),
        "High/low F1": (_fmt_float(dhl.get("f1")), "Balance of precision and recall"),
        "Relapse accuracy": (_fmt_float(rel.get("accuracy")), "Binary relapse label match"),
        "Relapse F1": (_fmt_float(rel.get("f1")), "Important when relapse is imbalanced"),
        "Needle retrieval": (_fmt_float(needle.get("single_token_rate")), "Exact validation-token echo rate"),
    }


def _predictions_preview_table(pred_csv: str) -> pd.DataFrame:
    if not os.path.isfile(pred_csv):
        return pd.DataFrame()
    df = pd.read_csv(pred_csv)
    cols = [
        c
        for c in [
            "case_id",
            "row_index",
            "index_side",
            "dispersion_true",
            "dispersion_score_pred",
            "dispersion_high_low_pred",
            "relapse_true",
            "relapse_pred",
            "retrieval_token_exact_match",
            "reasoning_summary",
        ]
        if c in df.columns
    ]
    if not cols:
        return df
    out = df[cols].copy()
    if "reasoning_summary" in out.columns:
        out["reasoning_summary"] = out["reasoning_summary"].astype(str).map(
            lambda s: (s[:180] + "…") if len(s) > 180 else s
        )
    return out


def _relapse_comparison_table(metrics: Dict[str, Any]) -> pd.DataFrame:
    comp = metrics.get("relapse_predictor_comparison", {}) or {}
    rows = []
    for name, payload in comp.items():
        if not isinstance(payload, dict):
            continue
        rows.append({
            "predictor": name,
            "accuracy": payload.get("accuracy"),
            "f1": payload.get("f1"),
            "auroc": payload.get("auroc"),
            "auprc": payload.get("auprc"),
            "best_f1": payload.get("best_f1"),
            "best_threshold": payload.get("best_threshold"),
            "note": payload.get("note"),
        })
    return pd.DataFrame(rows)


def _cost_summary_table(cost_json: Dict[str, Any]) -> pd.DataFrame:
    cumulative = cost_json.get("cumulative", cost_json) if isinstance(cost_json, dict) else {}
    if not cumulative:
        return pd.DataFrame()
    return pd.DataFrame([{
        "api_calls": cumulative.get("calls"),
        "prompt_tokens": cumulative.get("prompt_tokens"),
        "cached_tokens": cumulative.get("cached_tokens"),
        "uncached_prompt_tokens": cumulative.get("uncached_prompt_tokens"),
        "completion_tokens": cumulative.get("completion_tokens"),
        "reasoning_tokens": cumulative.get("reasoning_tokens"),
        "total_tokens": cumulative.get("total_tokens"),
        "estimated_cost_usd": cumulative.get("estimated_cost_usd"),
        "estimated_cache_savings_usd": cumulative.get("estimated_cache_savings_usd"),
    }])


def _build_config_section(
    root_out_dir: str,
    shotset_name: str,
    modality: str,
    config_dir: str,
) -> str:
    section_id = f"{shotset_name}__{modality}".replace(" ", "_")
    run_cfg = _read_json(os.path.join(config_dir, "run_config.json"))
    metrics = _read_json(os.path.join(config_dir, "evaluation_metrics_summary.json"))
    cost = _read_json(os.path.join(config_dir, "token_cost_report.json"))
    metrics_txt = _read_text(os.path.join(config_dir, "evaluation_metrics_from_csv.txt"))

    tags = [
        f'<span class="tag">{html.escape(shotset_name)}</span>',
        f'<span class="tag">{html.escape(modality_display_name(modality))}</span>',
    ]
    parts: List[str] = [
        f'<div class="config-card" id="{html.escape(section_id)}">',
        f"<h3>{''.join(tags)}</h3>",
        html_paragraph(
            "One complete few-shot evaluation run: a fixed shot set of labeled exemplars, a modality tier "
            "controlling which report text is shown, and held-out test cases excluded from training."
        ),
        "<h4>Run configuration</h4>",
        df_to_html_table(pd.DataFrame([{
            "high_rows": run_cfg.get("high_rows"),
            "low_rows": run_cfg.get("low_rows"),
            "training_rows": run_cfg.get("training_rows"),
            "n_test_cases": run_cfg.get("n_test_cases"),
            "n_skipped_missing_mri": run_cfg.get("n_skipped_missing_mri"),
        }]), max_rows=1),
    ]

    skipped_csv = os.path.join(config_dir, "skipped_cases_missing_mri.csv")
    if os.path.isfile(skipped_csv):
        skipped_df = pd.read_csv(skipped_csv)
        parts.extend([
            "<h4>Skipped cases (missing MRI)</h4>",
            html_paragraph(
                "These held-out rows were not sent to the model because this modality tier requires "
                "preoperative MRI text and the report was missing or a placeholder."
            ),
            df_to_html_table(skipped_df, max_rows=30),
        ])

    parts.extend([
        "<h4>Headline metrics</h4>",
        html_metric_cards(_metrics_cards_from_summary(metrics)),
        "<h4>Relapse predictor comparison</h4>",
        html_paragraph(
            "Compares three relapse signals on the same held-out cases: the LLM's binary relapse label, "
            "the predicted dispersion score as a continuous risk score, and the true dispersion score "
            "as a reference upper bound."
        ),
        df_to_html_table(_relapse_comparison_table(metrics), max_rows=10),
    ])

    cost_tbl = _cost_summary_table(cost)
    if len(cost_tbl):
        parts.extend([
            "<h4>Token and cost usage</h4>",
            html_paragraph(
                "Post-run cumulative token counts and estimated USD cost from API usage fields. "
                "Cached prompt tokens reflect prefix reuse across similar few-shot prompts."
            ),
            df_to_html_table(cost_tbl, max_rows=1),
        ])

    plot_blocks: List[str] = []
    for plot_name in PLOT_FILES_IN_ORDER:
        plot_path = os.path.join(config_dir, plot_name)
        if os.path.isfile(plot_path):
            plot_blocks.append(
                html_plot_block(
                    plot_path,
                    _relative_path(root_out_dir, plot_path),
                    title=plot_name.replace("_", " ").replace(".png", ""),
                )
            )
    if plot_blocks:
        parts.append("<h4>Diagnostic plots</h4>")
        parts.extend(plot_blocks)

    pred_csv = os.path.join(config_dir, "predictions_testing_cases.csv")
    pred_preview = _predictions_preview_table(pred_csv)
    if len(pred_preview):
        parts.extend([
            "<h4>Per-case predictions (preview)</h4>",
            html_paragraph(
                "Each row is one held-out patient case. Compare true vs predicted dispersion and relapse "
                "labels, and scan truncated reasoning summaries. Open the CSV for full evidence quotes."
            ),
            df_to_html_table(pred_preview, max_rows=40),
        ])

    for ev_name, caption in [
        (
            "evidence_attribution_dispersion_high_low.csv",
            "Smoothed odds ratios for quote-derived terms associated with predicted high vs low dispersion.",
        ),
        (
            "evidence_attribution_relapse_yes_no.csv",
            "Quote-derived lexical features associated with predicted relapse vs non-relapse.",
        ),
    ]:
        ev_path = os.path.join(config_dir, ev_name)
        if os.path.isfile(ev_path):
            ev_df = pd.read_csv(ev_path)
            parts.extend([f"<h4>{ev_name}</h4>", html_paragraph(caption), df_to_html_table(ev_df, max_rows=20)])

    if metrics_txt:
        parts.extend([
            "<h4>Full text metrics report</h4>",
            html_paragraph("Verbatim evaluation output written during the pipeline run."),
            html_preblock(metrics_txt),
        ])

    parts.append("</div>")
    return "\n".join(parts)


def build_approach1_results_html(root_out_dir: str, *, out_path: Optional[str] = None) -> str:
    """Scan an Approach 1 output directory and write a consolidated HTML review report."""
    root_out_dir = os.path.abspath(root_out_dir)
    out_path = out_path or os.path.join(root_out_dir, REPORT_FILENAME)
    config_dirs = discover_config_dirs(root_out_dir)

    intro = (
        "This report summarizes Approach 1 few-shot LLM predictions of tumor spatial dispersiveness "
        "and relapse risk from post-neoadjuvant clinical report text. Each section corresponds to one "
        "shot-set and modality combination. Metrics are computed on held-out cases that were not used "
        "as few-shot training exemplars."
    )

    toc_items = ['<li><a href="#overview">Overview</a></li>', '<li><a href="#aggregate">Aggregate summary</a></li>']
    for shotset, modality, _ in config_dirs:
        anchor = f"{shotset}__{modality}".replace(" ", "_")
        label = f"{shotset} / {modality_display_name(modality)}"
        toc_items.append(f'<li><a href="#{html.escape(anchor)}">{html.escape(label)}</a></li>')
    toc_items.append('<li><a href="#glossary">Metric glossary</a></li>')

    sections: List[str] = [
        f'<nav class="toc"><strong>Contents</strong><ul>{"".join(toc_items)}</ul></nav>',
        html_section(
            "What is Approach 1?",
            [
                html_paragraph(
                    "Approach 1 frames each patient as a text-to-outcome prediction task. The model receives "
                    "a descriptor guide, a small set of labeled exemplar cases (2 high-dispersion and "
                    "2 low-dispersion rows), and one held-out case per API call. It must return strict JSON "
                    "with a dispersion score in [0, 450], high/low dispersion (cutoff 85), relapse label, "
                    "verbatim evidence quotes, and a structured rationale."
                ),
                html_paragraph(
                    f"High/low dispersion is defined as score >= {int(config.DISPERSION_HIGH_THRESHOLD)}. "
                    "Three modality tiers are evaluated: MRI only, pathology only, and MRI + pathology combined. "
                    "MRI-missing cases are skipped for MRI-required tiers but remain in pathology-only runs."
                ),
            ],
            section_id="overview",
        ),
    ]

    agg_csv = os.path.join(root_out_dir, "all_tiers_metrics_summary.csv")
    if os.path.isfile(agg_csv):
        agg_df = pd.read_csv(agg_csv)
        sections.append(
            html_section(
                "Aggregate metrics across all runs",
                [
                    html_paragraph(
                        "One row per shot-set and modality tier. Use this table to compare performance "
                        "across input types before drilling into individual sections below."
                    ),
                    df_to_html_table(agg_df, max_rows=50),
                ],
                section_id="aggregate",
            )
        )

    if not config_dirs:
        sections.append(
            html_section(
                "No completed runs found",
                [
                    html_paragraph(
                        f"No subfolders with run_config.json were found under {root_out_dir}. "
                        "Run the pipeline first, then regenerate this report."
                    ),
                ],
            )
        )
    else:
        config_parts = [
            html_paragraph(
                f"Found {len(config_dirs)} completed configuration folder(s). "
                f"Report generated {datetime.now().strftime('%Y-%m-%d %H:%M')}."
            ),
        ]
        for shotset, modality, config_dir in config_dirs:
            config_parts.append(
                _build_config_section(root_out_dir, shotset, modality, config_dir)
            )
        sections.append(
            html_section("Per-configuration results", config_parts, section_id="configs")
        )

    sections.append(
        html_section(
            "Metric glossary",
            [html_preblock(explanation_text())],
            section_id="glossary",
        )
    )

    html_doc = build_html_report(
        title="Approach 1 Results Review",
        intro=intro,
        sections=sections,
    )
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_doc)
    print(f"[REPORT] Wrote HTML results review: {out_path}")
    return out_path
