"""Evaluation orchestration: metrics, reports, and plots."""

from __future__ import annotations

import json
import os
from typing import Any, Dict

import pandas as pd

from ..io_atomic import json_safe
from .evidence import evidence_attribution_report
from .metrics import (
    compare_relapse_predictors,
    evaluate_dispersion,
    evaluate_dispersion_high_low,
    evaluate_needle_retrieval,
    evaluate_relapse_labels,
    missingness_summary,
    prepare_predictions_for_eval,
)
from .plots import (
    plot_dispersion_residuals,
    plot_dispersion_scatter,
    plot_label_confusion_matrix,
    plot_needle_retrieval_rates,
    plot_pred_dispersion_by_relapse,
    plot_relapse_predictor_comparison,
    plot_top_evidence_features,
)


def explanation_text() -> str:
    return (
        "What the evaluation measures\n"
        "----------------------------\n"
        "Dispersion score (regression)\n"
        "- MAE: average |predicted - true| dispersion score. Lower is better.\n"
        "- RMSE: sqrt(mean((predicted - true)^2)). Penalizes large errors more than MAE.\n"
        "- Spearman rho: rank correlation between true and predicted dispersion.\n"
        "\n"
        "Dispersion high/low (classification)\n"
        "- True high/low is derived from the true dispersion score using the cutoff >= 85.\n"
        "- Accuracy and F1 evaluate the predicted high/low label.\n"
        "- Confusion matrix rows are true labels [0,1] and columns are predicted labels [0,1].\n"
        "\n"
        "Relapse (classification; label-only)\n"
        "- Accuracy and F1 compare relapse_pred against relapse_true.\n"
        "\n"
        "Relapse predictor comparison\n"
        "- A) LLM relapse_pred is a binary label.\n"
        "- B) predicted dispersion score is evaluated as a continuous risk score for relapse.\n"
        "- C) true dispersion score is an upper-bound signal check.\n"
        "\n"
        "Needle-in-the-haystack retrieval\n"
        "- A single synthetic non-clinical token is inserted outside report text.\n"
        "- Exact retrieval checks prompt attention / context retention.\n"
        "\n"
        "Evidence attribution analysis\n"
        "- Evidence quotes are converted into case-level lexical features.\n"
        "- Smoothed odds ratios summarize which evidence terms are associated with each predicted class.\n"
    )


def evaluate_and_plot(pred_df: pd.DataFrame, out_dir: str, title_suffix: str) -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)
    df = prepare_predictions_for_eval(pred_df)

    out_txt = os.path.join(out_dir, "evaluation_metrics_from_csv.txt")
    out_explain = os.path.join(out_dir, "evaluation_explanation.txt")
    metrics_json = os.path.join(out_dir, "evaluation_metrics_summary.json")

    scatter_png = os.path.join(out_dir, "dispersion_true_vs_pred_scatter.png")
    resid_png = os.path.join(out_dir, "dispersion_residuals_hist.png")
    dhl_cm_png = os.path.join(out_dir, "dispersion_high_low_confusion_matrix.png")
    relapse_cm_png = os.path.join(out_dir, "relapse_confusion_matrix.png")
    pred_disp_by_relapse_png = os.path.join(out_dir, "predicted_dispersion_by_true_relapse.png")
    relapse_predictor_compare_png = os.path.join(out_dir, "relapse_predictor_comparison.png")
    needle_rates_png = os.path.join(out_dir, "needle_retrieval_rates.png")
    evidence_disp_pos_png = os.path.join(out_dir, "evidence_features_dispersion_high.png")
    evidence_disp_neg_png = os.path.join(out_dir, "evidence_features_dispersion_low.png")
    evidence_rel_pos_png = os.path.join(out_dir, "evidence_features_relapse_yes.png")
    evidence_rel_neg_png = os.path.join(out_dir, "evidence_features_relapse_no.png")
    evidence_disp_csv = os.path.join(out_dir, "evidence_attribution_dispersion_high_low.csv")
    evidence_rel_csv = os.path.join(out_dir, "evidence_attribution_relapse_yes_no.csv")

    miss = missingness_summary(df)
    disp_report, disp_used, disp_metrics = evaluate_dispersion(df)
    dhl_report, _, dhl_metrics = evaluate_dispersion_high_low(df)
    rel_report, _, rel_metrics = evaluate_relapse_labels(df)
    rel_comp_report, rel_comp_metrics = compare_relapse_predictors(df)
    needle_report, needle_metrics = evaluate_needle_retrieval(df)
    evidence_report, evidence_outputs = evidence_attribution_report(df)

    report = "\n".join([
        "=== SecureGPT Evaluation from Predictions CSV ===",
        f"Run: {title_suffix}",
        f"Total rows: {len(df)}",
        "",
        miss,
        "",
        disp_report,
        "",
        dhl_report,
        "",
        rel_report,
        "",
        rel_comp_report,
        "",
        needle_report,
        "",
        evidence_report,
        "",
    ])

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(report)
    with open(out_explain, "w", encoding="utf-8") as f:
        f.write(explanation_text())

    if len(disp_used) > 0:
        plot_dispersion_scatter(disp_used, scatter_png, title_suffix)
        plot_dispersion_residuals(disp_used, resid_png, title_suffix)
    if "confusion_matrix" in dhl_metrics:
        plot_label_confusion_matrix(
            dhl_metrics["confusion_matrix"],
            dhl_cm_png,
            f"Dispersion High/Low Confusion Matrix ({title_suffix})",
        )
    if "confusion_matrix" in rel_metrics:
        plot_label_confusion_matrix(
            rel_metrics["confusion_matrix"],
            relapse_cm_png,
            f"Relapse Confusion Matrix ({title_suffix})",
        )
    plot_pred_dispersion_by_relapse(df, pred_disp_by_relapse_png, title_suffix)
    plot_relapse_predictor_comparison(rel_comp_metrics, relapse_predictor_compare_png, title_suffix)
    plot_needle_retrieval_rates(df, needle_rates_png, title_suffix)

    disp_tbl = evidence_outputs.get("dispersion_high_low_pred", pd.DataFrame())
    rel_tbl = evidence_outputs.get("relapse_pred", pd.DataFrame())
    if len(disp_tbl):
        disp_tbl.to_csv(evidence_disp_csv, index=False)
        plot_top_evidence_features(disp_tbl, evidence_disp_pos_png, f"Evidence Features: Predicted High Dispersion ({title_suffix})", positive=True)
        plot_top_evidence_features(disp_tbl, evidence_disp_neg_png, f"Evidence Features: Predicted Low Dispersion ({title_suffix})", positive=False)
    if len(rel_tbl):
        rel_tbl.to_csv(evidence_rel_csv, index=False)
        plot_top_evidence_features(rel_tbl, evidence_rel_pos_png, f"Evidence Features: Predicted Relapse ({title_suffix})", positive=True)
        plot_top_evidence_features(rel_tbl, evidence_rel_neg_png, f"Evidence Features: Predicted Non-Relapse ({title_suffix})", positive=False)

    summary = {
        "n_rows": int(len(df)),
        "dispersion_regression": {k: json_safe(v) for k, v in disp_metrics.items()},
        "dispersion_high_low": {k: json_safe(v) for k, v in dhl_metrics.items()},
        "relapse_label": {k: json_safe(v) for k, v in rel_metrics.items()},
        "relapse_predictor_comparison": json_safe(rel_comp_metrics),
        "needle_retrieval": json_safe(needle_metrics),
    }
    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[EVAL] Wrote metrics + plots to: {out_dir}")
    return summary
