"""HTML report rendering helpers."""

from __future__ import annotations

import html
import os
from typing import Dict, Optional, Sequence

import pandas as pd

# -----------------------------
# HTML report rendering
# -----------------------------

REPORT_HTML_STYLES = """
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  line-height: 1.5;
  color: #1f2933;
  background: #f7f9fc;
  margin: 0;
  padding: 0;
}
.page {
  max-width: 1200px;
  margin: 0 auto;
  padding: 2rem 1.5rem 3rem;
}
h1 { font-size: 1.9rem; margin: 0 0 0.75rem; }
h2 {
  font-size: 1.35rem;
  margin: 2rem 0 0.75rem;
  padding-bottom: 0.35rem;
  border-bottom: 2px solid #d9e2ec;
}
h3 { font-size: 1.05rem; margin: 0 0 0.35rem; }
.lead { color: #52606d; margin-bottom: 1.5rem; }
.section { margin-bottom: 1.75rem; }
.plot-card {
  background: #fff;
  border: 1px solid #d9e2ec;
  border-radius: 8px;
  padding: 1rem 1.1rem 1.2rem;
  margin: 1rem 0 1.25rem;
  box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
}
.plot-caption {
  color: #52606d;
  font-size: 0.95rem;
  margin: 0 0 0.75rem;
}
.plot-card img {
  display: block;
  max-width: 100%;
  height: auto;
  margin: 0 auto;
}
.table-wrap {
  overflow-x: auto;
  background: #fff;
  border: 1px solid #d9e2ec;
  border-radius: 8px;
  margin: 0.75rem 0 1rem;
}
table.data-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.88rem;
}
table.data-table th,
table.data-table td {
  border-bottom: 1px solid #e4e7eb;
  padding: 0.45rem 0.6rem;
  text-align: left;
  vertical-align: top;
}
table.data-table th {
  background: #f0f4f8;
  font-weight: 600;
  position: sticky;
  top: 0;
}
table.data-table tr:nth-child(even) td { background: #fbfdff; }
.note { color: #52606d; font-size: 0.95rem; margin: 0.5rem 0 0.75rem; }
"""


PLOT_EXPLANATIONS: Dict[str, str] = {
    "coefficient_sign_stability.png": (
        "Fraction of outer folds in which each feature's fitted coefficient kept the same sign. "
        "Values near 1.0 indicate stable directional effects across resamples."
    ),
    "weighted_mri_concepts.png": (
        "Top MRI ontology concepts ranked by pathology-calibrated weight. Higher weights reflect "
        "concepts that were both reliable against pathology language and informative for dispersion."
    ),
    "mri_pathology_reliability_heatmap.png": (
        "Average cross-modal concordance between MRI and pathology concepts within outer-training folds. "
        "Brighter cells indicate stronger MRI-to-pathology alignment used for calibration."
    ),
    "top_regression_predicted_vs_true.png": (
        "Held-out predictions from the best aggregate regression model versus true dispersion score. "
        "Points near the dashed identity line indicate better calibration of continuous predictions."
    ),
    "top_regression_residuals.png": (
        "Residuals (predicted minus true) for the top regression model. Random scatter around zero "
        "suggests no strong systematic bias across the prediction range."
    ),
    "top_dispersion_high_low_confusion_matrix.png": (
        "Confusion matrix for the best dispersion high/low classifier on deduplicated held-out cases."
    ),
    "top_dispersion_high_low_roc.png": (
        "ROC curve for the best dispersion high/low classifier. Higher curves indicate better "
        "discrimination across thresholds."
    ),
    "top_dispersion_high_low_pr.png": (
        "Precision-recall curve for dispersion high/low prediction. The dashed line is the no-skill "
        "baseline equal to event prevalence."
    ),
    "top_dispersion_high_low_calibration.png": (
        "Reliability diagram for predicted high-dispersion risk. Points near the diagonal indicate "
        "well-calibrated probabilities; labels show bin counts."
    ),
    "top_relapse_status_confusion_matrix.png": (
        "Confusion matrix for the best relapse classifier on deduplicated held-out cases."
    ),
    "top_relapse_status_roc.png": (
        "ROC curve for relapse prediction from the top aggregate classification model."
    ),
    "top_relapse_status_pr.png": (
        "Precision-recall curve for relapse prediction. Useful when relapse events are imbalanced."
    ),
    "top_relapse_status_calibration.png": (
        "Calibration plot for relapse risk predictions from the top model."
    ),
    "ranked_model_mae.png": (
        "Lowest mean absolute error (MAE) regression models across dataset, representation, and learner. "
        "Lower bars are better."
    ),
    "ranked_model_auprc.png": (
        "Highest area under the precision-recall curve (AUPRC) classification models. Higher bars are better."
    ),
    "ranked_model_auroc.png": (
        "Highest area under the ROC curve (AUROC) classification models. Higher bars are better."
    ),
    "bootstrap_ci_primary_metrics.png": (
        "Primary metric point estimates with 95% bootstrap confidence intervals after case-level "
        "deduplication."
    ),
    "nested_classification_comparison.png": (
        "AUROC for every evaluated classification setting in the nested resampling pipeline."
    ),
    "nested_relapse_auroc_comparison.png": (
        "Held-out AUROC for relapse-status models only, sorted for side-by-side comparison."
    ),
    "nested_relapse_auprc_comparison.png": (
        "Held-out AUPRC for relapse-status models. More informative than AUROC when relapse is rare."
    ),
    "nested_relapse_f1_comparison.png": (
        "Held-out F1 score for relapse-status models at the default classification threshold."
    ),
    "nested_relapse_brier_comparison.png": (
        "Held-out Brier score for relapse-status models. Lower scores indicate better probabilistic calibration."
    ),
    "nested_relapse_roc_curves_top_models.png": (
        "ROC curves for the top relapse models by aggregate AUROC/AUPRC on deduplicated predictions."
    ),
    "nested_relapse_pr_curves_top_models.png": (
        "Precision-recall curves for the top relapse models on deduplicated held-out predictions."
    ),
    "nested_regression_error_comparison.png": (
        "Held-out mean absolute error (MAE) for all regression settings. Lower bars are better."
    ),
    "nested_regression_correlation_comparison.png": (
        "Held-out Spearman rank correlation between predicted and true dispersion scores."
    ),
    "top_regression_spearman_rank.png": (
        "Rank of true versus rank of predicted dispersion for the top regression model. "
        "Points near the diagonal indicate preserved ordering across the cohort."
    ),
    "pathway_modality_comparison.png": (
        "Side-by-side comparison of primary metrics across dataset pathways "
        "(MRI-only, pathology-only, combined, calibrated MRI, teacher-student when present)."
    ),
    "feature_prevalence_by_modality.png": (
        "Mean training-set prevalence of stable phrase features by report modality."
    ),
    "top_regression_coefficients.png": (
        "Largest-magnitude regression coefficients from aggregate held-out models."
    ),
    "feature_count_by_modality.png": (
        "Count of candidate and stable features discovered per modality after rediscovery."
    ),
    "per_fold_regression_mae.png": (
        "Held-out MAE for the best regression setting within each outer split."
    ),
}


def plot_explanation_for(plot_path: str) -> str:
    base = os.path.basename(str(plot_path))
    return PLOT_EXPLANATIONS.get(
        base,
        "Diagnostic figure generated from nested held-out evaluation outputs.",
    )


def df_to_html_table(
    df: Optional[pd.DataFrame],
    max_rows: int = 25,
    float_digits: int = 3,
    table_class: str = "data-table",
) -> str:
    if df is None or len(df) == 0:
        return '<p class="note"><em>No rows available.</em></p>'
    tmp = df.head(max_rows).copy()
    for col in tmp.columns:
        if pd.api.types.is_float_dtype(tmp[col]):
            tmp[col] = tmp[col].map(lambda x: "" if pd.isna(x) else f"{x:.{float_digits}f}")
    table_html = tmp.to_html(index=False, border=0, classes=table_class, escape=True)
    return f'<div class="table-wrap">{table_html}</div>'


def html_paragraph(text: str) -> str:
    return f"<p>{html.escape(str(text))}</p>"


def html_section(title: str, parts: Sequence[str]) -> str:
    body = "\n".join(part for part in parts if part)
    return f'<section class="section"><h2>{html.escape(title)}</h2>\n{body}</section>'


def html_plot_block(plot_path: str, image_src: str, title: Optional[str] = None) -> str:
    plot_title = title or os.path.basename(str(plot_path))
    caption = plot_explanation_for(plot_path)
    return (
        f'<div class="plot-card">'
        f"<h3>{html.escape(plot_title)}</h3>"
        f'<p class="plot-caption">{html.escape(caption)}</p>'
        f'<img src="{html.escape(image_src)}" alt="{html.escape(plot_title)}">'
        f"</div>"
    )


def build_html_report(title: str, intro: str, sections: Sequence[str]) -> str:
    body = "\n".join(sections)
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n<head>\n'
        '<meta charset="utf-8">\n'
        f"<title>{html.escape(title)}</title>\n"
        f"<style>{REPORT_HTML_STYLES}</style>\n"
        "</head>\n<body>\n"
        '<div class="page">\n'
        f"<h1>{html.escape(title)}</h1>\n"
        f'<p class="lead">{html.escape(intro)}</p>\n'
        f"{body}\n"
        "</div>\n</body>\n</html>"
    )
