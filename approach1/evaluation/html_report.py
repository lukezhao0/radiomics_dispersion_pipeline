"""Self-contained HTML styling and helpers for Approach 1 results reports."""

from __future__ import annotations

import html
import os
from typing import Optional, Sequence

import pandas as pd

REPORT_HTML_STYLES = """
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  line-height: 1.55;
  color: #1f2933;
  background: #f7f9fc;
  margin: 0;
  padding: 0;
}
.page {
  max-width: 1240px;
  margin: 0 auto;
  padding: 2rem 1.5rem 3rem;
}
h1 { font-size: 2rem; margin: 0 0 0.5rem; color: #102a43; }
h2 {
  font-size: 1.35rem;
  margin: 2.25rem 0 0.75rem;
  padding-bottom: 0.35rem;
  border-bottom: 2px solid #d9e2ec;
  color: #243b53;
}
h3 { font-size: 1.05rem; margin: 0 0 0.35rem; color: #334e68; }
h4 { font-size: 0.98rem; margin: 1rem 0 0.35rem; color: #486581; }
.lead { color: #52606d; margin-bottom: 1.5rem; max-width: 920px; }
.section { margin-bottom: 1.75rem; }
.config-card {
  background: #fff;
  border: 1px solid #d9e2ec;
  border-radius: 10px;
  padding: 1.25rem 1.35rem 1.5rem;
  margin: 1.5rem 0 2rem;
  box-shadow: 0 2px 6px rgba(16, 24, 40, 0.05);
}
.metric-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 0.75rem;
  margin: 0.75rem 0 1rem;
}
.metric-card {
  background: #f0f4f8;
  border: 1px solid #d9e2ec;
  border-radius: 8px;
  padding: 0.75rem 0.85rem;
}
.metric-label {
  font-size: 0.78rem;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  color: #627d98;
  margin-bottom: 0.2rem;
}
.metric-value {
  font-size: 1.15rem;
  font-weight: 600;
  color: #102a43;
}
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
  font-size: 0.86rem;
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
}
table.data-table tr:nth-child(even) td { background: #fbfdff; }
.note { color: #52606d; font-size: 0.95rem; margin: 0.5rem 0 0.75rem; }
.preblock {
  background: #f0f4f8;
  border: 1px solid #d9e2ec;
  border-radius: 8px;
  padding: 0.85rem 1rem;
  overflow-x: auto;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 0.82rem;
  white-space: pre-wrap;
}
.toc {
  background: #fff;
  border: 1px solid #d9e2ec;
  border-radius: 8px;
  padding: 1rem 1.2rem;
  margin: 1rem 0 1.5rem;
}
.toc ul { margin: 0.35rem 0 0; padding-left: 1.2rem; }
.toc a { color: #1473e6; text-decoration: none; }
.toc a:hover { text-decoration: underline; }
.tag {
  display: inline-block;
  background: #e3f8ff;
  color: #0b69a3;
  border-radius: 999px;
  padding: 0.15rem 0.55rem;
  font-size: 0.78rem;
  margin-right: 0.35rem;
}
"""

PLOT_EXPLANATIONS: dict[str, str] = {
    "dispersion_true_vs_pred_scatter.png": (
        "Each point is one held-out case. The x-axis is the true dispersion score and the y-axis is "
        "the model prediction. Points near the dashed identity line indicate better calibration."
    ),
    "dispersion_residuals_hist.png": (
        "Distribution of prediction errors (predicted minus true dispersion). A centered distribution "
        "near zero suggests less systematic bias."
    ),
    "dispersion_high_low_confusion_matrix.png": (
        "Confusion matrix for high vs low dispersion classification. True high/low is defined using "
        "the cutoff dispersion score >= 85. Rows are true labels; columns are predicted labels."
    ),
    "relapse_confusion_matrix.png": (
        "Confusion matrix comparing predicted relapse label to true relapse status (0=non-relapsing, "
        "1=relapsing)."
    ),
    "predicted_dispersion_by_true_relapse.png": (
        "Distribution of predicted dispersion scores split by true relapse status. Helps visualize "
        "whether predicted dispersion separates relapsing from non-relapsing cases."
    ),
    "relapse_predictor_comparison.png": (
        "Compares relapse prediction signals: the LLM relapse label, predicted dispersion score as a "
        "continuous risk score, and true dispersion as an upper-bound reference."
    ),
    "needle_retrieval_rates.png": (
        "Needle-in-the-haystack check: the model must echo a synthetic validation token placed at the "
        "end of the prompt. High exact-match rates indicate the model retained non-clinical metadata."
    ),
    "evidence_features_dispersion_high.png": (
        "Lexical features derived from evidence quotes most associated with predicted high dispersion."
    ),
    "evidence_features_dispersion_low.png": (
        "Lexical features derived from evidence quotes most associated with predicted low dispersion."
    ),
    "evidence_features_relapse_yes.png": (
        "Evidence-quote features associated with predicted relapse (label=1)."
    ),
    "evidence_features_relapse_no.png": (
        "Evidence-quote features associated with predicted non-relapse (label=0)."
    ),
}


def plot_explanation_for(plot_path: str) -> str:
    return PLOT_EXPLANATIONS.get(
        os.path.basename(plot_path),
        "Diagnostic figure generated from held-out Approach 1 predictions.",
    )


def df_to_html_table(
    df: Optional[pd.DataFrame],
    *,
    max_rows: int = 25,
    float_digits: int = 3,
) -> str:
    if df is None or len(df) == 0:
        return '<p class="note"><em>No rows available.</em></p>'
    tmp = df.head(max_rows).copy()
    for col in tmp.columns:
        if pd.api.types.is_float_dtype(tmp[col]):
            tmp[col] = tmp[col].map(lambda x: "" if pd.isna(x) else f"{x:.{float_digits}f}")
    table_html = tmp.to_html(index=False, border=0, classes="data-table", escape=True)
    suffix = ""
    if len(df) > max_rows:
        suffix = f'<p class="note">Showing first {max_rows} of {len(df)} rows.</p>'
    return f'<div class="table-wrap">{table_html}</div>{suffix}'


def html_paragraph(text: str) -> str:
    return f"<p>{html.escape(str(text))}</p>"


def html_section(title: str, parts: Sequence[str], *, section_id: Optional[str] = None) -> str:
    body = "\n".join(part for part in parts if part)
    anchor = f' id="{html.escape(section_id)}"' if section_id else ""
    return f'<section class="section"{anchor}><h2>{html.escape(title)}</h2>\n{body}</section>'


def html_plot_block(plot_path: str, image_src: str, title: Optional[str] = None) -> str:
    plot_title = title or os.path.basename(plot_path)
    caption = plot_explanation_for(plot_path)
    return (
        f'<div class="plot-card">'
        f"<h3>{html.escape(plot_title)}</h3>"
        f'<p class="plot-caption">{html.escape(caption)}</p>'
        f'<img src="{html.escape(image_src)}" alt="{html.escape(plot_title)}">'
        f"</div>"
    )


def html_metric_cards(metrics: dict[str, tuple[str, str]]) -> str:
    cards = []
    for label, (value, hint) in metrics.items():
        cards.append(
            f'<div class="metric-card">'
            f'<div class="metric-label">{html.escape(label)}</div>'
            f'<div class="metric-value">{html.escape(value)}</div>'
            f'<div class="note">{html.escape(hint)}</div>'
            f"</div>"
        )
    return f'<div class="metric-grid">{"".join(cards)}</div>'


def html_preblock(text: str) -> str:
    return f'<pre class="preblock">{html.escape(text)}</pre>'


def build_html_report(title: str, intro: str, sections: Sequence[str]) -> str:
    body = "\n".join(sections)
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n<head>\n'
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{html.escape(title)}</title>\n"
        f"<style>{REPORT_HTML_STYLES}</style>\n"
        "</head>\n<body>\n"
        '<div class="page">\n'
        f"<h1>{html.escape(title)}</h1>\n"
        f'<p class="lead">{html.escape(intro)}</p>\n'
        f"{body}\n"
        "</div>\n</body>\n</html>"
    )
