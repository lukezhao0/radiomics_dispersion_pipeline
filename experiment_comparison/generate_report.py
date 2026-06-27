"""Generate standalone HTML comparison report."""

from __future__ import annotations

import html
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from .plot_comparisons import PLOT_CAPTIONS


def _esc(text: object) -> str:
    return html.escape(str(text) if text is not None else "")


def _df_to_html_table(df: pd.DataFrame, max_rows: int = 200) -> str:
    if df is None or df.empty:
        return "<p><em>No data available.</em></p>"
    display = df.head(max_rows)
    return display.to_html(index=False, border=0, classes="data-table", escape=True)


def _build_strongest_summary(best_df: pd.DataFrame) -> str:
    if best_df is None or best_df.empty:
        return "<p><em>Insufficient data to auto-summarize strongest results.</em></p>"

    bullets: List[str] = []
    for metric in ("spearman_rho", "auroc", "accuracy"):
        sub = best_df[best_df["metric"] == metric]
        if sub.empty:
            continue
        row = sub.loc[sub["value"].idxmax()]
        bullets.append(
            f"<li>Best <strong>{_esc(metric)}</strong> ({_esc(row['target'])}): "
            f"<strong>{row['value']:.4f}</strong> — {_esc(row['run_label'])} "
            f"({_esc(row['approach'])}, {_esc(row['model'])}, reasoning={_esc(row['reasoning'])})</li>"
        )
    if not bullets:
        return "<p><em>No performance metrics available for summary.</em></p>"
    return "<ul>" + "".join(bullets) + "</ul>"


def _build_limitations(
    availability_df: pd.DataFrame,
    plot_results: Dict[str, Optional[Path]],
) -> str:
    items: List[str] = []
    if availability_df is not None and not availability_df.empty:
        for _, row in availability_df.iterrows():
            missing = [
                c.replace("has_", "")
                for c in availability_df.columns
                if c.startswith("has_") and not bool(row.get(c))
            ]
            if missing:
                items.append(
                    f"<li><strong>{_esc(row['run_id'])}</strong>: missing {', '.join(_esc(m) for m in missing[:8])}"
                    f"{'…' if len(missing) > 8 else ''}</li>"
                )
    for plot_name, path in plot_results.items():
        if path is None:
            items.append(f"<li>Plot <strong>{_esc(plot_name)}</strong> was skipped (insufficient data).</li>")
    if not items:
        return "<p>All requested plots and key metrics were available where expected.</p>"
    return "<ul>" + "".join(items) + "</ul>"


def generate_html_report(
    *,
    output_path: Path,
    run_metadata_df: pd.DataFrame,
    manual_metrics_df: pd.DataFrame,
    availability_df: pd.DataFrame,
    best_metrics_df: pd.DataFrame,
    normalized_df: pd.DataFrame,
    plot_results: Dict[str, Optional[Path]],
    config_path: Path,
) -> Path:
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    plots_rel_dir = "plots"

    plot_sections = []
    for plot_name, plot_path in plot_results.items():
        caption = PLOT_CAPTIONS.get(plot_name, "")
        if plot_path is not None and plot_path.is_file():
            rel = f"{plots_rel_dir}/{plot_path.name}"
            plot_sections.append(
                f"""
                <section class="plot-block">
                  <h3>{_esc(plot_name.replace('_', ' ').title())}</h3>
                  <p class="caption">{_esc(caption)}</p>
                  <img src="{_esc(rel)}" alt="{_esc(plot_name)}" />
                </section>
                """
            )
        else:
            plot_sections.append(
                f"""
                <section class="plot-block missing">
                  <h3>{_esc(plot_name.replace('_', ' ').title())}</h3>
                  <p class="caption">{_esc(caption)}</p>
                  <p><em>Plot not generated — required metrics unavailable.</em></p>
                </section>
                """
            )

    # Key comparison table: best metrics pivoted
    if best_metrics_df is not None and not best_metrics_df.empty:
        key_best = best_metrics_df[
            best_metrics_df["metric"].isin(["spearman_rho", "auroc", "accuracy", "f1"])
        ][["run_label", "approach", "target", "metric", "value", "best_rule", "modality", "shotset", "model_key"]]
    else:
        key_best = pd.DataFrame()

    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Experiment Comparison Report</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 2rem; color: #1a1a1a; max-width: 1200px; }}
    h1, h2, h3 {{ color: #111; }}
    .meta {{ color: #555; font-size: 0.95rem; }}
    table.data-table {{ border-collapse: collapse; width: 100%; font-size: 0.85rem; margin: 1rem 0; }}
    table.data-table th, table.data-table td {{ border: 1px solid #ddd; padding: 6px 8px; text-align: left; }}
    table.data-table th {{ background: #f5f5f5; }}
    .plot-block {{ margin: 2rem 0; padding: 1rem; border: 1px solid #e0e0e0; border-radius: 6px; }}
    .plot-block img {{ max-width: 100%; height: auto; }}
    .caption {{ color: #444; font-size: 0.9rem; }}
    .missing {{ background: #fff8f0; }}
    section {{ margin-bottom: 2rem; }}
  </style>
</head>
<body>
  <h1>Clinical Tumor Dispersion — Cross-Experiment Comparison</h1>
  <p class="meta">Generated: {_esc(generated_at)} | Config: {_esc(config_path)}</p>

  <section>
    <h2>Included Runs</h2>
    {_df_to_html_table(run_metadata_df)}
  </section>

  <section>
    <h2>Manual Legacy Metrics</h2>
    {_df_to_html_table(manual_metrics_df)}
  </section>

  <section>
    <h2>Data Availability Summary</h2>
    <p>Which key metrics were found per run (from discovered artifacts).</p>
    {_df_to_html_table(availability_df)}
  </section>

  <section>
    <h2>Key Comparison — Best Metrics per Run</h2>
    <p>Best = max for Spearman/AUROC/accuracy/F1; min for MAE/RMSE/Brier/cost. See <code>summary_best_metrics.csv</code> for full provenance.</p>
    {_df_to_html_table(key_best)}
  </section>

  <section>
    <h2>Strongest Observed Results</h2>
    {_build_strongest_summary(best_metrics_df)}
  </section>

  <section>
    <h2>Comparison Plots</h2>
    {''.join(plot_sections)}
  </section>

  <section>
    <h2>Limitations &amp; Missing Data</h2>
    {_build_limitations(availability_df, plot_results)}
    <p>Normalized long-form metrics: <code>normalized_metrics_long.csv</code> ({len(normalized_df)} rows).</p>
  </section>
</body>
</html>
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_doc, encoding="utf-8")
    return output_path
