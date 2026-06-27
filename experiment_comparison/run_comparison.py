#!/usr/bin/env python3
"""CLI entry point for cross-experiment comparison."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import pandas as pd

from .discover_results import discover_all_runs, discovered_files_to_dataframe
from .extract_metrics import extract_from_discovered, raw_metrics_to_dataframe
from .generate_report import generate_html_report
from .load_config import ensure_output_dir, load_config
from .normalize_results import (
    build_data_availability_summary,
    build_run_metadata_table,
    compute_best_metrics,
    normalize_raw_metrics,
)
from .plot_comparisons import generate_all_plots


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def run_comparison(config_path: str | Path, verbose: bool = False) -> Path:
    _setup_logging(verbose)
    logger = logging.getLogger(__name__)

    config = load_config(config_path)
    out_dir = ensure_output_dir(config)
    plots_dir = out_dir / "plots"

    logger.info("Loading config from %s", config.config_path)
    logger.info("Output directory: %s", out_dir)

    # 1. Discover files
    discovered = discover_all_runs(config.runs)
    discovered_df = discovered_files_to_dataframe(discovered)
    discovered_df.to_csv(out_dir / "discovered_files.csv", index=False)

    # 2. Extract metrics
    raw_metrics = extract_from_discovered(
        config.runs, discovered, config.manual_results, config.metric_aliases
    )
    raw_df = raw_metrics_to_dataframe(raw_metrics)
    raw_df.to_csv(out_dir / "raw_extracted_metrics.csv", index=False)

    # 3. Normalize
    normalized_df = normalize_raw_metrics(raw_df)
    normalized_df.to_csv(out_dir / "normalized_metrics_long.csv", index=False)

    # 4. Best metrics & availability
    best_df = compute_best_metrics(normalized_df)
    best_df.to_csv(out_dir / "summary_best_metrics.csv", index=False)

    run_meta_df = build_run_metadata_table(config.runs, config.manual_results)
    availability_df = build_data_availability_summary(run_meta_df, normalized_df, discovered_df)
    availability_df.to_csv(out_dir / "data_availability_summary.csv", index=False)

    manual_rows = []
    for m in config.manual_results:
        for k, v in m.metrics.items():
            manual_rows.append({"run_id": m.id, "label": m.label, "metric_key": k, "value": v})
    manual_df = pd.DataFrame(manual_rows)

    # 5. Plots
    plot_results = generate_all_plots(normalized_df, plots_dir, config.plots)

    # 6. HTML report
    html_path = out_dir / config.output.html_filename
    generate_html_report(
        output_path=html_path,
        run_metadata_df=run_meta_df,
        manual_metrics_df=manual_df,
        availability_df=availability_df,
        best_metrics_df=best_df,
        normalized_df=normalized_df,
        plot_results=plot_results,
        config_path=config.config_path,
    )

    logger.info("Comparison complete. Report: %s", html_path)
    return html_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare pipeline experiment runs.")
    parser.add_argument(
        "--config", "-c", required=True, help="Path to YAML/JSON comparison config."
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging.")
    args = parser.parse_args(argv)
    try:
        run_comparison(args.config, verbose=args.verbose)
        return 0
    except Exception as exc:
        logging.error("Comparison failed: %s", exc)
        if args.verbose:
            raise
        return 1


if __name__ == "__main__":
    sys.exit(main())
