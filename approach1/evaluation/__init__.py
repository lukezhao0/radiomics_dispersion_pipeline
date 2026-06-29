"""Evaluation public API."""

from .evidence import build_evidence_feature_table, evidence_attribution_report, extract_ngram_features
from .metrics import (
    compare_relapse_predictors,
    evaluate_dispersion,
    evaluate_dispersion_high_low,
    evaluate_needle_retrieval,
    evaluate_relapse_labels,
    missingness_summary,
    prepare_predictions_for_eval,
)
from .plots import plot_dispersion_scatter
from .runner import evaluate_and_plot, explanation_text
from .bootstrap import compute_and_save_bootstrap_cis, compute_approach1_bootstrap_cis
from .results_report import REPORT_FILENAME, build_approach1_results_html, discover_config_dirs

__all__ = [
    "evaluate_and_plot",
    "explanation_text",
    "compute_and_save_bootstrap_cis",
    "compute_approach1_bootstrap_cis",
    "build_approach1_results_html",
    "discover_config_dirs",
    "REPORT_FILENAME",
    "prepare_predictions_for_eval",
    "evaluate_dispersion",
    "evaluate_dispersion_high_low",
    "evaluate_relapse_labels",
    "compare_relapse_predictors",
    "evaluate_needle_retrieval",
    "missingness_summary",
    "evidence_attribution_report",
    "build_evidence_feature_table",
    "extract_ngram_features",
    "plot_dispersion_scatter",
]
