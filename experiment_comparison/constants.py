"""Canonical metric names, aliases, and best-metric selection rules."""

from __future__ import annotations

from typing import Dict, Literal, Set

# Metrics where higher values indicate better performance.
HIGHER_IS_BETTER: Set[str] = {
    "spearman_rho",
    "pearson_r",
    "r2",
    "auroc",
    "auprc",
    "accuracy",
    "balanced_accuracy",
    "f1",
    "precision",
    "precision_ppv",
    "recall_sensitivity",
    "specificity",
    "npv",
    "needle_single_token_rate",
}

# Metrics where lower values indicate better performance.
LOWER_IS_BETTER: Set[str] = {
    "mae",
    "rmse",
    "brier",
    "cost_usd_actual",
    "cost_usd_apriori",
    "runtime_seconds",
    "api_calls",
    "prompt_tokens",
    "cached_tokens",
    "uncached_prompt_tokens",
    "completion_tokens",
    "reasoning_tokens",
    "total_tokens",
    "n_skipped",
    "malformed_responses",
}

# Map raw column / JSON keys to canonical metric names.
METRIC_ALIASES: Dict[str, str] = {
    "dispersion_spearman_rho": "spearman_rho",
    "dispersion_mae": "mae",
    "dispersion_rmse": "rmse",
    "dispersion_high_low_accuracy": "accuracy",
    "dispersion_high_low_f1": "f1",
    "dispersion_high_low_auroc": "auroc",
    "relapse_accuracy": "accuracy",
    "relapse_f1": "f1",
    "relapse_auroc": "auroc",
    "relapse_auprc": "auprc",
    "spearman_r": "spearman_rho",
    "pearson_r": "pearson_r",
    "recall_sensitivity": "recall_sensitivity",
    "precision_ppv": "precision_ppv",
    "estimated_cost_usd": "cost_usd_actual",
    "cache_aware_estimated_cost_usd": "cost_usd_apriori",
    "no_cache_estimated_cost_usd": "cost_usd_apriori_no_cache",
    "calls": "api_calls",
    "n_calls": "api_calls",
    "n_predictions": "n_cases",
    "n_skipped_missing_mri": "n_skipped",
}

# Map dataset_key values to normalized modality/pathway labels.
DATASET_KEY_TO_MODALITY: Dict[str, str] = {
    "mri": "mri_only",
    "path": "pathology_only",
    "combined": "mri_plus_pathology",
    "mri_pathcal_weighted": "mri_pathcal_weighted",
}

MODALITY_ALIASES: Dict[str, str] = {
    "mri_only": "mri_only",
    "pathology_only": "pathology_only",
    "mri_plus_pathology": "mri_plus_pathology",
}

TARGET_ALIASES: Dict[str, str] = {
    "dispersion_score": "dispersion_regression",
    "dispersion_high_low": "dispersion_high_low",
    "relapse_status": "relapse",
    "relapse_label": "relapse",
}

# Bootstrap CI column suffix patterns: metric -> (low_suffix, high_suffix)
CI_SUFFIXES = ("_ci_low", "_ci_high")

BEST_METRIC_RULES: Dict[str, Literal["max", "min"]] = {
    "spearman_rho": "max",
    "pearson_r": "max",
    "r2": "max",
    "auroc": "max",
    "auprc": "max",
    "accuracy": "max",
    "f1": "max",
    "mae": "min",
    "rmse": "min",
    "brier": "min",
    "cost_usd_actual": "min",
    "cost_usd_apriori": "min",
}

# File discovery patterns (basename substrings or regex-friendly stems).
DISCOVERY_PATTERNS = {
    "approach1_summary_csv": ["all_tiers_metrics_summary.csv"],
    "approach1_eval_json": ["evaluation_metrics_summary.json"],
    "approach1_cost_aggregate": ["llm_token_cost_report.json"],
    "approach1_cost_per_config": ["token_cost_report.json"],
    "approach1_run_config": ["run_config.json"],
    "approach1_run_log": ["run.log"],
    "approach2_metrics_summary": ["nested_outer_metrics_summary.csv"],
    "approach2_cost_actual": ["llm_token_cost_report.json"],
    "approach2_cost_apriori": ["llm_cost_estimate_apriori.json"],
    "approach2_run_log": ["run_log", "run.log"],
}

DEFAULT_PLOTS = [
    "best_regression_spearman",
    "best_high_low_classification",
    "best_relapse_classification",
    "approach1_shotset",
    "modality_comparison",
    "reasoning_comparison",
    "approach2_model_family",
    "approach2_representation",
    "cost_comparison",
    "performance_vs_cost",
    "performance_vs_runtime",
    "api_token_usage",
    "summary_heatmap",
]
