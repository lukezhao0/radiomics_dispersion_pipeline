"""Parse metrics from discovered result artifacts."""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .constants import CI_SUFFIXES, DATASET_KEY_TO_MODALITY, METRIC_ALIASES
from .discover_results import DiscoveredFile
from .load_config import ManualResultConfig, RunConfig

logger = logging.getLogger(__name__)


@dataclass
class RawMetric:
    run_id: str
    run_label: str
    approach: str
    model: str
    reasoning: str
    pipeline_version: str
    metric: str
    value: float
    raw_metric: str
    target: str
    task_type: str
    modality: str
    shotset: str
    dataset_key: str
    representation: str
    model_key: str
    pathway: str
    ci_low: Optional[float]
    ci_high: Optional[float]
    source_file: str
    extraction_confidence: str = "high"
    notes: str = ""


def _canonical_metric(name: str, aliases: Optional[Dict[str, str]] = None) -> str:
  key = name.strip()
  if aliases and key in aliases:
    key = aliases[key]
  return METRIC_ALIASES.get(key, key)


def _safe_float(val: Any) -> Optional[float]:
  if val is None or (isinstance(val, float) and pd.isna(val)):
    return None
  try:
    return float(val)
  except (TypeError, ValueError):
    return None


def _emit(
  *,
  run: RunConfig,
  metric: str,
  value: Any,
  raw_metric: str,
  target: str,
  task_type: str,
  modality: str = "",
  shotset: str = "",
  dataset_key: str = "",
  representation: str = "",
  model_key: str = "",
  pathway: str = "",
  ci_low: Any = None,
  ci_high: Any = None,
  source_file: str,
  confidence: str = "high",
  notes: str = "",
  aliases: Optional[Dict[str, str]] = None,
) -> Optional[RawMetric]:
  fv = _safe_float(value)
  if fv is None:
    return None
  return RawMetric(
    run_id=run.id,
    run_label=run.label,
    approach=run.approach,
    model=run.model,
    reasoning=run.reasoning,
    pipeline_version=run.pipeline_version,
    metric=_canonical_metric(metric, aliases),
    value=fv,
    raw_metric=raw_metric,
    target=target,
    task_type=task_type,
    modality=modality,
    shotset=shotset,
    dataset_key=dataset_key,
    representation=representation,
    model_key=model_key,
    pathway=pathway,
    ci_low=_safe_float(ci_low),
    ci_high=_safe_float(ci_high),
    source_file=source_file,
    extraction_confidence=confidence,
    notes=notes,
  )


def _parse_ci_columns(row: pd.Series, base_metric: str) -> Tuple[Optional[float], Optional[float]]:
  low_col = f"{base_metric}_ci_low"
  high_col = f"{base_metric}_ci_high"
  return _safe_float(row.get(low_col)), _safe_float(row.get(high_col))


def extract_approach1_summary_csv(run: RunConfig, path: str, aliases: Dict[str, str]) -> List[RawMetric]:
  metrics: List[RawMetric] = []
  try:
    df = pd.read_csv(path)
  except Exception as exc:
    logger.warning("Failed to read %s: %s", path, exc)
    return metrics

  for _, row in df.iterrows():
    shotset = str(row.get("shotset_name", ""))
    modality = str(row.get("modality", ""))
    n_pred = row.get("n_predictions")
    n_skip = row.get("n_skipped_missing_mri")

    for col in df.columns:
      if col in {"shotset_name", "modality"}:
        continue
      if col == "n_predictions":
        m = _emit(
          run=run, metric="n_cases", value=n_pred, raw_metric=col,
          target="operational", task_type="operational", modality=modality,
          shotset=shotset, source_file=path, aliases=aliases,
        )
        if m:
          metrics.append(m)
        continue
      if col == "n_skipped_missing_mri":
        m = _emit(
          run=run, metric="n_skipped", value=n_skip, raw_metric=col,
          target="operational", task_type="operational", modality=modality,
          shotset=shotset, source_file=path, aliases=aliases,
        )
        if m:
          metrics.append(m)
        continue

      if col.startswith("dispersion_") and "high_low" not in col:
        target, task_type = "dispersion_regression", "regression"
      elif "dispersion_high_low" in col:
        target, task_type = "dispersion_high_low", "classification"
      elif col.startswith("relapse_"):
        target, task_type = "relapse", "classification"
      elif col.startswith("needle_"):
        target, task_type = "needle_retrieval", "operational"
      else:
        target, task_type = "unknown", "unknown"

      notes = ""
      if col in ("dispersion_high_low_auroc", "dispersion_high_low_auprc"):
        notes = "score_source=dispersion_score_pred"
      m = _emit(
        run=run, metric=col, value=row[col], raw_metric=col,
        target=target, task_type=task_type, modality=modality, shotset=shotset,
        source_file=path, aliases=aliases, notes=notes,
      )
      if m:
        metrics.append(m)
  return metrics


def extract_approach1_eval_json(
  run: RunConfig, path: str, shotset: str, modality: str, aliases: Dict[str, str]
) -> List[RawMetric]:
  metrics: List[RawMetric] = []
  try:
    with open(path, "r", encoding="utf-8") as f:
      data = json.load(f)
  except Exception as exc:
    logger.warning("Failed to read %s: %s", path, exc)
    return metrics

  n_rows = data.get("n_rows")
  m = _emit(
    run=run, metric="n_cases", value=n_rows, raw_metric="n_rows",
    target="operational", task_type="operational", modality=modality,
    shotset=shotset, source_file=path, aliases=aliases,
  )
  if m:
    metrics.append(m)

  section_map = [
    ("dispersion_regression", "dispersion_regression", "regression"),
    ("dispersion_high_low", "dispersion_high_low", "classification"),
    ("relapse_label", "relapse", "classification"),
    ("needle_retrieval", "needle_retrieval", "operational"),
  ]
  for section_key, target, task_type in section_map:
    section = data.get(section_key, {}) or {}
    for k, v in section.items():
      if k == "confusion_matrix":
        continue
      note = ""
      if section_key == "dispersion_high_low" and k in ("auroc", "auprc"):
        note = "score_source=dispersion_score_pred"
      m = _emit(
        run=run, metric=k, value=v, raw_metric=f"{section_key}.{k}",
        target=target, task_type=task_type, modality=modality, shotset=shotset,
        source_file=path, aliases=aliases, notes=note,
      )
      if m:
        metrics.append(m)

  relapse_comp = data.get("relapse_predictor_comparison", {}) or {}
  for predictor, pred_metrics in relapse_comp.items():
    if not isinstance(pred_metrics, dict):
      continue
    for k, v in pred_metrics.items():
      if k == "note":
        continue
      m = _emit(
        run=run, metric=k, value=v, raw_metric=f"relapse_predictor_comparison.{predictor}.{k}",
        target="relapse", task_type="classification", modality=modality, shotset=shotset,
        model_key=predictor, source_file=path, aliases=aliases,
        notes=f"relapse_predictor={predictor}",
      )
      if m:
        metrics.append(m)
  return metrics


def extract_approach2_metrics_csv(run: RunConfig, path: str, aliases: Dict[str, str]) -> List[RawMetric]:
  metrics: List[RawMetric] = []
  try:
    df = pd.read_csv(path)
  except Exception as exc:
    logger.warning("Failed to read %s: %s", path, exc)
    return metrics

  metric_cols = [
    "auroc", "auprc", "brier", "accuracy", "balanced_accuracy", "f1",
    "precision", "precision_ppv", "recall_sensitivity", "specificity", "npv",
    "mae", "rmse", "r2", "spearman_rho", "pearson_r",
    "calibration_intercept", "calibration_slope",
    "tn", "fp", "fn", "tp", "n", "prevalence",
  ]

  for _, row in df.iterrows():
    dataset_key = str(row.get("dataset_key", ""))
    modality = DATASET_KEY_TO_MODALITY.get(dataset_key, dataset_key)
    representation = str(row.get("representation", ""))
    model_key = str(row.get("model_key", ""))
    task_type = str(row.get("task_type", ""))
    target = str(row.get("target_name", ""))

    for col in metric_cols:
      if col not in df.columns:
        continue
      val = row.get(col)
      if pd.isna(val) or val == "":
        continue
      ci_low, ci_high = _parse_ci_columns(row, col)
      m = _emit(
        run=run, metric=col, value=val, raw_metric=col,
        target=target, task_type=task_type, modality=modality,
        dataset_key=dataset_key, representation=representation, model_key=model_key,
        pathway=modality, ci_low=ci_low, ci_high=ci_high,
        source_file=path, aliases=aliases,
      )
      if m:
        metrics.append(m)
  return metrics


def extract_cost_json(run: RunConfig, path: str, aliases: Dict[str, str]) -> List[RawMetric]:
  metrics: List[RawMetric] = []
  try:
    with open(path, "r", encoding="utf-8") as f:
      data = json.load(f)
  except Exception as exc:
    logger.warning("Failed to read %s: %s", path, exc)
    return metrics

  cost_type = str(data.get("cost_type", ""))
  is_apriori = "apriori" in cost_type or "apriori" in os.path.basename(path)

  def _from_block(block: Dict[str, Any], prefix: str = "") -> None:
    if not isinstance(block, dict):
      return
    mapping = {
      "estimated_cost_usd": "cost_usd_actual" if not is_apriori else "cost_usd_apriori",
      "cache_aware_estimated_cost_usd": "cost_usd_apriori",
      "no_cache_estimated_cost_usd": "cost_usd_apriori_no_cache",
      "calls": "api_calls",
      "n_calls": "api_calls",
      "prompt_tokens": "prompt_tokens",
      "cached_tokens": "cached_tokens",
      "uncached_prompt_tokens": "uncached_prompt_tokens",
      "completion_tokens": "completion_tokens",
      "reasoning_tokens": "reasoning_tokens",
      "total_tokens": "total_tokens",
      "estimated_cache_savings_usd": "cache_savings_usd",
    }
    for raw_key, canon in mapping.items():
      if raw_key in block:
        m = _emit(
          run=run, metric=canon, value=block[raw_key], raw_metric=f"{prefix}{raw_key}",
          target="operational", task_type="operational", source_file=path,
          aliases=aliases, notes=f"cost_type={cost_type or 'unknown'}",
        )
        if m:
          metrics.append(m)

  if "cumulative" in data:
    _from_block(data["cumulative"], "cumulative.")
  _from_block(data, "root.")

  return metrics


def _parse_runtime_from_log(path: str) -> Optional[float]:
  """Estimate wall-clock seconds from first and last ISO timestamps in a log."""
  ts_pattern = re.compile(r"\[RESUME SESSION\] (\d{4}-\d{2}-\d{2}T[\d:]+)")
  timestamps: List[str] = []
  try:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
      for line in f:
        m = ts_pattern.search(line)
        if m:
          timestamps.append(m.group(1))
  except OSError:
    return None
  if len(timestamps) < 2:
    return None
  try:
    t0 = pd.Timestamp(timestamps[0])
    t1 = pd.Timestamp(timestamps[-1])
    return float((t1 - t0).total_seconds())
  except Exception:
    return None


def extract_run_log(run: RunConfig, path: str, aliases: Dict[str, str]) -> List[RawMetric]:
  metrics: List[RawMetric] = []
  runtime = _parse_runtime_from_log(path)
  if runtime is not None:
    m = _emit(
      run=run, metric="runtime_seconds", value=runtime, raw_metric="log_timestamp_span",
      target="operational", task_type="operational", source_file=path,
      aliases=aliases, confidence="medium",
      notes="Estimated from first/last [RESUME SESSION] timestamps in log",
    )
    if m:
      metrics.append(m)
  return metrics


def extract_manual_results(manual: List[ManualResultConfig], aliases: Dict[str, str]) -> List[RawMetric]:
  metrics: List[RawMetric] = []
  manual_target_map = {
    "regression_spearman_best": ("dispersion_regression", "regression", "spearman_rho"),
    "high_low_accuracy_best": ("dispersion_high_low", "classification", "accuracy"),
    "high_low_auroc_best": ("dispersion_high_low", "classification", "auroc"),
    "relapse_auroc_best": ("relapse", "classification", "auroc"),
  }
  for entry in manual:
    run = RunConfig(
      id=entry.id,
      label=entry.label,
      approach=entry.approach,
      model=entry.model,
      reasoning=entry.reasoning,
      pipeline_version=entry.pipeline_version,
      path=None,
      notes=entry.notes,
    )
    for raw_key, value in entry.metrics.items():
      if raw_key in manual_target_map:
        target, task_type, metric = manual_target_map[raw_key]
      else:
        target, task_type, metric = "unknown", "unknown", raw_key
      m = _emit(
        run=run, metric=metric, value=value, raw_metric=raw_key,
        target=target, task_type=task_type, source_file="manual_config",
        aliases=aliases, notes="manually supplied legacy metric",
      )
      if m:
        metrics.append(m)
  return metrics


def extract_from_discovered(
  runs: List[RunConfig],
  discovered: List[DiscoveredFile],
  manual: List[ManualResultConfig],
  aliases: Dict[str, str],
) -> List[RawMetric]:
  run_by_id = {r.id: r for r in runs}
  all_metrics: List[RawMetric] = []

  for d in discovered:
    run = run_by_id.get(d.run_id)
    if run is None:
      continue
    path = d.file_path
    kind = d.artifact_kind

    if kind == "approach1_summary_csv":
      all_metrics.extend(extract_approach1_summary_csv(run, path, aliases))
    elif kind == "approach1_eval_json":
      all_metrics.extend(
        extract_approach1_eval_json(run, path, d.shotset, d.modality, aliases)
      )
    elif kind == "approach2_metrics_summary":
      all_metrics.extend(extract_approach2_metrics_csv(run, path, aliases))
    elif kind in {
        "approach1_cost_aggregate",
        "approach1_cost_per_config",
        "approach2_cost_actual",
        "approach2_cost_apriori",
        "approach2_cost_apriori_initial",
        "cost_json",
    }:
      all_metrics.extend(extract_cost_json(run, path, aliases))
    elif kind in {"approach1_run_log", "approach2_run_log", "run_log"}:
      all_metrics.extend(extract_run_log(run, path, aliases))

  all_metrics.extend(extract_manual_results(manual, aliases))
  all_metrics.extend(_aggregate_approach1_per_config_costs(all_metrics, runs))
  logger.info("Extracted %d raw metric rows", len(all_metrics))
  return all_metrics


def _aggregate_approach1_per_config_costs(
  metrics: List[RawMetric], runs: List[RunConfig]
) -> List[RawMetric]:
  """Sum per-config token_cost_report.json rows when no run-level aggregate exists."""
  extra: List[RawMetric] = []
  run_ids_with_aggregate = {
    m.run_id
    for m in metrics
    if m.metric == "cost_usd_actual"
    and "llm_token_cost_report.json" in m.source_file
  }
  cost_metrics = ("cost_usd_actual", "api_calls", "prompt_tokens", "cached_tokens",
                  "uncached_prompt_tokens", "completion_tokens", "reasoning_tokens", "total_tokens")
  for run in runs:
    if run.approach != "approach1" or run.id in run_ids_with_aggregate:
      continue
    per_config = [
      m for m in metrics
      if m.run_id == run.id
      and m.target == "operational"
      and "token_cost_report.json" in m.source_file
      and m.metric in cost_metrics
    ]
    if not per_config:
      continue
    by_metric: Dict[str, float] = {}
    for m in per_config:
      by_metric[m.metric] = by_metric.get(m.metric, 0.0) + m.value
    for metric, value in by_metric.items():
      extra.append(
        RawMetric(
          run_id=run.id,
          run_label=run.label,
          approach=run.approach,
          model=run.model,
          reasoning=run.reasoning,
          pipeline_version=run.pipeline_version,
          metric=metric,
          value=value,
          raw_metric=f"aggregated_per_config_{metric}",
          target="operational",
          task_type="operational",
          modality="",
          shotset="",
          dataset_key="",
          representation="",
          model_key="",
          pathway="",
          ci_low=None,
          ci_high=None,
          source_file="aggregated_from_per_config_token_cost_report",
          extraction_confidence="medium",
          notes="Sum of per-config token_cost_report.json (no run-level aggregate found)",
        )
      )
  return extra


def raw_metrics_to_dataframe(metrics: List[RawMetric]) -> pd.DataFrame:
  if not metrics:
    return pd.DataFrame()
  return pd.DataFrame([m.__dict__ for m in metrics])
