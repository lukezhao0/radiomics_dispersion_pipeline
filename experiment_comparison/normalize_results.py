"""Normalize extracted metrics into long-form tables and best-metric summaries."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .constants import BEST_METRIC_RULES, HIGHER_IS_BETTER, LOWER_IS_BETTER, TARGET_ALIASES

logger = logging.getLogger(__name__)


def normalize_raw_metrics(df: pd.DataFrame) -> pd.DataFrame:
  """Return long-form normalized metrics with canonical target labels."""
  if df.empty:
    return pd.DataFrame(
      columns=[
        "run_id", "run_label", "approach", "model", "reasoning", "pipeline_version",
        "modality", "shotset", "dataset_key", "representation", "model_key", "pathway",
        "target", "task_type", "metric", "value", "ci_low", "ci_high",
        "source_file", "raw_metric", "extraction_confidence", "notes",
      ]
    )

  out = df.copy()
  out["target"] = out["target"].map(lambda t: TARGET_ALIASES.get(str(t), str(t)))
  out["metric"] = out["metric"].astype(str)
  out["value"] = pd.to_numeric(out["value"], errors="coerce")
  out = out.dropna(subset=["value"])
  return out


def _best_direction(metric: str) -> str:
  if metric in BEST_METRIC_RULES:
    return BEST_METRIC_RULES[metric]
  if metric in HIGHER_IS_BETTER:
    return "max"
  if metric in LOWER_IS_BETTER:
    return "min"
  return "max"


def compute_best_metrics(df: pd.DataFrame) -> pd.DataFrame:
  """
  For each run and target, select the best observed metric value.

  Best is max for performance metrics (Spearman, AUROC, accuracy, F1) and min for
  error/cost metrics (MAE, RMSE, Brier, cost). Operational metrics are excluded.
  """
  if df.empty:
    return pd.DataFrame()

  perf = df[df["task_type"].isin(["regression", "classification"])].copy()
  if perf.empty:
    return pd.DataFrame()

  group_cols = [
    "run_id", "run_label", "approach", "model", "reasoning", "pipeline_version",
    "target", "metric",
  ]
  rows: List[Dict] = []

  for keys, grp in perf.groupby(group_cols, dropna=False):
    direction = _best_direction(str(keys[-1]))
    idx = grp["value"].idxmax() if direction == "max" else grp["value"].idxmin()
    best_row = grp.loc[idx].to_dict()
    best_row["best_rule"] = direction
    best_row["n_candidates"] = len(grp)
    rows.append(best_row)

  best_df = pd.DataFrame(rows)
  best_df["best_metric_name"] = best_df.apply(
    lambda r: f"{r['target']}_{r['metric']}_best", axis=1
  )
  return best_df


def build_data_availability_summary(
  runs_df: pd.DataFrame,
  normalized_df: pd.DataFrame,
  discovered_df: pd.DataFrame,
) -> pd.DataFrame:
  """Summarize which key metrics were found per run."""
  key_metrics = [
    ("dispersion_regression", "spearman_rho"),
    ("dispersion_regression", "mae"),
    ("dispersion_high_low", "accuracy"),
    ("dispersion_high_low", "auroc"),
    ("relapse", "auroc"),
    ("relapse", "f1"),
    ("operational", "cost_usd_actual"),
    ("operational", "cost_usd_apriori"),
    ("operational", "api_calls"),
    ("operational", "runtime_seconds"),
  ]

  run_ids = list(runs_df["run_id"].unique()) if not runs_df.empty else []
  if not run_ids and not normalized_df.empty:
    run_ids = list(normalized_df["run_id"].unique())

  rows = []
  for run_id in run_ids:
    run_metrics = normalized_df[normalized_df["run_id"] == run_id]
    run_disc = discovered_df[discovered_df["run_id"] == run_id] if not discovered_df.empty else pd.DataFrame()
    row = {
      "run_id": run_id,
      "n_discovered_files": len(run_disc),
      "n_metric_rows": len(run_metrics),
      "has_run_directory": bool(len(run_disc)),
    }
    for target, metric in key_metrics:
      found = run_metrics[
        (run_metrics["target"] == target) & (run_metrics["metric"] == metric)
      ]
      col = f"has_{target}_{metric}"
      row[col] = bool(len(found))
      row[f"n_{target}_{metric}"] = len(found)
    rows.append(row)
  return pd.DataFrame(rows)


def build_run_metadata_table(runs: list, manual: list) -> pd.DataFrame:
  rows = []
  for r in runs:
    rows.append(
      {
        "run_id": r.id,
        "label": r.label,
        "approach": r.approach,
        "model": r.model,
        "reasoning": r.reasoning,
        "pipeline_version": r.pipeline_version,
        "path": str(r.path) if r.path else "",
        "notes": r.notes,
        "source": "directory",
      }
    )
  for m in manual:
    rows.append(
      {
        "run_id": m.id,
        "label": m.label,
        "approach": m.approach,
        "model": m.model,
        "reasoning": m.reasoning,
        "pipeline_version": m.pipeline_version,
        "path": "",
        "notes": m.notes,
        "source": "manual",
      }
    )
  return pd.DataFrame(rows)
