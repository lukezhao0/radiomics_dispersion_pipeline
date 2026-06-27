"""Discover result artifacts inside run directories."""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from .constants import DISCOVERY_PATTERNS
from .load_config import RunConfig

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".csv", ".json", ".jsonl", ".txt", ".md", ".html"}


@dataclass
class DiscoveredFile:
    run_id: str
    run_label: str
    approach: str
    file_path: str
    relative_path: str
    file_type: str
    artifact_kind: str
    shotset: str = ""
    modality: str = ""
    confidence: str = "high"


def _classify_artifact(rel_path: str, filename: str) -> str:
  """Map a file to a known artifact kind using name patterns."""
  lower = filename.lower()
  rel_lower = rel_path.lower()

  for kind, patterns in DISCOVERY_PATTERNS.items():
    for pat in patterns:
      if pat.lower() in lower or pat.lower() in rel_lower:
        return kind

  if "metrics" in lower and lower.endswith(".csv"):
    return "metrics_csv"
  if lower.endswith("evaluation_metrics_summary.json"):
    return "approach1_eval_json"
  if lower.endswith("nested_outer_metrics_summary.csv"):
    return "approach2_metrics_summary"
  if "cost" in lower and lower.endswith(".json"):
    return "cost_json"
  if lower.endswith(".html") and "report" in lower:
    return "html_report"
  if lower.endswith("run.log") or "run_log" in lower:
    return "run_log"
  return "other"


def _parse_shotset_modality(rel_path: str) -> tuple[str, str]:
  """Extract shotset and modality from Approach 1 nested paths."""
  parts = Path(rel_path).parts
  shotset = ""
  modality = ""
  for i, part in enumerate(parts):
    if part.startswith("shotset_"):
      shotset = part
      if i + 1 < len(parts):
        modality = parts[i + 1]
      break
  return shotset, modality


def discover_run_files(run: RunConfig) -> List[DiscoveredFile]:
  """Walk a run directory and catalog likely result files."""
  found: List[DiscoveredFile] = []
  if run.path is None or not run.path.is_dir():
    logger.warning("Run directory missing for %s: %s", run.id, run.path)
    return found

  root = run.path
  for dirpath, _dirnames, filenames in os.walk(root):
    for fname in sorted(filenames):
      ext = os.path.splitext(fname)[1].lower()
      if ext not in SUPPORTED_EXTENSIONS:
        continue
      abs_path = os.path.join(dirpath, fname)
      rel_path = os.path.relpath(abs_path, root)
      artifact_kind = _classify_artifact(rel_path, fname)
      shotset, modality = _parse_shotset_modality(rel_path)
      confidence = "high" if artifact_kind != "other" else "low"
      found.append(
        DiscoveredFile(
          run_id=run.id,
          run_label=run.label,
          approach=run.approach,
          file_path=abs_path,
          relative_path=rel_path,
          file_type=ext.lstrip("."),
          artifact_kind=artifact_kind,
          shotset=shotset,
          modality=modality,
          confidence=confidence,
        )
      )
  logger.info("Discovered %d files for run %s", len(found), run.id)
  return found


def discover_all_runs(runs: List[RunConfig]) -> List[DiscoveredFile]:
  all_files: List[DiscoveredFile] = []
  for run in runs:
    all_files.extend(discover_run_files(run))
  return all_files


def discovered_files_to_dataframe(files: List[DiscoveredFile]) -> pd.DataFrame:
  if not files:
    return pd.DataFrame(
      columns=[
        "run_id",
        "run_label",
        "approach",
        "file_path",
        "relative_path",
        "file_type",
        "artifact_kind",
        "shotset",
        "modality",
        "confidence",
      ]
    )
  return pd.DataFrame([f.__dict__ for f in files])
