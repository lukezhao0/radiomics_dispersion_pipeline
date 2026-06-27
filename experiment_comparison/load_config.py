"""Load and validate comparison configuration from YAML or JSON."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .constants import DEFAULT_PLOTS


@dataclass
class RunConfig:
    id: str
    label: str
    approach: str
    model: str
    reasoning: str
    pipeline_version: str
    path: Optional[Path] = None
    notes: str = ""


@dataclass
class ManualResultConfig:
    id: str
    label: str
    approach: str
    model: str
    reasoning: str
    pipeline_version: str
    metrics: Dict[str, float] = field(default_factory=dict)
    notes: str = ""


@dataclass
class OutputConfig:
    output_dir: Path
    html_filename: str = "comparison_report.html"


@dataclass
class ComparisonConfig:
    runs: List[RunConfig]
    manual_results: List[ManualResultConfig]
    output: OutputConfig
    plots: List[str]
    metric_aliases: Dict[str, str]
    config_path: Path


def _require_fields(obj: Dict[str, Any], fields: List[str], context: str) -> None:
    missing = [f for f in fields if f not in obj or obj[f] in (None, "")]
    if missing:
        raise ValueError(f"{context}: missing required field(s): {', '.join(missing)}")


def _load_raw_config(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        data = yaml.safe_load(text)
    elif path.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        raise ValueError(f"Unsupported config format: {path.suffix}")
    if not isinstance(data, dict):
        raise ValueError("Config root must be a mapping/object.")
    return data


def _resolve_path(base_dir: Path, raw_path: Optional[str]) -> Optional[Path]:
    if raw_path is None:
        return None
    p = Path(raw_path)
    if not p.is_absolute():
        p = (base_dir / p).resolve()
    return p


def load_config(config_path: str | Path) -> ComparisonConfig:
    """Load comparison config, resolve relative paths against config file location."""
    config_path = Path(config_path).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")

    raw = _load_raw_config(config_path)
    base_dir = config_path.parent

    runs: List[RunConfig] = []
    for i, entry in enumerate(raw.get("runs", []) or []):
        ctx = f"runs[{i}]"
        _require_fields(
            entry,
            ["id", "label", "approach", "model", "reasoning", "pipeline_version", "path"],
            ctx,
        )
        resolved = _resolve_path(base_dir, entry["path"])
        runs.append(
            RunConfig(
                id=str(entry["id"]),
                label=str(entry["label"]),
                approach=str(entry["approach"]),
                model=str(entry["model"]),
                reasoning=str(entry["reasoning"]),
                pipeline_version=str(entry["pipeline_version"]),
                path=resolved,
                notes=str(entry.get("notes", "")),
            )
        )

    manual_results: List[ManualResultConfig] = []
    for i, entry in enumerate(raw.get("manual_results", []) or []):
        ctx = f"manual_results[{i}]"
        _require_fields(
            entry,
            ["id", "label", "approach", "model", "reasoning", "pipeline_version"],
            ctx,
        )
        metrics_raw = entry.get("metrics", {}) or {}
        metrics = {str(k): float(v) for k, v in metrics_raw.items()}
        manual_results.append(
            ManualResultConfig(
                id=str(entry["id"]),
                label=str(entry["label"]),
                approach=str(entry["approach"]),
                model=str(entry["model"]),
                reasoning=str(entry["reasoning"]),
                pipeline_version=str(entry["pipeline_version"]),
                metrics=metrics,
                notes=str(entry.get("notes", "")),
            )
        )

    output_raw = raw.get("output", {}) or {}
    out_dir = _resolve_path(base_dir, output_raw.get("output_dir", "output"))
    if out_dir is None:
        out_dir = (base_dir / "output").resolve()
    output = OutputConfig(
        output_dir=out_dir,
        html_filename=str(output_raw.get("html_filename", "comparison_report.html")),
    )

    plots = list(raw.get("plots", DEFAULT_PLOTS) or DEFAULT_PLOTS)
    metric_aliases = {str(k): str(v) for k, v in (raw.get("metric_aliases", {}) or {}).items()}

    return ComparisonConfig(
        runs=runs,
        manual_results=manual_results,
        output=output,
        plots=plots,
        metric_aliases=metric_aliases,
        config_path=config_path,
    )


def ensure_output_dir(config: ComparisonConfig) -> Path:
    config.output.output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = config.output.output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    return config.output.output_dir
