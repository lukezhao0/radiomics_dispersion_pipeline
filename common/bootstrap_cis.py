"""Case-level bootstrap percentile confidence intervals for metric dictionaries."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

MetricFn = Callable[[pd.DataFrame], Optional[float]]


@dataclass(frozen=True)
class BootstrapMetricSpec:
    task: str
    metric: str
    compute: MetricFn


@dataclass
class BootstrapCIResult:
    task: str
    metric: str
    point_estimate: float
    ci_lower: float
    ci_upper: float
    n_bootstrap_requested: int
    n_bootstrap_valid: int
    notes: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def bootstrap_percentile_cis(
    df: pd.DataFrame,
    specs: Sequence[BootstrapMetricSpec],
    n_bootstrap: int,
    random_seed: int,
    *,
    ci_percentiles: Tuple[float, float] = (2.5, 97.5),
) -> List[BootstrapCIResult]:
    """Resample rows with replacement and compute percentile CIs per metric.

    Each ``spec.compute`` should return a finite float on success, or ``None`` when
    the metric is undefined for that bootstrap replicate (e.g. single-class AUROC).
    """
    if n_bootstrap <= 0 or len(df) < 2:
        return []

    rng = np.random.default_rng(int(random_seed))
    work = df.reset_index(drop=True).copy()
    results: List[BootstrapCIResult] = []

    for spec in specs:
        point = spec.compute(work)
        point_val = float(point) if point is not None and np.isfinite(point) else np.nan

        samples: List[float] = []
        n_skipped = 0
        for _ in range(int(n_bootstrap)):
            idx = rng.integers(0, len(work), len(work))
            boot = work.iloc[idx].copy()
            try:
                val = spec.compute(boot)
            except Exception:
                n_skipped += 1
                continue
            if val is None or not np.isfinite(val):
                n_skipped += 1
                continue
            samples.append(float(val))

        n_valid = len(samples)
        if n_valid > 0:
            ci_low = float(np.percentile(samples, ci_percentiles[0]))
            ci_high = float(np.percentile(samples, ci_percentiles[1]))
        else:
            ci_low = np.nan
            ci_high = np.nan

        notes_parts: List[str] = []
        if n_skipped > 0:
            notes_parts.append(
                f"Skipped {n_skipped}/{n_bootstrap} replicates (metric undefined or failed)"
            )
        if n_valid == 0:
            notes_parts.append("No valid bootstrap replicates; CI bounds are NA")
        if point is None or not np.isfinite(point_val):
            notes_parts.append("Point estimate undefined on full dataset")

        results.append(
            BootstrapCIResult(
                task=spec.task,
                metric=spec.metric,
                point_estimate=point_val,
                ci_lower=ci_low,
                ci_upper=ci_high,
                n_bootstrap_requested=int(n_bootstrap),
                n_bootstrap_valid=n_valid,
                notes="; ".join(notes_parts),
            )
        )

    return results


def bootstrap_results_to_dataframe(results: Sequence[BootstrapCIResult]) -> pd.DataFrame:
    if not results:
        return pd.DataFrame(
            columns=[
                "task",
                "metric",
                "point_estimate",
                "ci_lower",
                "ci_upper",
                "n_bootstrap_requested",
                "n_bootstrap_valid",
                "notes",
            ]
        )
    return pd.DataFrame([r.to_dict() for r in results])
