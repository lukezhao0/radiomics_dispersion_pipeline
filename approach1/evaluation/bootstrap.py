"""Bootstrap confidence intervals for Approach 1 held-out metrics."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, mean_absolute_error, recall_score
from scipy.stats import spearmanr

from common.bootstrap_cis import (
    BootstrapCIResult,
    BootstrapMetricSpec,
    bootstrap_percentile_cis,
    bootstrap_results_to_dataframe,
)

from ..config import DEFAULT_BOOTSTRAP_N, DEFAULT_BOOTSTRAP_SEED
from .metrics import prepare_predictions_for_eval, safe_auroc_auprc

BOOTSTRAP_CSV_FILENAME = "bootstrap_metric_cis.csv"
BOOTSTRAP_JSON_FILENAME = "bootstrap_metric_cis.json"


def _finite_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> Optional[float]:
    if len(y_true) < 2:
        return None
    rho = spearmanr(y_true, y_pred).correlation
    if rho is None or not np.isfinite(rho):
        return None
    return float(rho)


def _binary_label_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Optional[float]]:
    if len(y_true) == 0:
        return {"f1": None, "sensitivity": None, "specificity": None}
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    f1v = float(f1_score(y_true, y_pred, zero_division=0))
    sensitivity = float(recall_score(y_true, y_pred, zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape != (2, 2):
        return {"f1": f1v, "sensitivity": sensitivity, "specificity": None}
    tn, fp, fn, tp = cm.ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else None
    return {"f1": f1v, "sensitivity": sensitivity, "specificity": specificity}


def _continuous_dispersion_metrics(df: pd.DataFrame) -> Dict[str, Optional[float]]:
    mask = np.isfinite(df["dispersion_true"]) & np.isfinite(df["dispersion_score_pred"])
    used = df.loc[mask]
    if len(used) == 0:
        return {"spearman_rho": None, "mae": None}
    y_true = used["dispersion_true"].astype(float).values
    y_pred = used["dispersion_score_pred"].astype(float).values
    return {
        "spearman_rho": _finite_spearman(y_true, y_pred),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def _high_low_dispersion_metrics(df: pd.DataFrame) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {
        "auroc": None,
        "auprc": None,
        "f1": None,
        "sensitivity": None,
        "specificity": None,
    }

    label_mask = df["dispersion_true_high_low"].notna() & df["dispersion_high_low_pred"].notna()
    if label_mask.any():
        y_true = df.loc[label_mask, "dispersion_true_high_low"].astype(int).values
        y_pred = df.loc[label_mask, "dispersion_high_low_pred"].astype(int).values
        out.update(_binary_label_metrics(y_true, y_pred))

    score_mask = df["dispersion_true_high_low"].notna() & np.isfinite(df["dispersion_score_pred"])
    if score_mask.any():
        y_true_score = df.loc[score_mask, "dispersion_true_high_low"].astype(int).values
        scores = df.loc[score_mask, "dispersion_score_pred"].astype(float).values
        auroc, auprc, _ = safe_auroc_auprc(y_true_score, scores)
        out["auroc"] = auroc
        out["auprc"] = auprc

    return out


def _relapse_metrics(df: pd.DataFrame) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {
        "auroc": None,
        "auprc": None,
        "f1": None,
        "sensitivity": None,
        "specificity": None,
    }
    mask = df["relapse_true"].notna() & df["relapse_pred"].notna()
    if not mask.any():
        return out
    y_true = df.loc[mask, "relapse_true"].astype(int).values
    y_pred = df.loc[mask, "relapse_pred"].astype(int).values
    out.update(_binary_label_metrics(y_true, y_pred))
    auroc, auprc, _ = safe_auroc_auprc(y_true, y_pred.astype(float))
    out["auroc"] = auroc
    out["auprc"] = auprc
    return out


def approach1_bootstrap_metric_specs() -> List[BootstrapMetricSpec]:
    specs: List[BootstrapMetricSpec] = []

    def add_task(task: str, metric: str, fn):
        specs.append(
            BootstrapMetricSpec(
                task=task,
                metric=metric,
                compute=lambda df, _fn=fn, _metric=metric: _fn(df).get(_metric),
            )
        )

    for metric in ("spearman_rho", "mae"):
        add_task("continuous_dispersion", metric, _continuous_dispersion_metrics)
    for metric in ("auroc", "auprc", "f1", "sensitivity", "specificity"):
        add_task("high_low_dispersion", metric, _high_low_dispersion_metrics)
    for metric in ("auroc", "auprc", "f1", "sensitivity", "specificity"):
        add_task("relapse_prediction", metric, _relapse_metrics)

    return specs


def compute_approach1_bootstrap_cis(
    pred_df: pd.DataFrame,
    *,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_N,
    random_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> List[BootstrapCIResult]:
    df = prepare_predictions_for_eval(pred_df)
    if n_bootstrap <= 0 or len(df) < 2:
        return []
    return bootstrap_percentile_cis(
        df,
        approach1_bootstrap_metric_specs(),
        n_bootstrap=n_bootstrap,
        random_seed=random_seed,
    )


def bootstrap_metadata(n_rows: int, n_bootstrap: int, random_seed: int) -> Dict[str, Any]:
    return {
        "n_cases": int(n_rows),
        "n_bootstrap_requested": int(n_bootstrap),
        "random_seed": int(random_seed),
        "ci_method": "case-level percentile bootstrap (2.5th, 97.5th percentiles)",
        "relapse_auroc_note": (
            "relapse_pred is the LLM binary relapse label (0/1), not a continuous probability; "
            "AUROC/AUPRC rank cases by this binary score and should be interpreted accordingly."
        ),
    }


def save_bootstrap_results(
    results: Sequence[BootstrapCIResult],
    out_dir: str,
    *,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, BOOTSTRAP_CSV_FILENAME)
    json_path = os.path.join(out_dir, BOOTSTRAP_JSON_FILENAME)

    df = bootstrap_results_to_dataframe(results)
    df.to_csv(csv_path, index=False)

    payload = {
        "metadata": metadata or {},
        "metrics": [r.to_dict() for r in results],
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return {"csv": csv_path, "json": json_path}


def compute_and_save_bootstrap_cis(
    pred_df: pd.DataFrame,
    out_dir: str,
    *,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_N,
    random_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> Dict[str, Any]:
    df = prepare_predictions_for_eval(pred_df)
    results = compute_approach1_bootstrap_cis(
        df,
        n_bootstrap=n_bootstrap,
        random_seed=random_seed,
    )
    paths = save_bootstrap_results(
        results,
        out_dir,
        metadata=bootstrap_metadata(len(df), n_bootstrap, random_seed),
    )
    if results:
        print(f"[BOOTSTRAP] Wrote {len(results)} metric CIs to: {paths['csv']}")
    return {
        "n_metrics": len(results),
        "paths": paths,
        "metrics": [r.to_dict() for r in results],
        "metadata": bootstrap_metadata(len(df), n_bootstrap, random_seed),
    }
