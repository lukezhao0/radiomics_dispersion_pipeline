"""Dispersion, relapse, and needle-retrieval metric computation."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    roc_auc_score,
)

from ..config import DISPERSION_HIGH_THRESHOLD
from ..schema.records import parse_jsonish_list
from ..text_utils import rmse


def coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def coerce_int01(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    return s.where(s.isin([0, 1]), np.nan)


def prepare_predictions_for_eval(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "dispersion_true_high_low" not in df.columns:
        df["dispersion_true_high_low"] = df["dispersion_true"].apply(
            lambda x: int(float(x) >= DISPERSION_HIGH_THRESHOLD) if pd.notna(x) else np.nan
        )
    for col in ["row_index", "dispersion_true", "dispersion_score_pred"]:
        if col in df.columns:
            df[col] = coerce_numeric(df[col])
    for col in [
        "relapse_true",
        "relapse_pred",
        "dispersion_true_high_low",
        "dispersion_high_low_pred",
        "retrieval_token_exact_match",
    ]:
        if col in df.columns:
            df[col] = coerce_int01(df[col])
    if "key_evidence" in df.columns:
        df["key_evidence_list"] = df["key_evidence"].apply(parse_jsonish_list)
    else:
        df["key_evidence_list"] = [[] for _ in range(len(df))]
    return df


def missingness_summary(df: pd.DataFrame) -> str:
    cols = [
        "dispersion_true",
        "dispersion_true_high_low",
        "dispersion_score_pred",
        "dispersion_high_low_pred",
        "relapse_true",
        "relapse_pred",
        "row_index",
        "retrieval_token_exact_match",
    ]
    lines = ["Missingness / validity summary (count of NaN):"]
    for c in cols:
        if c in df.columns:
            lines.append(f"  {c:<34} {int(df[c].isna().sum())}")
    return "\n".join(lines)


def evaluate_dispersion(df: pd.DataFrame) -> Tuple[str, pd.DataFrame, Dict[str, Any]]:
    mask = np.isfinite(df["dispersion_true"]) & np.isfinite(df["dispersion_score_pred"])
    used = df.loc[mask].copy()
    if len(used) == 0:
        return "Dispersion score (regression):\n  No valid rows.", used, {}

    y_true = used["dispersion_true"].astype(float).values
    y_pred = used["dispersion_score_pred"].astype(float).values
    mae = mean_absolute_error(y_true, y_pred)
    rmse_val = rmse(y_true, y_pred)
    rho = spearmanr(y_true, y_pred).correlation
    rho_val = float(rho) if (rho is not None and not np.isnan(rho)) else np.nan

    lines = [
        "Dispersion score (regression):",
        f"  N_used = {len(used)} / {len(df)}",
        f"  MAE  = {mae:.4f}",
        f"  RMSE = {rmse_val:.4f}",
        f"  Spearman rho = {rho_val:.4f}" if np.isfinite(rho_val) else "  Spearman rho = nan",
    ]
    return "\n".join(lines), used, {"mae": mae, "rmse": rmse_val, "spearman_rho": rho_val}


def evaluate_dispersion_high_low(df: pd.DataFrame) -> Tuple[str, pd.DataFrame, Dict[str, Any]]:
    mask = df["dispersion_true_high_low"].notna() & df["dispersion_high_low_pred"].notna()
    used = df.loc[mask].copy()
    if len(used) == 0:
        return "Dispersion high/low classification:\n  No valid rows.", used, {}

    y_true = used["dispersion_true_high_low"].astype(int).values
    y_pred = used["dispersion_high_low_pred"].astype(int).values
    acc = accuracy_score(y_true, y_pred)
    f1v = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    metrics: Dict[str, Any] = {"accuracy": acc, "f1": f1v, "confusion_matrix": cm}

    score_mask = df["dispersion_true_high_low"].notna() & np.isfinite(df["dispersion_score_pred"])
    score_used = df.loc[score_mask]
    lines = [
        f"Dispersion high/low (true high defined as dispersion_true >= {int(DISPERSION_HIGH_THRESHOLD)}):",
        f"  N_used (label) = {len(used)} / {len(df)}",
        f"  Accuracy = {acc:.4f}  (from dispersion_high_low_pred)",
        f"  F1       = {f1v:.4f}",
        "  Confusion matrix (rows=true [0,1], cols=pred [0,1]):",
        f"  {cm.tolist()}",
    ]

    if len(score_used) > 0:
        y_true_score = score_used["dispersion_true_high_low"].astype(int).values
        scores = score_used["dispersion_score_pred"].astype(float).values
        auroc, auprc, note = safe_auroc_auprc(y_true_score, scores)
        t_best, f1_best = best_f1_threshold(y_true_score, scores)
        metrics.update(
            {
                "auroc": auroc,
                "auprc": auprc,
                "auroc_note": note,
                "best_f1_at_score_threshold": f1_best,
                "best_score_threshold": t_best,
                "auroc_score_source": "dispersion_score_pred",
            }
        )
        lines.extend(
            [
                f"  N_used (score) = {len(score_used)} / {len(df)}",
                "  Score-based ranking (dispersion_score_pred vs true high/low):",
                f"    AUROC = {auroc if auroc is not None else 'NA'}",
                f"    AUPRC = {auprc if auprc is not None else 'NA'}",
                f"    Best F1 @ score threshold = {f1_best:.4f} (t={t_best:.4f})",
                f"    Note: {note}",
            ]
        )
    else:
        metrics["auroc_note"] = "No valid rows with dispersion_score_pred for AUROC."
        lines.append("  Score-based AUROC: no valid dispersion_score_pred rows.")

    return "\n".join(lines), used, metrics


def evaluate_relapse_labels(df: pd.DataFrame) -> Tuple[str, pd.DataFrame, Dict[str, Any]]:
    mask = df["relapse_true"].notna() & df["relapse_pred"].notna()
    used = df.loc[mask].copy()
    if len(used) == 0:
        return "Relapse classification:\n  No valid rows.", used, {}

    y_true = used["relapse_true"].astype(int).values
    y_pred = used["relapse_pred"].astype(int).values
    acc = accuracy_score(y_true, y_pred)
    f1v = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    metrics: Dict[str, Any] = {"accuracy": acc, "f1": f1v, "confusion_matrix": cm}
    auroc, auprc, note = safe_auroc_auprc(y_true, y_pred.astype(float))
    metrics.update(
        {
            "auroc": auroc,
            "auprc": auprc,
            "auroc_note": note,
            "auroc_score_source": "relapse_pred",
        }
    )

    lines = [
        "Relapse (classification):",
        f"  N_used = {len(used)} / {len(df)}",
        f"  Accuracy = {acc:.4f}  (from relapse_pred)",
        f"  F1       = {f1v:.4f}",
        "  Confusion matrix (rows=true [0,1], cols=pred [0,1]):",
        f"  {cm.tolist()}",
        "  Score-based ranking (relapse_pred vs true relapse):",
        f"    AUROC = {auroc if auroc is not None else 'NA'}",
        f"    AUPRC = {auprc if auprc is not None else 'NA'}",
        f"    Note: {note}",
    ]
    return "\n".join(lines), used, metrics


def safe_auroc_auprc(y_true: np.ndarray, scores: np.ndarray) -> Tuple[Optional[float], Optional[float], str]:
    uniq = np.unique(y_true)
    if len(uniq) < 2:
        return None, None, "Only one class present in y_true; AUROC/AUPRC undefined."
    try:
        auroc = float(roc_auc_score(y_true, scores))
    except Exception as e:
        return None, None, f"AUROC failed: {e}"
    try:
        auprc = float(average_precision_score(y_true, scores))
    except Exception as e:
        return auroc, None, f"AUPRC failed: {e}"
    return auroc, auprc, "ok"


def best_f1_threshold(y_true: np.ndarray, scores: np.ndarray) -> Tuple[float, float]:
    vals = np.unique(scores[~np.isnan(scores)])
    if len(vals) == 0:
        return 0.5, float("nan")
    if len(vals) > 400:
        vals = np.unique(np.quantile(scores, np.linspace(0.0, 1.0, 401)))

    best_t = float(vals[0])
    best_f1 = -1.0
    for t in vals:
        pred = (scores >= t).astype(int)
        f1v = f1_score(y_true, pred, zero_division=0)
        if f1v > best_f1:
            best_f1 = float(f1v)
            best_t = float(t)
    return best_t, best_f1


def compare_relapse_predictors(df: pd.DataFrame) -> Tuple[str, Dict[str, Any]]:
    base = df[df["relapse_true"].notna()].copy()
    if len(base) == 0:
        return "Relapse predictor comparison:\n  No valid rows with relapse_true.", {}

    metrics: Dict[str, Any] = {}
    lines = [
        "Relapse predictor comparison (AUROC/AUPRC/F1):",
        f"  N_with_relapse_true = {len(base)} / {len(df)}",
        "",
    ]

    a_mask = base["relapse_pred"].notna()
    if a_mask.sum() > 0:
        y_pred_lbl = base.loc[a_mask, "relapse_pred"].astype(int).values
        y_true_a = base.loc[a_mask, "relapse_true"].astype(int).values
        acc = accuracy_score(y_true_a, y_pred_lbl)
        f1v = f1_score(y_true_a, y_pred_lbl, zero_division=0)
        auroc, auprc, note = safe_auroc_auprc(y_true_a, y_pred_lbl.astype(float))
        metrics["llm_relapse_pred"] = {"accuracy": acc, "f1": f1v, "auroc": auroc, "auprc": auprc, "note": note}
        lines.append("  A) LLM relapse_pred (binary label):")
        lines.append(f"     Accuracy={acc:.4f}  F1={f1v:.4f}  AUROC={auroc if auroc is not None else 'NA'}  AUPRC={auprc if auprc is not None else 'NA'}")
        lines.append(f"     Note: {note}")
    else:
        lines.append("  A) LLM relapse_pred: no valid predictions.")
    lines.append("")

    b_mask = np.isfinite(base["dispersion_score_pred"])
    if b_mask.sum() > 0:
        scores = base.loc[b_mask, "dispersion_score_pred"].astype(float).values
        y_true_b = base.loc[b_mask, "relapse_true"].astype(int).values
        auroc, auprc, note = safe_auroc_auprc(y_true_b, scores)
        t_best, f1_best = best_f1_threshold(y_true_b, scores)
        metrics["predicted_dispersion_score"] = {"auroc": auroc, "auprc": auprc, "note": note, "best_f1": f1_best, "best_threshold": t_best}
        lines.append("  B) Predicted dispersion score (continuous risk score):")
        lines.append(f"     AUROC={auroc if auroc is not None else 'NA'}  AUPRC={auprc if auprc is not None else 'NA'}  BestF1@threshold={f1_best:.4f} (t={t_best:.4f})")
    else:
        lines.append("  B) Predicted dispersion score: no valid values.")
    lines.append("")

    c_mask = np.isfinite(base["dispersion_true"])
    if c_mask.sum() > 0:
        scores = base.loc[c_mask, "dispersion_true"].astype(float).values
        y_true_c = base.loc[c_mask, "relapse_true"].astype(int).values
        auroc, auprc, note = safe_auroc_auprc(y_true_c, scores)
        t_best, f1_best = best_f1_threshold(y_true_c, scores)
        metrics["true_dispersion_score"] = {"auroc": auroc, "auprc": auprc, "note": note, "best_f1": f1_best, "best_threshold": t_best}
        lines.append("  C) True dispersion score (continuous risk score; upper-bound signal check):")
        lines.append(f"     AUROC={auroc if auroc is not None else 'NA'}  AUPRC={auprc if auprc is not None else 'NA'}  BestF1@threshold={f1_best:.4f} (t={t_best:.4f})")
    else:
        lines.append("  C) True dispersion score: no valid values.")

    return "\n".join(lines), metrics


def evaluate_needle_retrieval(df: pd.DataFrame) -> Tuple[str, Dict[str, Any]]:
    if "retrieval_token_exact_match" not in df.columns or len(df) == 0:
        return "Needle retrieval:\n  No rows.", {}
    exact = df["retrieval_token_exact_match"].dropna().astype(int)
    rate = float(exact.mean()) if len(exact) else np.nan
    failures = int((df["retrieval_token_exact_match"] == 0).sum())
    lines = [
        "Needle-in-the-haystack retrieval evaluation:",
        f"  N_rows = {len(df)}",
        f"  Single-token exact retrieval rate = {rate:.4f}" if np.isfinite(rate) else "  Single-token exact retrieval rate = NA",
        f"  Single-token failures = {failures}",
    ]
    return "\n".join(lines), {"single_token_rate": rate, "single_token_failures": failures}
