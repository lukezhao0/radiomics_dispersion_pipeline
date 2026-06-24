"""Evaluation plot generation."""

from __future__ import annotations

from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_dispersion_scatter(used: pd.DataFrame, out_path: str, title_suffix: str) -> None:
    y_true = used["dispersion_true"].astype(float).values
    y_pred = used["dispersion_score_pred"].astype(float).values
    plt.figure()
    plt.scatter(y_true, y_pred)
    lo = float(np.nanmin([y_true.min(), y_pred.min()]))
    hi = float(np.nanmax([y_true.max(), y_pred.max()]))
    if lo == hi:
        lo -= 1.0
        hi += 1.0
    plt.plot([lo, hi], [lo, hi])
    plt.xlabel("True dispersion score")
    plt.ylabel("Predicted dispersion score")
    plt.title(f"True vs Predicted Dispersion Score ({title_suffix})")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_dispersion_residuals(used: pd.DataFrame, out_path: str, title_suffix: str) -> None:
    residuals = used["dispersion_score_pred"].astype(float).values - used["dispersion_true"].astype(float).values
    plt.figure()
    plt.hist(residuals, bins=25)
    plt.xlabel("Residual (predicted - true)")
    plt.ylabel("Count")
    plt.title(f"Dispersion Score Residuals ({title_suffix})")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_label_confusion_matrix(cm: np.ndarray, out_path: str, title: str) -> None:
    plt.figure()
    plt.imshow(cm)
    plt.title(title)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.xticks([0, 1], ["0", "1"])
    plt.yticks([0, 1], ["0", "1"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(int(cm[i, j])), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_pred_dispersion_by_relapse(df: pd.DataFrame, out_path: str, title_suffix: str) -> None:
    mask = df["relapse_true"].notna() & np.isfinite(df["dispersion_score_pred"])
    used = df.loc[mask].copy()
    if len(used) == 0:
        return
    g0 = used.loc[used["relapse_true"] == 0, "dispersion_score_pred"].astype(float).values
    g1 = used.loc[used["relapse_true"] == 1, "dispersion_score_pred"].astype(float).values
    plt.figure()
    if len(g0) > 0 and len(g1) > 0:
        plt.violinplot([g0, g1], showmeans=True, showextrema=True)
        plt.xticks([1, 2], ["Relapse=0", "Relapse=1"])
    else:
        plt.hist(used["dispersion_score_pred"].astype(float).values, bins=25)
        plt.xlabel("Predicted dispersion score")
    plt.ylabel("Predicted dispersion score")
    plt.title(f"Predicted Dispersion by True Relapse ({title_suffix})")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_relapse_predictor_comparison(metrics: Dict[str, Any], out_path: str, title_suffix: str) -> None:
    methods = ["llm_relapse_pred", "predicted_dispersion_score", "true_dispersion_score"]
    labels = ["LLM relapse_pred", "Pred disp score", "True disp score"]
    auroc_vals: List[float] = []
    auprc_vals: List[float] = []
    f1_vals: List[float] = []
    for m in methods:
        d = metrics.get(m, {})
        auroc_vals.append(d.get("auroc", np.nan) if d.get("auroc", None) is not None else np.nan)
        auprc_vals.append(d.get("auprc", np.nan) if d.get("auprc", None) is not None else np.nan)
        f1_vals.append(d.get("f1", np.nan) if m == "llm_relapse_pred" else d.get("best_f1", np.nan))
    x = np.arange(len(methods))
    width = 0.25
    plt.figure()
    plt.bar(x - width, auroc_vals, width)
    plt.bar(x, auprc_vals, width)
    plt.bar(x + width, f1_vals, width)
    plt.xticks(x, labels, rotation=20, ha="right")
    plt.ylabel("Metric value")
    plt.title(f"Relapse Predictor Comparison ({title_suffix})")
    plt.legend(["AUROC", "AUPRC", "F1"])
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_needle_retrieval_rates(df: pd.DataFrame, out_path: str, title_suffix: str) -> None:
    if "retrieval_token_exact_match" not in df.columns or len(df) == 0:
        return
    val = float(df["retrieval_token_exact_match"].mean())
    plt.figure()
    plt.bar(["Single token"], [val])
    plt.ylim(0, 1.05)
    plt.ylabel("Exact retrieval rate")
    plt.title(f"Needle Retrieval Accuracy ({title_suffix})")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_top_evidence_features(
    tbl: pd.DataFrame,
    out_path: str,
    title: str,
    positive: bool = True,
    min_support: int = 3,
    top_k: int = 15,
) -> None:
    if tbl is None or len(tbl) == 0:
        return
    sub = tbl[tbl["support"] >= min_support].copy()
    if len(sub) == 0:
        return
    sub = sub.sort_values("odds_ratio_pos_vs_neg", ascending=not positive).head(top_k)
    labels = sub["feature"].astype(str).tolist()
    vals = sub["odds_ratio_pos_vs_neg"].astype(float).tolist()
    plt.figure(figsize=(10, 6))
    y = np.arange(len(labels))
    plt.barh(y, vals)
    plt.yticks(y, labels)
    plt.xlabel("Smoothed odds ratio (class 1 vs class 0)")
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
