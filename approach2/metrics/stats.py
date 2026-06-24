"""Pure statistical helpers for approach2 evaluation metrics."""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error


def safe_spearman(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    if len(x) < 3 or len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return np.nan, np.nan
    rho, p = spearmanr(x, y)
    return float(rho), float(p)


def safe_pearson(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    if len(x) < 3 or len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return np.nan, np.nan
    r, p = pearsonr(x, y)
    return float(r), float(p)


def rmse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def calibration_intercept_slope(y_true: np.ndarray, prob: np.ndarray) -> Tuple[float, float]:
    prob = np.clip(np.asarray(prob, dtype=float), 1e-6, 1 - 1e-6)
    logits = np.log(prob / (1.0 - prob)).reshape(-1, 1)
    y_true = np.asarray(y_true, dtype=int)
    if len(np.unique(y_true)) < 2:
        return np.nan, np.nan
    try:
        model = LogisticRegression(C=1e6, solver="lbfgs")
        model.fit(logits, y_true)
        return float(model.intercept_[0]), float(model.coef_[0][0])
    except Exception:
        return np.nan, np.nan
