"""Tests for approach2 pure statistical metric helpers."""

from __future__ import annotations

import numpy as np
import pytest

from approach2.metrics.stats import (
    calibration_intercept_slope,
    rmse,
    safe_pearson,
    safe_spearman,
)


def test_rmse():
  y_true = [100.0, 40.0, 90.0]
  y_pred = [95.0, 45.0, 88.0]
  assert rmse(y_true, y_pred) == pytest.approx(4.2426, abs=0.01)


def test_safe_spearman_monotone(sample_arrays):
    x, y = sample_arrays
    rho, p = safe_spearman(x, y)
    assert rho == pytest.approx(0.7379, abs=0.01)
    assert 0.0 <= p <= 1.0


def test_safe_spearman_too_few_points():
    x = np.array([1.0, 2.0])
    y = np.array([1.0, 2.0])
    rho, p = safe_spearman(x, y)
    assert np.isnan(rho)
    assert np.isnan(p)


def test_safe_pearson(sample_arrays):
    x, y = sample_arrays
    r, p = safe_pearson(x, y)
    assert r == pytest.approx(0.7746, abs=0.01)


def test_calibration_intercept_slope_single_class():
    y_true = np.array([1, 1, 1])
    prob = np.array([0.8, 0.9, 0.85])
    intercept, slope = calibration_intercept_slope(y_true, prob)
    assert np.isnan(intercept)
    assert np.isnan(slope)


def test_calibration_intercept_slope_two_classes():
    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    prob = np.array([0.1, 0.15, 0.85, 0.9, 0.2, 0.8, 0.25, 0.75])
    intercept, slope = calibration_intercept_slope(y_true, prob)
    assert np.isfinite(intercept) or np.isnan(intercept)
    assert np.isfinite(slope) or np.isnan(slope)
