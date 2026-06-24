"""Golden metrics tests on synthetic prediction data."""

from __future__ import annotations

import pandas as pd
import pytest

from approach1.evaluation.metrics import (
    evaluate_dispersion,
    evaluate_dispersion_high_low,
    evaluate_relapse_labels,
    prepare_predictions_for_eval,
)


def _sample_predictions_df() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "dispersion_true": 100.0,
            "dispersion_score_pred": 95.0,
            "dispersion_high_low_pred": 1,
            "relapse_true": 1,
            "relapse_pred": 1,
            "retrieval_token_exact_match": 1,
            "key_evidence": ["scattered carcinoma"],
        },
        {
            "dispersion_true": 40.0,
            "dispersion_score_pred": 45.0,
            "dispersion_high_low_pred": 0,
            "relapse_true": 0,
            "relapse_pred": 0,
            "retrieval_token_exact_match": 1,
            "key_evidence": ["single focus"],
        },
        {
            "dispersion_true": 90.0,
            "dispersion_score_pred": 88.0,
            "dispersion_high_low_pred": 1,
            "relapse_true": 1,
            "relapse_pred": 0,
            "retrieval_token_exact_match": 0,
            "key_evidence": ["multifocal residual"],
        },
    ])


def test_dispersion_regression_metrics():
    df = prepare_predictions_for_eval(_sample_predictions_df())
    _, _, metrics = evaluate_dispersion(df)
    # abs errors: |95-100|=5, |45-40|=5, |88-90|=2 => MAE = 4.0
    assert metrics["mae"] == pytest.approx(4.0, abs=0.01)
    assert metrics["rmse"] == pytest.approx(4.2426, abs=0.01)
    assert metrics["spearman_rho"] == pytest.approx(1.0, abs=0.01)


def test_dispersion_high_low_accuracy():
    df = prepare_predictions_for_eval(_sample_predictions_df())
    _, _, metrics = evaluate_dispersion_high_low(df)
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["f1"] == pytest.approx(1.0)


def test_relapse_classification_metrics():
    df = prepare_predictions_for_eval(_sample_predictions_df())
    _, _, metrics = evaluate_relapse_labels(df)
    assert metrics["accuracy"] == pytest.approx(2 / 3)
    assert "confusion_matrix" in metrics
