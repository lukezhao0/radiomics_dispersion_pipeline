"""Regression tests for extraction pipeline wiring after modularization."""

from __future__ import annotations

import pandas as pd
import pytest

from approach2.extraction.data import make_case_from_row
from approach2.extraction.pipeline import extract_subset_records
from approach2.evaluation.plots import (
    plot_classification_comparison,
    plot_regression_correlation_comparison,
    plot_regression_error_comparison,
)


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "case_id": ["c1"],
            "preop_MRI_text": ["MRI shows mass in left breast."],
            "path_report_text": ["Invasive ductal carcinoma, multifocal."],
            "index_side": ["left"],
            "dispersion_invasive_DCIS_geographic": [92.0],
            "relapse": [0],
        }
    )


def test_make_case_from_row_uses_safe_text() -> None:
    case = make_case_from_row(_sample_df(), 0)
    assert case.case_id == "c1"
    assert "MRI" in case.preop_mri
    assert "multifocal" in case.path_report
    assert case.dispersion_true == 92.0
    assert case.relapse_true == 0


def test_make_case_from_row_handles_nan_text() -> None:
    df = _sample_df()
    df.loc[0, "preop_MRI_text"] = float("nan")
    case = make_case_from_row(df, 0)
    assert case.preop_mri == ""


def test_extract_subset_records_missing_text_no_api_call(tmp_path, monkeypatch) -> None:
    df = _sample_df()
    df.loc[0, "path_report_text"] = float("nan")

    def _fail_if_called(*_args, **_kwargs):
        raise AssertionError("API extraction should not run for missing report text")

    monkeypatch.setattr("approach2.extraction.pipeline.extract_case_features", _fail_if_called)

    records = extract_subset_records(
        df=df,
        row_indices=[0],
        report_mode="path",
        split_id="test_split",
        split_role="train",
        checkpoint_dir=str(tmp_path),
        max_workers=1,
        resume=False,
        force_reextract=False,
    )
    assert len(records) == 1
    assert records[0]["selected_report_missing"] == 1


def test_plot_functions_tolerate_empty_metrics_df(tmp_path) -> None:
    empty = pd.DataFrame()
    out_png = str(tmp_path / "plot.png")
    plot_classification_comparison(empty, out_png)
    plot_regression_error_comparison(empty, out_png)
    plot_regression_correlation_comparison(empty, out_png)
    assert not (tmp_path / "plot.png").exists()
