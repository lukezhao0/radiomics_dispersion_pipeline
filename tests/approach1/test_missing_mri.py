"""Missing MRI report handling and placeholder detection."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from approach1.data import load_cases
from approach1.splits import build_run_configs
from approach1.text_utils import has_report_text, safe_text


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, False),
        ("", False),
        ("   ", False),
        (float("nan"), False),
        (np.nan, False),
        (pd.NA, False),
        ("nan", False),
        ("N/A", False),
        ("missing", False),
        ("None", False),
        ("null", False),
        ("<NA>", False),
        ("Left breast MRI shows enhancement.", True),
    ],
)
def test_has_report_text_placeholders(value, expected: bool) -> None:
    assert has_report_text(value) is expected


def test_safe_text_normalizes_pandas_na() -> None:
    assert safe_text(pd.NA) == ""
    assert safe_text(np.nan) == ""


def _build_synthetic_csv(tmp_path, mri_by_row: dict[int, str]) -> str:
    rows = []
    for i in range(110):
        rows.append(
            {
                "case_id": f"C{i:03d}",
                "preop_MRI_text": mri_by_row.get(i, f"MRI report text for case {i}."),
                "path_report_text": f"Pathology report text for case {i}.",
                "index_side": "left",
                "dispersion_invasive_DCIS_geographic": 50.0 + (i % 50),
                "relapse": i % 2,
            }
        )
    path = tmp_path / "cases.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


def test_missing_mri_skipped_for_mri_tiers_only(tmp_path) -> None:
    missing_rows = {5: "", 10: "N/A", 15: "missing", 20: "nan", 25: "   "}
    csv_path = _build_synthetic_csv(tmp_path, missing_rows)
    df = load_cases(csv_path)
    configs = build_run_configs(df, str(tmp_path / "out"))

    for rc in configs:
        skipped = {idx for idx, _ in rc.skipped_missing_mri}
        test_rows = {idx for idx, _ in rc.test_cases_with_idxs}

        if rc.modality in {"mri_only", "mri_plus_pathology"}:
            assert missing_rows.keys() <= skipped
            assert not (missing_rows.keys() & test_rows)
        elif rc.modality == "pathology_only":
            assert not skipped
            assert missing_rows.keys() <= test_rows
