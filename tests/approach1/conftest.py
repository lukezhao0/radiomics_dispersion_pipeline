"""Shared pytest fixtures for approach1 tests."""

from __future__ import annotations

import pandas as pd
import pytest

from approach1.models import Case


@pytest.fixture
def sample_case() -> Case:
    return Case(
        case_id="SYNTH_001",
        preop_mri="Left breast MRI shows non-mass enhancement in upper outer quadrant.",
        path_report="Residual invasive ductal carcinoma with scattered single cells.",
        index_side="left",
        dispersion_true=120.0,
        relapse_true=1,
    )


@pytest.fixture
def minimal_csv(tmp_path) -> str:
    # Enough rows to cover default SHOT_SETS indices (max row index 102).
    rows = []
    for i in range(110):
        rows.append({
            "case_id": f"SYNTH_{i:03d}",
            "preop_MRI_text": f"MRI report text for case {i} left breast.",
            "path_report_text": f"Pathology report text for case {i} residual carcinoma.",
            "index_side": "left",
            "dispersion_invasive_DCIS_geographic": 50.0 + (i % 50),
            "relapse": i % 2,
        })
    path = tmp_path / "minimal_cases.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)
