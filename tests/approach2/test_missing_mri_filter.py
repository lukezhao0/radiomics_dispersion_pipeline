"""MRI-missing filter behavior for Approach 2 datasets."""

from __future__ import annotations

import pandas as pd

from approach2.eval_data import (
    dataset_requires_mri_report,
    filter_missing_mri_for_dataset,
    has_usable_mri_report,
    summarize_cohort_report_availability,
)


def _raw_df() -> pd.DataFrame:
    return pd.DataFrame({
        "case_id": ["c1", "c2", "c3"],
        "preop_MRI_text": ["MRI text", "", "NA"],
        "path_report_text": ["path", "path2", "path3"],
        "index_side": ["L", "R", "L"],
        "dispersion_invasive_DCIS_geographic": [50, 60, 70],
        "relapse": [0, 1, 0],
    })


def test_dataset_requires_mri_only_for_mri_pathways() -> None:
    assert dataset_requires_mri_report("mri")
    assert dataset_requires_mri_report("combined")
    assert dataset_requires_mri_report("mri_pathcal_weighted")
    assert dataset_requires_mri_report("mri_teacher_student")
    assert not dataset_requires_mri_report("path")


def test_filter_drops_missing_mri_for_mri_not_path() -> None:
    raw = _raw_df()
    dataset_df = pd.DataFrame({
        "case_id": ["c1", "c2", "c3"],
        "row_index": [0, 1, 2],
        "dispersion_true": [50, 60, 70],
    })
    out_mri, stats_mri = filter_missing_mri_for_dataset(dataset_df, raw, "mri", "outer_split_001")
    assert len(out_mri) == 1
    assert stats_mri["n_skipped_missing_mri"] == 2
    out_path, stats_path = filter_missing_mri_for_dataset(dataset_df, raw, "path", "outer_split_001")
    assert len(out_path) == 3
    assert stats_path["n_skipped_missing_mri"] == 0


def test_has_usable_mri_report() -> None:
    raw = _raw_df()
    assert has_usable_mri_report(raw, 0)
    assert not has_usable_mri_report(raw, 1)
    assert not has_usable_mri_report(raw, 2)


def test_cohort_summary_counts() -> None:
    raw = _raw_df()
    target = pd.DataFrame({
        "case_id": ["c1", "c2", "c3"],
        "row_index": [0, 1, 2],
        "dispersion_true": [50, 60, 70],
        "dispersion_true_high_low": [0, 0, 0],
        "relapse_true": [0, 1, 0],
    })
    summary = summarize_cohort_report_availability(raw, target)
    assert summary["n_total_eligible_cases"] == 3
    assert summary["n_usable_mri"] == 1
    assert summary["n_missing_mri"] == 2
