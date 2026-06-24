"""Target frame construction and MRI-missing filters."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from approach2.extraction import _is_missing_text, _true_dispersion_high_low
from approach2.io_atomic import atomic_write_df as _atomic_write_df


def ensure_case_id(df: pd.DataFrame) -> pd.DataFrame:
    if "case_id" not in df.columns:
        df = df.copy()
        df["case_id"] = [f"row_{i}" for i in range(len(df))]
    return df


def get_target_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = ensure_case_id(df).copy()
    out["row_index"] = list(range(len(out)))
    out["dispersion_true"] = pd.to_numeric(
        out["dispersion_invasive_DCIS_geographic"], errors="coerce"
    )
    out["dispersion_true_high_low"] = out["dispersion_true"].apply(_true_dispersion_high_low)
    out["relapse_true"] = pd.to_numeric(out["relapse"], errors="coerce")
    out = out[out["dispersion_true"].notna() & out["dispersion_true_high_low"].notna()].copy()
    out["dispersion_true_high_low"] = out["dispersion_true_high_low"].astype(int)
    out["row_index"] = out["row_index"].astype(int)
    return out




def _raw_df_with_row_index(raw_df: pd.DataFrame) -> pd.DataFrame:
    out = ensure_case_id(raw_df).copy()
    if "row_index" not in out.columns:
        out["row_index"] = list(range(len(out)))
    return out


def _mri_missing_row_indices(raw_df: pd.DataFrame) -> set:
    raw = _raw_df_with_row_index(raw_df)
    if "preop_MRI_text" not in raw.columns:
        return set()
    return set(raw.loc[raw["preop_MRI_text"].apply(_is_missing_text), "row_index"].astype(int).tolist())


def _pathology_usable_row_indices(raw_df: pd.DataFrame) -> set:
    raw = _raw_df_with_row_index(raw_df)
    if "path_report_text" not in raw.columns:
        return set()
    return set(raw.loc[~raw["path_report_text"].apply(_is_missing_text), "row_index"].astype(int).tolist())


def has_usable_mri_report(raw_df: pd.DataFrame, row_index: int) -> bool:
    """Return True when preop_MRI_text is present and non-placeholder."""
    raw = _raw_df_with_row_index(raw_df)
    if int(row_index) not in set(raw["row_index"].astype(int)):
        return False
    row = raw.loc[raw["row_index"].astype(int) == int(row_index)].iloc[0]
    return not _is_missing_text(row.get("preop_MRI_text", ""))


def has_usable_pathology_report(raw_df: pd.DataFrame, row_index: int) -> bool:
    raw = _raw_df_with_row_index(raw_df)
    if int(row_index) not in set(raw["row_index"].astype(int)):
        return False
    row = raw.loc[raw["row_index"].astype(int) == int(row_index)].iloc[0]
    return not _is_missing_text(row.get("path_report_text", ""))


def summarize_cohort_report_availability(
    raw_df: pd.DataFrame,
    target_df: pd.DataFrame,
) -> Dict[str, Any]:
    """Cohort-level counts for logging and saved summaries."""
    eligible_rows = set(target_df["row_index"].astype(int).tolist())
    mri_missing = _mri_missing_row_indices(raw_df) & eligible_rows
    path_usable = _pathology_usable_row_indices(raw_df) & eligible_rows
    mri_usable = eligible_rows - mri_missing
    return {
        "n_total_eligible_cases": len(eligible_rows),
        "n_usable_pathology": len(path_usable),
        "n_usable_mri": len(mri_usable),
        "n_missing_mri": len(mri_missing),
        "missing_mri_row_indices": sorted(mri_missing),
        "missing_mri_case_ids": sorted(
            target_df.loc[target_df["row_index"].astype(int).isin(mri_missing), "case_id"].astype(str).tolist()
        ),
    }


def filter_cases_for_modality(
    case_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    dataset_key: str,
) -> pd.DataFrame:
    """Drop cases that cannot support the requested dataset/pathway."""
    if not dataset_requires_mri_report(dataset_key) or len(case_df) == 0:
        return case_df
    missing_rows = _mri_missing_row_indices(raw_df)
    if not missing_rows or "row_index" not in case_df.columns:
        return case_df
    return case_df[~case_df["row_index"].astype(int).isin(missing_rows)].copy().reset_index(drop=True)


def dataset_requires_mri_report(dataset_key: str) -> bool:
    """Return True for datasets whose features cannot be interpreted without MRI text."""
    dataset_key = str(dataset_key)
    return (
        dataset_key == "mri"
        or dataset_key == "combined"
        or dataset_key.startswith("mri_pathcal")
        or dataset_key.startswith("mri_teacher_student")
    )


def filter_missing_mri_for_dataset(
    dataset_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    dataset_key: str,
    split_id: str,
    split_role: str = "all",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Drop MRI-missing cases from MRI-derived evaluations.

    Pathology-only evaluations intentionally keep these cases because every case is
    expected to have a pathology report. Returns filtered dataframe and stats dict.
    """
    stats: Dict[str, Any] = {
        "split_id": split_id,
        "dataset_key": dataset_key,
        "split_role": split_role,
        "n_before_filter": len(dataset_df),
        "n_skipped_missing_mri": 0,
        "n_after_filter": len(dataset_df),
        "dropped_case_ids": [],
        "drop_reason": "",
    }
    if not dataset_requires_mri_report(dataset_key) or len(dataset_df) == 0:
        stats["drop_reason"] = "not_mri_required_dataset"
        return dataset_df, stats

    missing_rows = _mri_missing_row_indices(raw_df)
    if not missing_rows or "row_index" not in dataset_df.columns:
        stats["drop_reason"] = "no_missing_mri_in_cohort"
        return dataset_df, stats

    before = len(dataset_df)
    mask = dataset_df["row_index"].astype(int).isin(missing_rows)
    dropped_ids = (
        dataset_df.loc[mask, "case_id"].astype(str).tolist()
        if "case_id" in dataset_df.columns else []
    )
    out = dataset_df[~mask].copy().reset_index(drop=True)
    skipped = before - len(out)
    stats.update({
        "n_before_filter": before,
        "n_skipped_missing_mri": skipped,
        "n_after_filter": len(out),
        "dropped_case_ids": dropped_ids,
        "drop_reason": "missing_preop_MRI_text",
    })
    if skipped > 0:
        print(
            f"[MISSING_MRI] split={split_id} role={split_role} dataset={dataset_key}: "
            f"skipped {skipped} case rows with missing preop_MRI_text "
            f"(before={before} after={len(out)})."
        )
    return out, stats


def write_mri_missing_case_summary(out_dir: str, summary_rows: Sequence[Dict[str, Any]]) -> str:
    """Write per-split/per-dataset MRI skip counts to CSV."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "mri_missing_case_summary.csv")
    if not summary_rows:
        pd.DataFrame(columns=[
            "split_id", "dataset_key", "split_role", "n_before_filter",
            "n_skipped_missing_mri", "n_after_filter", "drop_reason", "dropped_case_ids",
        ]).to_csv(path, index=False)
    else:
        rows = []
        for row in summary_rows:
            rec = dict(row)
            ids = rec.pop("dropped_case_ids", [])
            rec["dropped_case_ids"] = ";".join(map(str, ids))
            rows.append(rec)
        _atomic_write_df(pd.DataFrame(rows), path)
    print(f"[SAVE] Wrote MRI-missing filter summary: {path}")
    return path
