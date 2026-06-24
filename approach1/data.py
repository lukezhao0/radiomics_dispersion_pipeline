"""CSV loading and Case construction."""

from __future__ import annotations

import pandas as pd

from .models import Case
from .text_utils import safe_text


def load_cases(csv_path: str) -> pd.DataFrame:
    print(f"[DATA] Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)

    required_cols = {
        "preop_MRI_text",
        "path_report_text",
        "index_side",
        "dispersion_invasive_DCIS_geographic",
        "relapse",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")

    if "case_id" not in df.columns:
        df = df.copy()
        df["case_id"] = [f"row_{i}" for i in range(len(df))]

    print(f"[DATA] Loaded dataframe: rows={len(df)} cols={len(df.columns)}")
    return df


def make_case_from_row(df: pd.DataFrame, idx: int) -> Case:
    row = df.iloc[idx]
    dispersion = row["dispersion_invasive_DCIS_geographic"]
    relapse = row["relapse"]
    return Case(
        case_id=str(row["case_id"]),
        preop_mri=safe_text(row["preop_MRI_text"]),
        path_report=safe_text(row["path_report_text"]),
        index_side=safe_text(row["index_side"]),
        dispersion_true=float(dispersion) if pd.notna(dispersion) else None,
        relapse_true=int(relapse) if pd.notna(relapse) else None,
    )
