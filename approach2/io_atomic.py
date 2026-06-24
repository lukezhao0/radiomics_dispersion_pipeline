"""Atomic CSV writes and safe checkpoint reads for approach2."""

from __future__ import annotations

import os

import pandas as pd


def atomic_write_df(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def safe_read_csv_if_exists(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"[RESUME] Failed to read checkpoint table {path}: {e}")
        return pd.DataFrame()
