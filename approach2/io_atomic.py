"""Atomic CSV writes and safe checkpoint reads for approach2."""

from __future__ import annotations

import json
import os

import pandas as pd


def atomic_write_df(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def atomic_write_json(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def is_valid_json_file(path: str, min_size: int = 2) -> bool:
    """Return True when path exists and contains parseable non-empty JSON."""
    if not os.path.exists(path):
        return False
    try:
        if os.path.getsize(path) < min_size:
            return False
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj is not None
    except Exception:
        return False


def safe_read_csv_if_exists(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"[RESUME] Failed to read checkpoint table {path}: {e}")
        return pd.DataFrame()
