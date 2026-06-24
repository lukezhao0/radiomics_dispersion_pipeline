"""Prediction JSONL/CSV persistence and loading."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Tuple

import pandas as pd

from ..io_atomic import atomic_write_text
from ..models import Case, RunConfig
from ..schema.records import normalize_pred_record, validate_saved_pred_record


def ingest_saved_prediction(
    raw: Dict[str, Any],
    rc: RunConfig,
    test_by_idx: Dict[int, Case],
    source: str,
    line_ref: str,
    by_row: Dict[int, Dict[str, Any]],
    warnings: List[str],
) -> None:
    try:
        idx = int(raw.get("row_index"))
    except (TypeError, ValueError):
        warnings.append(f"{source} {line_ref}: missing/invalid row_index")
        return
    if idx not in test_by_idx:
        warnings.append(f"{source} {line_ref}: row_index={idx} not in current test set; ignored")
        return
    test_case = test_by_idx[idx]
    ok, msg = validate_saved_pred_record(raw, rc, idx, test_case)
    if not ok:
        warnings.append(f"{source} {line_ref}: row_index={idx} invalid: {msg}")
        return
    if idx in by_row:
        warnings.append(f"{source} {line_ref}: duplicate row_index={idx}; keeping latest valid record")
    by_row[idx] = normalize_pred_record(raw)


def load_predictions_from_jsonl(
    path: str,
    rc: RunConfig,
    test_cases_with_idxs: List[Tuple[int, Case]],
) -> Tuple[Dict[int, Dict[str, Any]], List[str]]:
    by_row: Dict[int, Dict[str, Any]] = {}
    warnings: List[str] = []
    if not os.path.isfile(path):
        return by_row, warnings

    test_by_idx = {idx: c for idx, c in test_cases_with_idxs}
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                warnings.append(f"{path}:{line_no}: invalid JSON: {exc}")
                continue
            if not isinstance(raw, dict):
                warnings.append(f"{path}:{line_no}: record is not a JSON object")
                continue
            ingest_saved_prediction(raw, rc, test_by_idx, path, f"line {line_no}", by_row, warnings)
    return by_row, warnings


def load_predictions_from_csv(
    path: str,
    rc: RunConfig,
    test_cases_with_idxs: List[Tuple[int, Case]],
) -> Tuple[Dict[int, Dict[str, Any]], List[str]]:
    by_row: Dict[int, Dict[str, Any]] = {}
    warnings: List[str] = []
    if not os.path.isfile(path):
        return by_row, warnings

    test_by_idx = {idx: c for idx, c in test_cases_with_idxs}
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        warnings.append(f"{path}: failed to read CSV: {exc}")
        return by_row, warnings

    for row_no, row in df.iterrows():
        raw = row.to_dict()
        ingest_saved_prediction(raw, rc, test_by_idx, path, f"csv row {row_no}", by_row, warnings)
    return by_row, warnings


def load_existing_case_predictions(
    run_out_dir: str,
    rc: RunConfig,
) -> Tuple[Dict[int, Dict[str, Any]], str, List[str]]:
    jsonl_path = os.path.join(run_out_dir, "predictions_testing_cases.jsonl")
    csv_path = os.path.join(run_out_dir, "predictions_testing_cases.csv")
    warnings: List[str] = []

    by_row, jsonl_warnings = load_predictions_from_jsonl(jsonl_path, rc, rc.test_cases_with_idxs)
    warnings.extend(jsonl_warnings)
    source = "jsonl" if by_row else ""

    if len(by_row) < len(rc.test_cases_with_idxs):
        csv_by_row, csv_warnings = load_predictions_from_csv(csv_path, rc, rc.test_cases_with_idxs)
        warnings.extend(csv_warnings)
        for idx, rec in csv_by_row.items():
            if idx not in by_row:
                by_row[idx] = rec
        if csv_by_row:
            source = "jsonl+csv" if source else "csv"

    return by_row, source, warnings


def predictions_dict_to_dataframe(
    by_row: Dict[int, Dict[str, Any]],
    rc: RunConfig,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for idx, _ in rc.test_cases_with_idxs:
        if idx not in by_row:
            raise KeyError(f"Missing prediction for row_index={idx}")
        rows.append(by_row[idx])
    return pd.DataFrame(rows)


def write_predictions_jsonl(path: str, records: List[Dict[str, Any]]) -> None:
    text = "\n".join(json.dumps(r, ensure_ascii=False) for r in records)
    if text:
        text += "\n"
    atomic_write_text(path, text)


def append_prediction_jsonl(path: str, record: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_predictions_csv(path: str, pred_df: pd.DataFrame) -> None:
    pred_df_for_csv = pred_df.copy()
    if len(pred_df_for_csv):
        pred_df_for_csv["key_evidence"] = pred_df_for_csv["key_evidence"].apply(
            lambda x: json.dumps(x, ensure_ascii=False)
        )
        pred_df_for_csv["structured_rationale"] = pred_df_for_csv["structured_rationale"].apply(
            lambda x: json.dumps(x, ensure_ascii=False)
        )
    tmp = f"{path}.tmp.{os.getpid()}"
    pred_df_for_csv.to_csv(tmp, index=False)
    os.replace(tmp, path)
