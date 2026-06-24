"""Extraction orchestration, checkpoints, and batch record I/O."""

from __future__ import annotations

import json
import os
import re
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..api.client import call_securegpt_chat
from ..api.cost import (
    GLOBAL_API_SEMAPHORE,
    configure_global_api_concurrency,
    get_global_api_concurrency,
)
from ..extraction.config import MAX_TOKENS, RATE_LIMIT_SLEEP_S, REPORT_CONFIG
from ..extraction.data import (
    Case,
    _selected_report_field,
    _selected_report_label,
    _selected_report_text,
    load_cases,
    make_case_from_row,
    make_missing_extraction_record,
)
from ..prompts.builder import build_user_prompt
from ..extraction.schema import (
    _extract_json_from_text,
    _normalize_extraction_obj,
    _sanitize_extraction_obj_for_validation,
    _validate_extraction_obj,
)

def extract_case_features(test_case: Case, report_mode: str) -> Dict[str, Any]:
    user_prompt = build_user_prompt(test_case, report_mode)
    selected_text = _selected_report_text(test_case, report_mode)
    last_err: Optional[str] = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(
                f"[EXTRACT] case_id={test_case.case_id} attempt={attempt}/{MAX_RETRIES} "
                f"report_mode={report_mode}"
            )
            raw = call_securegpt_chat(user_prompt)
            obj = _extract_json_from_text(raw)
            obj = _normalize_extraction_obj(obj, report_mode=report_mode)
            obj = _sanitize_extraction_obj_for_validation(
                obj=obj,
                report_text=selected_text,
                report_mode=report_mode,
            )

            warnings = obj.get("validation_warnings", []) or []
            if warnings:
                print(
                    f"[VALIDATION] case_id={test_case.case_id} "
                    f"n_warnings={len(warnings)} "
                    f"n_repaired={obj.get('n_repaired_phrase_quotes', 0)} "
                    f"n_dropped={obj.get('n_dropped_phrase_items', 0)}"
                )
                for w in warnings[:8]:
                    print(f"[VALIDATION] {w}")
                if len(warnings) > 8:
                    print(f"[VALIDATION] ... {len(warnings) - 8} additional warnings omitted from console")

            ok, msg = _validate_extraction_obj(
                obj,
                expected_case_id=test_case.case_id,
                report_text=selected_text,
                report_mode=report_mode,
            )
            if not ok:
                raise ValueError(f"Validation failed after sanitization: {msg}. Raw head: {raw[:500]}")

            print(f"[EXTRACT] case_id={test_case.case_id} VALID JSON received.")
            return obj

        except Exception as e:
            last_err = str(e)
            print(f"[RETRY] case_id={test_case.case_id} attempt={attempt}/{MAX_RETRIES} error={last_err}")
            sleep_s = BACKOFF_BASE_S ** (attempt - 1)
            print(f"[RETRY] Sleeping for {sleep_s:.2f}s before retry...")
            time.sleep(sleep_s)

    raise RuntimeError(
        f"Failed to get valid extraction for case_id={test_case.case_id}. Last error: {last_err}"
    )

# -----------------------------
# Main pipeline helpers
# -----------------------------

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
    return Case(
        case_id=str(row["case_id"]),
        preop_mri=_safe_text(row["preop_MRI_text"]),
        path_report=_safe_text(row["path_report_text"]),
        index_side=_safe_text(row["index_side"]),
        dispersion_true=float(row["dispersion_invasive_DCIS_geographic"])
        if pd.notna(row["dispersion_invasive_DCIS_geographic"])
        else None,
        relapse_true=int(row["relapse"]) if pd.notna(row["relapse"]) else None,
    )


def make_missing_extraction_record(
    test_case: Case,
    row_index: int,
    report_mode: str,
    split_id: Optional[str] = None,
    split_role: Optional[str] = None,
) -> Dict[str, Any]:
    selected_field = _selected_report_field(report_mode)
    selected_text = _selected_report_text(test_case, report_mode)
    return {
        "case_id": test_case.case_id,
        "row_index": row_index,
        "report_mode": report_mode,
        "selected_report_field": selected_field,
        "selected_report_text": selected_text,
        "selected_report_missing": 1,
        "selected_report_missing_reason": f"Missing {selected_field}; no extraction generated.",
        "has_preop_mri": int(not _is_missing_text(test_case.preop_mri)),
        "has_path_report": int(not _is_missing_text(test_case.path_report)),
        "dispersion_true": test_case.dispersion_true,
        "dispersion_true_high_low": _true_dispersion_high_low(test_case.dispersion_true),
        "relapse_true": test_case.relapse_true,
        "outer_split_id": split_id or "",
        "outer_split_role": split_role or "",
        "seed_aligned_phrases": [],
        "denovo_candidate_phrases": [],
        "quantitative_attributes": {
            "extent_cm": None,
            "largest_focus_cm": None,
            "margin_distance_mm": None,
            "lvi_present": None,
            "dcis_burden": None,
            "nme_present": None,
            "satellite_lesions_present": None,
            "multifocal_present": None,
            "multicentric_present": None,
            "residual_disease_minimal": None,
            "single_localized_residual": None,
            "diffuse_scattered_residual": None,
        },
        "report_level_summary": {
            "distribution_pattern": "unknown",
            "distribution_evidence_quote": "",
            "localization_vs_scatter_note": "Selected modality missing; no lexical extraction performed.",
        },
    }


def summarize_run(extractions: List[Dict[str, Any]], report_mode: str, split_id: Optional[str] = None) -> str:
    total = len(extractions)
    missing = sum(int(bool(x.get("selected_report_missing", 0))) for x in extractions)
    used = total - missing
    n_seed = sum(len(x.get("seed_aligned_phrases", [])) for x in extractions)
    n_denovo = sum(len(x.get("denovo_candidate_phrases", [])) for x in extractions)

    lines = []
    lines.append("=== Lexical Feature Discovery Run Summary ===")
    lines.append(f"report_mode = {report_mode}")
    if split_id:
        lines.append(f"outer_split_id = {split_id}")
    lines.append(f"N_total_rows = {total}")
    lines.append(f"N_selected_report_missing = {missing}")
    lines.append(f"N_rows_extracted = {used}")
    lines.append(f"N_seed_aligned_phrases_total = {n_seed}")
    lines.append(f"N_denovo_candidate_phrases_total = {n_denovo}")
    return "\n".join(lines)


def _extract_single_subset_record(
    df: pd.DataFrame,
    row_index: int,
    report_mode: str,
    split_id: Optional[str],
    split_role: Optional[str],
    sleep_between_calls_s: float,
    position: int,
    total: int,
) -> Dict[str, Any]:
    test_case = make_case_from_row(df, row_index)
    selected_text = _selected_report_text(test_case, report_mode)
    selected_field = _selected_report_field(report_mode)

    print('-' * 80)
    print(
        f"[CASE] {position}/{total} case_id={test_case.case_id} "
        f"row_index={row_index} report_mode={report_mode}"
    )
    print(
        f"[CASE] selected_report_field={selected_field} "
        f"selected_chars={len(selected_text)} split_id={split_id or 'NA'} split_role={split_role or 'NA'}"
    )

    if _is_missing_text(selected_text):
        rec = make_missing_extraction_record(
            test_case=test_case,
            row_index=row_index,
            report_mode=report_mode,
            split_id=split_id,
            split_role=split_role,
        )
        print('[CASE] Selected report missing; wrote placeholder row.')
        return rec

    obj = extract_case_features(test_case, report_mode)

    rec = {
        'case_id': test_case.case_id,
        'row_index': row_index,
        'report_mode': report_mode,
        'selected_report_field': obj['selected_report_field'],
        'selected_report_text': selected_text,
        'selected_report_missing': 0,
        'selected_report_missing_reason': '',
        'has_preop_mri': int(not _is_missing_text(test_case.preop_mri)),
        'has_path_report': int(not _is_missing_text(test_case.path_report)),
        'dispersion_true': test_case.dispersion_true,
        'dispersion_true_high_low': _true_dispersion_high_low(test_case.dispersion_true),
        'relapse_true': test_case.relapse_true,
        'outer_split_id': split_id or '',
        'outer_split_role': split_role or '',
        "validation_warnings": [],
        "n_validation_warnings": 0,
        "n_repaired_phrase_quotes": 0,
        "n_dropped_phrase_items": 0,
        'seed_aligned_phrases': obj['seed_aligned_phrases'],
        'denovo_candidate_phrases': obj['denovo_candidate_phrases'],
        'quantitative_attributes': obj['quantitative_attributes'],
        'report_level_summary': obj['report_level_summary'],
    }

    print(
        '[CASE] Extraction OK: '
        f"n_seed={len(rec['seed_aligned_phrases'])} "
        f"n_denovo={len(rec['denovo_candidate_phrases'])}"
    )
    time.sleep(sleep_between_calls_s)
    return rec


def _resolve_default_api_workers(requested_workers: Optional[int]) -> int:
    if requested_workers is not None:
        return max(1, int(requested_workers))
    cpu_count = os.cpu_count() or 1
    if cpu_count >= 8:
        return 2
    return 1

def _safe_filename_component(s: Any, max_len: int = 100) -> str:
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s or "NA")).strip("_")
    return (s or "NA")[:max_len]


def _checkpoint_root_dir(
    checkpoint_dir: Optional[str],
    report_mode: str,
    split_id: Optional[str],
    split_role: Optional[str],
) -> Optional[str]:
    if not checkpoint_dir:
        return None
    root = os.path.join(
        checkpoint_dir,
        "_case_checkpoints",
        _safe_filename_component(report_mode),
        _safe_filename_component(split_id or "standalone"),
        _safe_filename_component(split_role or "subset"),
    )
    os.makedirs(root, exist_ok=True)
    return root


def _case_checkpoint_path(checkpoint_root: str, row_index: int, case_id: str, report_mode: str) -> str:
    fname = (
        f"row_{int(row_index):06d}__"
        f"case_{_safe_filename_component(case_id)}__"
        f"mode_{_safe_filename_component(report_mode)}.json"
    )
    return os.path.join(checkpoint_root, fname)


def _write_json_atomic(obj: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, path)


def _load_json_record(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else None
    except Exception as e:
        print(f"[CHECKPOINT] Could not load checkpoint {path}: {e}")
        return None


def _record_matches_request(
    rec: Dict[str, Any],
    row_index: int,
    case_id: str,
    report_mode: str,
    split_id: Optional[str],
    split_role: Optional[str],
) -> bool:
    return (
        str(rec.get("case_id")) == str(case_id)
        and int(rec.get("row_index", -999999)) == int(row_index)
        and str(rec.get("report_mode")) == str(report_mode)
        and str(rec.get("outer_split_id", "")) == str(split_id or "")
        and str(rec.get("outer_split_role", "")) == str(split_role or "")
    )


def _extract_single_subset_record_with_checkpoint(
    df: pd.DataFrame,
    row_index: int,
    report_mode: str,
    split_id: Optional[str],
    split_role: Optional[str],
    sleep_between_calls_s: float,
    position: int,
    total: int,
    checkpoint_root: Optional[str],
    resume: bool,
    force_reextract: bool,
) -> Dict[str, Any]:
    test_case = make_case_from_row(df, row_index)
    ckpt_path = (
        _case_checkpoint_path(checkpoint_root, row_index, test_case.case_id, report_mode)
        if checkpoint_root else None
    )

    if ckpt_path and resume and not force_reextract and os.path.exists(ckpt_path):
        rec = _load_json_record(ckpt_path)
        if rec is not None and _record_matches_request(rec, row_index, test_case.case_id, report_mode, split_id, split_role):
            print(
                f"[CHECKPOINT] Reusing cached extraction: case_id={test_case.case_id} "
                f"row_index={row_index} report_mode={report_mode} path={ckpt_path}"
            )
            rec["checkpoint_status"] = "loaded"
            return rec
        print(f"[CHECKPOINT] Existing checkpoint did not match request; re-extracting: {ckpt_path}")

    try:
        rec = _extract_single_subset_record(
            df=df,
            row_index=row_index,
            report_mode=report_mode,
            split_id=split_id,
            split_role=split_role,
            sleep_between_calls_s=sleep_between_calls_s,
            position=position,
            total=total,
        )
        rec["checkpoint_status"] = "fresh"
        rec["checkpoint_written_at"] = datetime.now().isoformat(timespec="seconds")
        if ckpt_path:
            _write_json_atomic(rec, ckpt_path)
            print(f"[CHECKPOINT] Wrote case checkpoint: {ckpt_path}")
        return rec
    except Exception:
        print(
            f"[CHECKPOINT] Extraction failed before checkpoint could be written for "
            f"case_id={test_case.case_id} row_index={row_index} report_mode={report_mode}"
        )
        print(traceback.format_exc())
        raise

def extract_subset_records(
    df: pd.DataFrame,
    row_indices: Sequence[int],
    report_mode: str,
    split_id: Optional[str] = None,
    split_role: Optional[str] = None,
    sleep_between_calls_s: float = RATE_LIMIT_SLEEP_S,
    max_workers: int = 1,
    checkpoint_dir: Optional[str] = None,
    resume: bool = True,
    force_reextract: bool = False,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    row_indices = [int(x) for x in row_indices]
    max_workers = max(1, int(max_workers))
    checkpoint_root = _checkpoint_root_dir(
        checkpoint_dir=checkpoint_dir,
        report_mode=report_mode,
        split_id=split_id,
        split_role=split_role,
    )

    print(
        f"[RUN] Beginning extraction over subset. "
        f"report_mode={report_mode} split_id={split_id or 'NA'} split_role={split_role or 'NA'} "
        f"n_rows={len(row_indices)} max_api_workers={max_workers} "
        f"global_api_cap={get_global_api_concurrency()} "
        f"resume={resume} force_reextract={force_reextract} "
        f"checkpoint_root={checkpoint_root or 'disabled'}"
    )

    if max_workers == 1 or len(row_indices) <= 1:
        for n, idx in enumerate(row_indices, 1):
            rec = _extract_single_subset_record_with_checkpoint(
                df=df,
                row_index=idx,
                report_mode=report_mode,
                split_id=split_id,
                split_role=split_role,
                sleep_between_calls_s=sleep_between_calls_s,
                position=n,
                total=len(row_indices),
                checkpoint_root=checkpoint_root,
                resume=resume,
                force_reextract=force_reextract,
            )
            records.append(rec)
        return records

    print(
        f"[RUN] Parallel API extraction enabled for report_mode={report_mode} "
        f"with max_workers={max_workers}. Logs may interleave across cases."
    )
    indexed_records: Dict[int, Dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_position = {
            executor.submit(
                _extract_single_subset_record_with_checkpoint,
                df,
                idx,
                report_mode,
                split_id,
                split_role,
                sleep_between_calls_s,
                position,
                len(row_indices),
                checkpoint_root,
                resume,
                force_reextract,
            ): position - 1
            for position, idx in enumerate(row_indices, 1)
        }
        for future in as_completed(future_to_position):
            pos = future_to_position[future]
            indexed_records[pos] = future.result()

    records = [indexed_records[i] for i in sorted(indexed_records)]
    return records


def write_extractions(
    extractions: List[Dict[str, Any]],
    out_dir: str,
    report_mode: str,
    filename_prefix: Optional[str] = None,
    split_id: Optional[str] = None,
) -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    prefix = f"{filename_prefix}_" if filename_prefix else ""

    extractions_csv = os.path.join(out_dir, f"{prefix}case_phrase_extractions_{report_mode}.csv")
    extractions_jsonl = os.path.join(out_dir, f"{prefix}case_phrase_extractions_{report_mode}.jsonl")
    summary_txt = os.path.join(out_dir, f"{prefix}run_summary_{report_mode}.txt")

    # with open(extractions_jsonl, "w", encoding="utf-8") as f_jsonl:
    #     for rec in extractions:
    #         f_jsonl.write(json.dumps(rec, ensure_ascii=False) + "\n")

    out_df = pd.DataFrame(extractions)
    if len(out_df) == 0:
        out_df = pd.DataFrame(columns=[
            "case_id",
            "row_index",
            "report_mode",
            "selected_report_field",
            "selected_report_text",
            "selected_report_missing",
            "selected_report_missing_reason",
            "has_preop_mri",
            "has_path_report",
            "dispersion_true",
            "dispersion_true_high_low",
            "relapse_true",
            "outer_split_id",
            "outer_split_role",
            "seed_aligned_phrases",
            "denovo_candidate_phrases",
            "quantitative_attributes",
            "report_level_summary",
        ])

    for col in [
        "seed_aligned_phrases",
        "denovo_candidate_phrases",
        "quantitative_attributes",
        "report_level_summary",
    ]:
        if col in out_df.columns:
            out_df[col] = out_df[col].apply(lambda x: json.dumps(x, ensure_ascii=False))

    tmp_csv = f"{extractions_csv}.tmp.{os.getpid()}"
    out_df.to_csv(tmp_csv, index=False)
    os.replace(tmp_csv, extractions_csv)

    summary = summarize_run(extractions, report_mode, split_id=split_id)
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write(summary)

    tmp_jsonl = f"{extractions_jsonl}.tmp.{os.getpid()}"
    with open(tmp_jsonl, "w", encoding="utf-8") as f_jsonl:
        for rec in extractions:
            f_jsonl.write(json.dumps(rec, ensure_ascii=False) + "\n")
    os.replace(tmp_jsonl, extractions_jsonl)

    print(f"[SAVE] Wrote JSONL: {extractions_jsonl}")
    print(f"[SAVE] Wrote CSV:   {extractions_csv}")
    print(f"[SAVE] Wrote summary: {summary_txt}")

    return {
        "csv": extractions_csv,
        "jsonl": extractions_jsonl,
        "summary": summary_txt,
    }


# -----------------------------
# Main pipeline
# -----------------------------

def run_pipeline(
    csv_path: str,
    out_dir: str,
    report_mode: str,
    resume: bool = True,
    force_reextract: bool = False,
    row_indices: Optional[Sequence[int]] = None,
    split_id: Optional[str] = None,
    split_role: Optional[str] = None,
    filename_prefix: Optional[str] = None,
    max_api_workers: Optional[int] = None,
    assume_yes: bool = False,
) -> Dict[str, str]:
    print("=" * 80)
    print("[START] SecureGPT lexical feature discovery pipeline")
    print(f"[START] CSV_PATH={csv_path}")
    print(f"[START] OUT_DIR={out_dir}")
    print(f"[START] REPORT_MODE={report_mode}")
    if split_id:
        print(f"[START] SPLIT_ID={split_id}")
    if split_role:
        print(f"[START] SPLIT_ROLE={split_role}")
    print("=" * 80)

    os.makedirs(out_dir, exist_ok=True)
    max_api_workers = _resolve_default_api_workers(max_api_workers)
    configure_global_api_concurrency(max_api_workers)
    print(f"[START] MAX_API_WORKERS={max_api_workers}")
    df = load_cases(csv_path)

    if row_indices is None:
        row_indices = list(range(len(df)))
    else:
        row_indices = [int(x) for x in row_indices]

    prompt_counts: List[int] = []
    prompt_modes: List[str] = []
    for idx in row_indices:
        case = make_case_from_row(df, int(idx))
        if _is_missing_text(_selected_report_text(case, report_mode)):
            continue
        prompt_counts.append(estimate_prompt_tokens_for_case(case, report_mode))
        prompt_modes.append(report_mode)
    estimate = summarize_apriori_cost_estimate(prompt_counts, prompt_modes, max_completion_tokens=MAX_TOKENS)
    print_apriori_cost_estimate_report(estimate, label=f"standalone {report_mode} extraction")
    confirm_cost_estimate_or_exit(estimate, assume_yes=assume_yes)

    preflight_check()

    print(f"[RUN] Total cases to iterate = {len(row_indices)}")
    print(f"[RUN] Writing outputs to: {out_dir}")

    t_run0 = time.time()
    extractions = extract_subset_records(
        df=df,
        row_indices=row_indices,
        report_mode=report_mode,
        split_id=split_id,
        split_role=split_role,
        sleep_between_calls_s=RATE_LIMIT_SLEEP_S,
        max_workers=max_api_workers,
        checkpoint_dir=out_dir,
        resume=resume,
        force_reextract=force_reextract,
    )
    paths = write_extractions(
        extractions=extractions,
        out_dir=out_dir,
        report_mode=report_mode,
        filename_prefix=filename_prefix,
        split_id=split_id,
    )

    dt_run = time.time() - t_run0
    print("=" * 80)
    print(f"[DONE] Extraction complete in {dt_run/60:.2f} minutes.")
    print("=" * 80)
    print("\n" + summarize_run(extractions, report_mode, split_id=split_id))
    print_cumulative_report()
    write_cost_tracker_json(out_dir)
    print("[END] All done.")
    return paths


def _load_row_indices_json(path: Optional[str]) -> Optional[List[int]]:
    if path is None:
        return None
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, list):
        raise ValueError("--row-indices-json must contain a JSON list of integer row indices.")
    return [int(x) for x in obj]
