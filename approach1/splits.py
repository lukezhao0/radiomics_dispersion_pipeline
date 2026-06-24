"""Train/test split planning and RunConfig construction."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Tuple

import pandas as pd

from .api.cost import estimate_apriori_pipeline_cost
from .config import MODALITY_TIERS, SHOT_SETS
from .data import make_case_from_row
from .models import Case, RunConfig
from .prompts.templates import build_training_block
from .text_utils import has_report_text, modality_requires_mri, modality_uses_pathology


def validate_shot_rows(df: pd.DataFrame, shotset: Dict[str, Any]) -> None:
    rows = list(shotset["high_rows"]) + list(shotset["low_rows"])
    if len(rows) != len(set(rows)):
        raise ValueError(f"Duplicate training rows in {shotset['name']}: {rows}")
    if min(rows) < 0 or max(rows) >= len(df):
        raise ValueError(
            f"Training rows for {shotset['name']} out of range for dataframe length {len(df)}: {rows}. "
            "Row numbering is 0-based pandas iloc."
        )
    if len(shotset["high_rows"]) != 2 or len(shotset["low_rows"]) != 2:
        raise ValueError(f"Each shot set must have exactly 2 high and 2 low rows. Got: {shotset}")


def validate_training_modality_availability(training_cases: List[Tuple[int, Case]], modality: str) -> None:
    bad: List[str] = []
    for idx, c in training_cases:
        if modality_requires_mri(modality) and not has_report_text(c.preop_mri):
            bad.append(f"row {idx} case_id={c.case_id} missing MRI")
        if modality_uses_pathology(modality) and not has_report_text(c.path_report):
            bad.append(f"row {idx} case_id={c.case_id} missing pathology")
    if bad:
        raise ValueError(
            f"Cannot build {modality} training block because required reports are missing:\n"
            + "\n".join(f"  - {b}" for b in bad)
        )


def build_run_configs(df: pd.DataFrame, root_out_dir: str) -> List[RunConfig]:
    run_configs: List[RunConfig] = []
    all_idxs = list(range(len(df)))

    for shotset in SHOT_SETS:
        validate_shot_rows(df, shotset)
        high_rows = list(shotset["high_rows"])
        low_rows = list(shotset["low_rows"])
        training_rows = high_rows + low_rows
        training_cases_with_idxs = [(idx, make_case_from_row(df, idx)) for idx in training_rows]

        for modality in MODALITY_TIERS:
            validate_training_modality_availability(training_cases_with_idxs, modality)
            training_cases = [c for _, c in training_cases_with_idxs]
            training_block = build_training_block(training_cases, modality)

            test_idxs_all = [i for i in all_idxs if i not in set(training_rows)]
            test_cases_with_idxs: List[Tuple[int, Case]] = []
            skipped_missing_mri: List[Tuple[int, Case]] = []
            for idx in test_idxs_all:
                c = make_case_from_row(df, idx)
                if modality_requires_mri(modality) and not has_report_text(c.preop_mri):
                    skipped_missing_mri.append((idx, c))
                    continue
                test_cases_with_idxs.append((idx, c))

            run_out_dir = os.path.join(root_out_dir, shotset["name"], modality)
            apriori_cost = estimate_apriori_pipeline_cost(training_block, test_cases_with_idxs, modality)
            run_configs.append(
                RunConfig(
                    shotset_name=shotset["name"],
                    high_rows=high_rows,
                    low_rows=low_rows,
                    training_rows=training_rows,
                    modality=modality,
                    run_out_dir=run_out_dir,
                    training_block=training_block,
                    test_cases_with_idxs=test_cases_with_idxs,
                    skipped_missing_mri=skipped_missing_mri,
                    apriori_cost=apriori_cost,
                )
            )
    return run_configs


def write_run_config(rc: RunConfig) -> None:
    os.makedirs(rc.run_out_dir, exist_ok=True)
    path = os.path.join(rc.run_out_dir, "run_config.json")
    payload = {
        "shotset_name": rc.shotset_name,
        "high_rows": rc.high_rows,
        "low_rows": rc.low_rows,
        "training_rows": rc.training_rows,
        "modality": rc.modality,
        "n_test_cases": len(rc.test_cases_with_idxs),
        "n_skipped_missing_mri": len(rc.skipped_missing_mri),
        "skipped_missing_mri_rows": [idx for idx, _ in rc.skipped_missing_mri],
        "apriori_cost": rc.apriori_cost,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def write_skipped_cases(rc: RunConfig) -> None:
    if not rc.skipped_missing_mri:
        return
    rows = []
    for idx, c in rc.skipped_missing_mri:
        rows.append({
            "row_index": idx,
            "case_id": c.case_id,
            "index_side": c.index_side,
            "skip_reason": "missing_preop_MRI_text_required_for_this_tier",
            "has_preop_mri": has_report_text(c.preop_mri),
            "has_path_report": has_report_text(c.path_report),
        })
    pd.DataFrame(rows).to_csv(os.path.join(rc.run_out_dir, "skipped_cases_missing_mri.csv"), index=False)


def empty_predictions_df() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "shotset_name",
        "modality",
        "case_id",
        "row_index",
        "index_side",
        "has_preop_mri",
        "has_path_report",
        "dispersion_true",
        "dispersion_true_high_low",
        "relapse_true",
        "dispersion_score_pred",
        "dispersion_high_low_pred",
        "relapse_pred",
        "key_evidence",
        "retrieval_token_expected",
        "retrieval_check_token_returned",
        "retrieval_check_correct_reported",
        "retrieval_token_exact_match",
        "reasoning_summary",
        "structured_rationale",
    ])
