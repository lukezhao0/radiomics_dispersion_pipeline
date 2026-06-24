"""Split checkpoint fingerprinting and artifact validation for resume safety."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from approach2.splits import case_id_list_hash

RESUME_SCRIPT_VERSION = 1

_REQUIRED_MARKER_KEYS = (
    "split_id",
    "csv_path",
    "outer_scheme",
    "outer_repeats",
    "outer_test_frac",
    "outer_folds",
    "random_seed",
)


def build_split_resume_fingerprint(args: Any, split_id: str) -> Dict[str, Any]:
    """Fingerprint fields that must match before reusing a completed split checkpoint."""
    return {
        "resume_script_version": RESUME_SCRIPT_VERSION,
        "split_id": str(split_id),
        "csv_path": str(getattr(args, "csv_path", "")),
        "outer_scheme": str(getattr(args, "outer_scheme", "")),
        "outer_repeats": int(getattr(args, "outer_repeats", 0) or 0),
        "outer_test_frac": float(getattr(args, "outer_test_frac", 0.0) or 0.0),
        "outer_folds": int(getattr(args, "outer_folds", 0) or 0),
        "random_seed": int(getattr(args, "random_seed", 0) or 0),
        "stability_threshold": float(getattr(args, "stability_threshold", 0.0) or 0.0),
        "target_stable_features_per_modality": int(
            getattr(args, "target_stable_features_per_modality", 0) or 0
        ),
        "modalities": sorted(getattr(args, "modalities", []) or []),
        "representations": sorted(getattr(args, "representations", []) or []),
    }


def split_resume_fingerprints_compatible(
    saved: Dict[str, Any],
    current: Dict[str, Any],
) -> Tuple[bool, str]:
    keys = [
        "resume_script_version",
        "split_id",
        "csv_path",
        "outer_scheme",
        "outer_repeats",
        "outer_test_frac",
        "outer_folds",
        "random_seed",
        "stability_threshold",
        "target_stable_features_per_modality",
        "modalities",
        "representations",
    ]
    for key in keys:
        if saved.get(key) != current.get(key):
            return False, f"{key}: saved={saved.get(key)!r} current={current.get(key)!r}"
    return True, ""


def load_split_manifest(split_dir: str, split_id: str) -> Optional[Dict[str, Any]]:
    path = os.path.join(split_dir, f"{split_id}_split_manifest.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        return manifest if isinstance(manifest, dict) else None
    except Exception as e:
        print(f"[RESUME] Could not read split manifest {path}: {e}")
        return None


def manifest_matches_split_membership(
    manifest: Dict[str, Any],
    train_case_ids: Sequence[str],
    test_case_ids: Sequence[str],
) -> Tuple[bool, str]:
    saved_train = [str(x) for x in manifest.get("train_case_ids", [])]
    saved_test = [str(x) for x in manifest.get("test_case_ids", [])]
    current_train = [str(x) for x in train_case_ids]
    current_test = [str(x) for x in test_case_ids]
    if saved_train != current_train or saved_test != current_test:
        return False, "train/test case ID lists differ from saved manifest"
    if manifest.get("train_case_hash") != case_id_list_hash(current_train):
        return False, "train_case_hash mismatch"
    if manifest.get("test_case_hash") != case_id_list_hash(current_test):
        return False, "test_case_hash mismatch"
    return True, ""


def validate_completed_checkpoint_tables(
    loaded: Dict[str, pd.DataFrame],
    require_predictions: bool = True,
) -> Tuple[bool, str]:
    for key, df in loaded.items():
        if not isinstance(df, pd.DataFrame):
            return False, f"{key} is not a DataFrame"
        if key == "predictions_df" and require_predictions and len(df) == 0:
            return False, "predictions_df is empty"
    return True, ""


def validate_split_marker(marker: Dict[str, Any], args: Any, split_id: str) -> Tuple[bool, str]:
    if str(marker.get("split_id")) != str(split_id):
        return False, f"marker split_id={marker.get('split_id')!r} expected={split_id!r}"
    saved_fp = marker.get("fingerprint")
    if isinstance(saved_fp, dict) and saved_fp:
        current_fp = build_split_resume_fingerprint(args, split_id)
        return split_resume_fingerprints_compatible(saved_fp, current_fp)
    legacy_checks = {
        "csv_path": str(getattr(args, "csv_path", "")),
        "outer_scheme": str(getattr(args, "outer_scheme", "")),
        "outer_repeats": int(getattr(args, "outer_repeats", 0) or 0),
        "outer_test_frac": float(getattr(args, "outer_test_frac", 0.0) or 0.0),
        "outer_folds": int(getattr(args, "outer_folds", 0) or 0),
        "random_seed": int(getattr(args, "random_seed", 0) or 0),
    }
    for key, expected in legacy_checks.items():
        if key in marker and marker.get(key) != expected:
            return False, f"{key}: saved={marker.get(key)!r} current={expected!r}"
    return True, ""


def indices_from_manifest(
    target_df: pd.DataFrame,
    manifest: Dict[str, Any],
) -> Tuple[List[int], List[int]]:
    """Reconstruct outer train/test row positions from saved provenance."""
    id_to_pos = {str(cid): int(pos) for pos, cid in enumerate(target_df["case_id"].astype(str))}
    train_pos = [id_to_pos[cid] for cid in manifest.get("train_case_ids", []) if str(cid) in id_to_pos]
    test_pos = [id_to_pos[cid] for cid in manifest.get("test_case_ids", []) if str(cid) in id_to_pos]
    return train_pos, test_pos
