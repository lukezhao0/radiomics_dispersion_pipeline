"""Config fingerprinting for resume/checkpoint compatibility."""

from __future__ import annotations

from typing import Any, Dict, Tuple

from .. import config
from ..models import RunConfig


def build_config_fingerprint(rc: RunConfig) -> Dict[str, Any]:
    return {
        "resume_script_version": config.RESUME_SCRIPT_VERSION,
        "shotset_name": rc.shotset_name,
        "high_rows": list(rc.high_rows),
        "low_rows": list(rc.low_rows),
        "training_rows": list(rc.training_rows),
        "modality": rc.modality,
        "n_test_cases": len(rc.test_cases_with_idxs),
        "test_row_indices": [idx for idx, _ in rc.test_cases_with_idxs],
        "skipped_missing_mri_rows": [idx for idx, _ in rc.skipped_missing_mri],
        "deployment": config.DEPLOYMENT,
        "max_tokens": config.MAX_TOKENS,
        "temperature": config.TEMPERATURE,
    }


def config_fingerprints_compatible(saved: Dict[str, Any], rc: RunConfig) -> Tuple[bool, str]:
    current = build_config_fingerprint(rc)
    keys = [
        "resume_script_version",
        "shotset_name",
        "high_rows",
        "low_rows",
        "training_rows",
        "modality",
        "n_test_cases",
        "test_row_indices",
        "skipped_missing_mri_rows",
    ]
    for key in keys:
        if saved.get(key) != current.get(key):
            return False, f"{key}: saved={saved.get(key)!r} current={current.get(key)!r}"
    return True, ""
