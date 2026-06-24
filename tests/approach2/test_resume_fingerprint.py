"""Split resume fingerprint compatibility tests."""

from __future__ import annotations

import argparse
from types import SimpleNamespace

from approach2.checkpoint import (
    build_split_resume_fingerprint,
    split_resume_fingerprints_compatible,
    validate_split_marker,
)
from approach2.splits import case_id_list_hash


def _args(**kwargs):
    defaults = dict(
        csv_path="/data/cases.csv",
        outer_scheme="repeated_mc",
        outer_repeats=5,
        outer_test_frac=0.2,
        outer_folds=5,
        random_seed=17,
        stability_threshold=0.35,
        target_stable_features_per_modality=0,
        modalities=["mri", "path"],
        representations=["group_binary"],
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def test_fingerprint_roundtrip() -> None:
    args = _args()
    fp = build_split_resume_fingerprint(args, "outer_split_001")
    ok, msg = split_resume_fingerprints_compatible(fp, fp)
    assert ok, msg


def test_fingerprint_rejects_csv_change() -> None:
    saved = build_split_resume_fingerprint(_args(), "outer_split_001")
    current = build_split_resume_fingerprint(_args(csv_path="/other.csv"), "outer_split_001")
    ok, msg = split_resume_fingerprints_compatible(saved, current)
    assert not ok
    assert "csv_path" in msg


def test_validate_split_marker_legacy_without_fingerprint_block() -> None:
    args = _args()
    marker = {
        "split_id": "outer_split_001",
        "csv_path": args.csv_path,
        "outer_scheme": args.outer_scheme,
        "outer_repeats": args.outer_repeats,
        "outer_test_frac": args.outer_test_frac,
        "outer_folds": args.outer_folds,
        "random_seed": args.random_seed,
    }
    ok, _ = validate_split_marker(marker, args, "outer_split_001")
    assert ok


def test_case_id_list_hash_stable() -> None:
    assert case_id_list_hash(["a", "b"]) == case_id_list_hash(["a", "b"])
    assert case_id_list_hash(["a", "b"]) != case_id_list_hash(["b", "a"])
