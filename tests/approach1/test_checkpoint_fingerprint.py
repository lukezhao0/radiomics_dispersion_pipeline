"""Checkpoint fingerprint compatibility tests."""

from __future__ import annotations

from approach1.checkpoint.fingerprint import build_config_fingerprint, config_fingerprints_compatible
from approach1.models import RunConfig


def _minimal_run_config(tmp_path) -> RunConfig:
    return RunConfig(
        shotset_name="test_shotset",
        high_rows=[0, 1],
        low_rows=[2, 3],
        training_rows=[0, 1, 2, 3],
        modality="pathology_only",
        run_out_dir=str(tmp_path),
        training_block="EXAMPLE",
        test_cases_with_idxs=[(4, None)],  # type: ignore[arg-type]
        skipped_missing_mri=[],
        apriori_cost={"n_calls": 1},
    )


def test_fingerprint_roundtrip(tmp_path):
    rc = _minimal_run_config(tmp_path)
    fp = build_config_fingerprint(rc)
    ok, msg = config_fingerprints_compatible(fp, rc)
    assert ok, msg


def test_fingerprint_rejects_modality_change(tmp_path):
    rc = _minimal_run_config(tmp_path)
    fp = build_config_fingerprint(rc)
    rc2 = RunConfig(
        shotset_name=rc.shotset_name,
        high_rows=rc.high_rows,
        low_rows=rc.low_rows,
        training_rows=rc.training_rows,
        modality="mri_only",
        run_out_dir=rc.run_out_dir,
        training_block=rc.training_block,
        test_cases_with_idxs=rc.test_cases_with_idxs,
        skipped_missing_mri=rc.skipped_missing_mri,
        apriori_cost=rc.apriori_cost,
    )
    ok, msg = config_fingerprints_compatible(fp, rc2)
    assert not ok
    assert "modality" in msg
