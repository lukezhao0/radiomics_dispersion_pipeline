"""Train/test split leakage guards."""

from __future__ import annotations

from approach1.data import load_cases
from approach1.splits import build_run_configs, resolve_shot_sets


def test_training_rows_disjoint_from_test(minimal_csv: str, tmp_path):
    df = load_cases(minimal_csv)
    configs = build_run_configs(df, str(tmp_path / "out"))
    for rc in configs:
        training = set(rc.training_rows)
        test_indices = {idx for idx, _ in rc.test_cases_with_idxs}
        assert training.isdisjoint(test_indices), (
            f"Leakage in {rc.shotset_name}/{rc.modality}: "
            f"overlap={training & test_indices}"
        )


def test_each_shotset_has_four_training_rows(minimal_csv: str, tmp_path):
    df = load_cases(minimal_csv)
    configs = build_run_configs(df, str(tmp_path / "out"))
    for rc in configs:
        assert len(rc.training_rows) == 4
        assert len(rc.high_rows) == 2
        assert len(rc.low_rows) == 2


def test_build_run_configs_filters_shotsets(minimal_csv: str, tmp_path):
    df = load_cases(minimal_csv)
    selected = resolve_shot_sets(["shotset_high_0_2_low_101_102"])
    configs = build_run_configs(df, str(tmp_path / "out"), shot_sets=selected)

    assert {rc.shotset_name for rc in configs} == {"shotset_high_0_2_low_101_102"}
    assert len(configs) == 3
    assert configs[0].high_rows == [0, 2]
    assert configs[0].low_rows == [101, 102]
