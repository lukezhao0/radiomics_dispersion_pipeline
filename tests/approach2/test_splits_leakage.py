"""Outer split leakage and k-fold coverage guards."""

from __future__ import annotations

import numpy as np
import pytest

from approach2.splits import build_outer_splits, validate_outer_splits


def test_repeated_mc_splits_have_no_train_test_overlap() -> None:
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 1])
    splits = build_outer_splits(y, "repeated_mc", random_seed=17, n_repeats=3, test_frac=0.2, n_folds=5)
    validate_outer_splits(splits, "repeated_mc", len(y))


def test_stratified_kfold_each_case_in_test_once() -> None:
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 1])
    splits = build_outer_splits(y, "stratified_kfold", random_seed=17, n_repeats=5, test_frac=0.2, n_folds=5)
    validate_outer_splits(splits, "stratified_kfold", len(y))
    assert len(splits) == 5


def test_validate_outer_splits_detects_overlap() -> None:
    splits = [(np.array([0, 1, 2]), np.array([2, 3, 4]))]
    with pytest.raises(ValueError, match="overlap"):
        validate_outer_splits(splits, "repeated_mc", 5)
