"""Shared fixtures for approach2 package tests."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def sample_arrays() -> tuple[np.ndarray, np.ndarray]:
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([2.0, 4.0, 5.0, 4.0, 5.0])
    return x, y
