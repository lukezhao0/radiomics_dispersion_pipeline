"""Sklearn model specifications and feature filtering for approach2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class LowInfoFeatureFilter(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        min_non_missing: int = 2,
        min_nonzero: int = 1,
        min_unique_non_missing: int = 2,
    ):
        self.min_non_missing = min_non_missing
        self.min_nonzero = min_nonzero
        self.min_unique_non_missing = min_unique_non_missing

    def fit(self, X: Any, y: Any = None) -> "LowInfoFeatureFilter":
        X_df = pd.DataFrame(X).copy()
        self.feature_names_in_ = list(X_df.columns)

        non_missing = X_df.notna().sum(axis=0)
        nonzero = (X_df.fillna(0.0) != 0).sum(axis=0)
        unique_non_missing = X_df.nunique(dropna=True)

        keep = (
            (non_missing >= self.min_non_missing)
            & (nonzero >= self.min_nonzero)
            & (unique_non_missing >= self.min_unique_non_missing)
        )

        if int(keep.sum()) == 0:
            keep = pd.Series(True, index=X_df.columns)

        self.selected_features_ = list(X_df.columns[keep.values])
        self.support_mask_ = keep.values.astype(bool)
        return self

    def transform(self, X: Any) -> pd.DataFrame:
        X_df = pd.DataFrame(X).copy()
        if X_df.shape[1] == len(getattr(self, "feature_names_in_", [])):
            X_df.columns = self.feature_names_in_
        return X_df.loc[:, self.selected_features_]

    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> np.ndarray:
        return np.asarray(self.selected_features_, dtype=object)


@dataclass
class ModelSpec:
    key: str
    task_type: str
    family: str
    estimator_name: str
    scoring: str
    supports_probability: bool
    notes: str
