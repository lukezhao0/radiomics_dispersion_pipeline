"""Model specs, pipelines, fitting, and teacher-student."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import html
import json
import math
import os
import re
import sys
import time
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, HuberRegressor, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    mutual_info_score,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold, StratifiedShuffleSplit, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR, SVC

from approach2.config import (
    AMBIGUITY_GROUPS,
    CANONICAL_GROUP_PATTERNS,
    COEF_ZERO_TOL,
    DEFAULT_BOOTSTRAP_N,
    DISPERSION_TRUE_HIGH_THRESHOLD,
    DISTRIBUTION_GROUPS,
    EPS,
    INNER_CV_MAX_SPLITS,
    META_COLS,
    NEGATION_PATTERNS,
    RANDOM_SEED,
    SHARED_CONCEPT_ONTOLOGY,
    SPATIAL_MORPH_RESPONSE_GROUPS,
    TARGET_NAME_DISPERSION_HIGH_LOW,
    TARGET_NAME_DISPERSION_SCORE,
    TARGET_NAME_RELAPSE_STATUS,
    UNCERTAINTY_PATTERNS,
)
from approach2.extraction import (
    MAX_TOKENS,
    Tee,
    _is_missing_text,
    _selected_report_text,
    _true_dispersion_high_low,
    build_html_report,
    build_user_prompt,
    confirm_cost_estimate_or_exit,
    configure_global_api_concurrency,
    df_to_html_table,
    estimate_prompt_tokens_from_messages,
    extract_subset_records,
    html_paragraph,
    html_plot_block,
    html_section,
    load_cases,
    make_case_from_row,
    preflight_check,
    print_apriori_cost_estimate_report,
    print_cumulative_report,
    summarize_apriori_cost_estimate,
    write_cost_tracker_json,
    write_extractions,
)
from approach2.io_atomic import atomic_write_df as _atomic_write_df
from approach2.io_atomic import safe_read_csv_if_exists as _safe_read_csv_if_exists
from approach2.metrics import calibration_intercept_slope, rmse, safe_pearson, safe_spearman
from approach2.models import LowInfoFeatureFilter, ModelSpec
from approach2.text_utils import (
    clean_phrase_for_display,
    detect_negation,
    detect_uncertainty,
    make_slug,
    normalize_text,
    parse_jsonish,
    resolve_default_api_workers,
    resolve_default_ml_n_jobs,
    resolve_default_parallel_modality_workers,
)

# -----------------------------
# Model builders
# -----------------------------

def make_inner_cv_classification(y_train: np.ndarray, random_seed: int) -> StratifiedKFold:
    class_counts = pd.Series(y_train).value_counts()
    max_splits = int(class_counts.min()) if len(class_counts) else 2
    n_splits = max(2, min(INNER_CV_MAX_SPLITS, max_splits))
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)


def make_inner_cv_regression(n_train: int, random_seed: int) -> KFold:
    n_splits = max(2, min(INNER_CV_MAX_SPLITS, n_train))
    return KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)


def build_pipeline_and_grid(spec: ModelSpec, n_features: int, n_train: int) -> Tuple[Pipeline, List[Dict[str, Any]]]:
    base_steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("lowinfo", LowInfoFeatureFilter()),
        ("scaler", StandardScaler()),
    ]

    if spec.key == "ridge_logistic":
        model = LogisticRegression(
            penalty="l2",
            solver="liblinear",
            class_weight="balanced",
            max_iter=5000,
        )
        grid = [{"model__C": [0.01, 0.1, 1.0, 10.0, 100.0]}]
    elif spec.key == "elasticnet_logistic":
        model = LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            class_weight="balanced",
            max_iter=5000,
        )
        grid = [{"model__C": [0.01, 0.1, 1.0, 10.0], "model__l1_ratio": [0.1, 0.5, 0.9]}]
    elif spec.key == "linear_svm":
        model = SVC(
            kernel="linear",
            probability=True,
            class_weight="balanced",
        )
        grid = [{"model__C": [0.01, 0.1, 1.0, 10.0, 100.0]}]
    elif spec.key == "ridge_regression":
        model = Ridge()
        grid = [{"model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]}]
    elif spec.key == "elasticnet_regression":
        model = ElasticNet(max_iter=5000)
        grid = [{"model__alpha": [0.001, 0.01, 0.1, 1.0], "model__l1_ratio": [0.1, 0.5, 0.9]}]
    elif spec.key == "pls_regression":
        max_comp = max(1, min(10, n_features, max(1, n_train - 1)))
        model = PLSRegression()
        grid = [{"model__n_components": list(range(1, max_comp + 1))}]
    elif spec.key == "linear_svr":
        model = LinearSVR(max_iter=10000)
        grid = [{"model__C": [0.01, 0.1, 1.0, 10.0], "model__epsilon": [0.01, 0.1, 1.0]}]
    elif spec.key == "huber_regression":
        model = HuberRegressor(max_iter=1000)
        grid = [{"model__alpha": [1e-5, 1e-4, 1e-3], "model__epsilon": [1.1, 1.35, 1.6]}]
    else:
        raise ValueError(f"Unsupported model spec: {spec.key}")

    pipe = Pipeline(base_steps + [("model", model)])
    return pipe, grid


def get_model_specs() -> List[ModelSpec]:
    return [
        ModelSpec("ridge_logistic", "classification", "ridge_logistic", "LogisticRegression(L2)", "roc_auc", True, "Primary classification baseline."),
        ModelSpec("elasticnet_logistic", "classification", "elasticnet_logistic", "LogisticRegression(ElasticNet)", "roc_auc", True, "Secondary sparse classification model."),
        ModelSpec("linear_svm", "classification", "linear_svm", "SVC(kernel=linear)", "roc_auc", True, "Linear SVM sensitivity analysis."),
        ModelSpec("ridge_regression", "regression", "ridge_regression", "Ridge", "neg_mean_absolute_error", False, "Primary regression baseline."),
        ModelSpec("elasticnet_regression", "regression", "elasticnet_regression", "ElasticNet", "neg_mean_absolute_error", False, "Sparse linear regression sensitivity."),
        ModelSpec("pls_regression", "regression", "pls_regression", "PLSRegression", "neg_mean_absolute_error", False, "Low-rank correlated-feature regression."),
        ModelSpec("linear_svr", "regression", "linear_svr", "LinearSVR", "neg_mean_absolute_error", False, "Linear SVR sensitivity analysis."),
        ModelSpec("huber_regression", "regression", "huber_regression", "HuberRegressor", "neg_mean_absolute_error", False, "Robust regression for heavy tails/outliers."),
    ]


# -----------------------------
# Metrics and coefficients
# -----------------------------

def classification_metrics(pred_df: pd.DataFrame) -> Dict[str, Any]:
    df = pred_df.copy()
    df = df[df["y_true"].notna() & df["y_prob"].notna() & df["y_pred"].notna()].copy()

    out = {
        "n": len(df),
        "prevalence": np.nan,
        "auprc_no_skill_baseline": np.nan,
        "positive_prediction_rate": np.nan,
        "auroc": np.nan,
        "auprc": np.nan,
        "brier": np.nan,
        "accuracy": np.nan,
        "balanced_accuracy": np.nan,
        "f1": np.nan,
        "precision": np.nan,
        "precision_ppv": np.nan,
        "recall_sensitivity": np.nan,
        "specificity": np.nan,
        "npv": np.nan,
        "tn": np.nan,
        "fp": np.nan,
        "fn": np.nan,
        "tp": np.nan,
        "calibration_intercept": np.nan,
        "calibration_slope": np.nan,
    }
    if len(df) == 0:
        return out

    y_true = df["y_true"].astype(int).values
    prob = np.clip(df["y_prob"].astype(float).values, 0.0, 1.0)
    y_pred = df["y_pred"].astype(int).values

    out["prevalence"] = float(np.mean(y_true))
    out["auprc_no_skill_baseline"] = out["prevalence"]
    out["positive_prediction_rate"] = float(np.mean(y_pred))
    out["brier"] = float(brier_score_loss(y_true, prob))
    out["accuracy"] = float(accuracy_score(y_true, y_pred))
    out["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    out["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    out["precision_ppv"] = out["precision"]
    out["recall_sensitivity"] = float(recall_score(y_true, y_pred, zero_division=0))

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out.update({"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})
    out["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else np.nan
    out["npv"] = float(tn / (tn + fn)) if (tn + fn) > 0 else np.nan
    if pd.notna(out["recall_sensitivity"]) and pd.notna(out["specificity"]):
        out["balanced_accuracy"] = float((out["recall_sensitivity"] + out["specificity"]) / 2.0)

    if len(np.unique(y_true)) >= 2:
        out["auroc"] = float(roc_auc_score(y_true, prob))
        out["auprc"] = float(average_precision_score(y_true, prob))
        ci, cs = calibration_intercept_slope(y_true, prob)
        out["calibration_intercept"] = ci
        out["calibration_slope"] = cs
    return out


def regression_metrics(pred_df: pd.DataFrame) -> Dict[str, Any]:
    y_true = pred_df["y_true"].astype(float).values
    y_pred = pred_df["y_pred_value"].astype(float).values
    spearman_rho, _ = safe_spearman(y_true, y_pred)
    pearson_r, _ = safe_pearson(y_true, y_pred)
    return {
        "n": len(pred_df),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse(y_true, y_pred),
        "r2": float(r2_score(y_true, y_pred)),
        "spearman_rho": spearman_rho,
        "pearson_r": pearson_r,
    }




def model_target_specs_for_model(spec: ModelSpec) -> List[Dict[str, str]]:
    if spec.task_type == "regression":
        return [{
            "target_name": TARGET_NAME_DISPERSION_SCORE,
            "target_col": "dispersion_true",
            "task_type": "regression",
        }]
    if spec.task_type == "classification":
        return [
            {
                "target_name": TARGET_NAME_DISPERSION_HIGH_LOW,
                "target_col": "dispersion_true_high_low",
                "task_type": "classification",
            },
            {
                "target_name": TARGET_NAME_RELAPSE_STATUS,
                "target_col": "relapse_true",
                "task_type": "classification",
            },
        ]
    raise ValueError(f"Unsupported task_type: {spec.task_type}")


def prepare_task_frames(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str,
    task_type: str,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    train = train_df.copy()
    test = test_df.copy()
    train = train[train[target_col].notna()].copy()
    test = test[test[target_col].notna()].copy()

    X_train = train[list(feature_cols)].copy()
    X_test = test[list(feature_cols)].copy()
    if task_type == "classification":
        y_train = train[target_col].astype(int)
        y_test = test[target_col].astype(int)
    else:
        y_train = train[target_col].astype(float)
        y_test = test[target_col].astype(float)
    return X_train, y_train, X_test, y_test


def annotate_coefficient_table(
    coef_df: pd.DataFrame,
    X_train: pd.DataFrame,
    X_test: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Add lightweight interpretation metadata to fitted coefficient tables."""
    if coef_df is None or len(coef_df) == 0 or "feature" not in coef_df.columns:
        return coef_df if coef_df is not None else pd.DataFrame()

    out = coef_df.copy()
    X_train_df = pd.DataFrame(X_train).copy()
    X_test_df = pd.DataFrame(X_test).copy() if X_test is not None else pd.DataFrame(columns=X_train_df.columns)

    train_prev = {}
    test_prev = {}
    train_mean = {}
    for col in X_train_df.columns:
        vals = pd.to_numeric(X_train_df[col], errors="coerce").fillna(0.0)
        train_prev[str(col)] = float((vals != 0).mean()) if len(vals) else np.nan
        train_mean[str(col)] = float(vals.mean()) if len(vals) else np.nan
    for col in X_test_df.columns:
        vals = pd.to_numeric(X_test_df[col], errors="coerce").fillna(0.0)
        test_prev[str(col)] = float((vals != 0).mean()) if len(vals) else np.nan

    def _feature_modality(feature: str) -> str:
        feature = str(feature)
        if feature.startswith("mri__") or feature.startswith("weighted__") or feature.startswith("mri_"):
            return "mri"
        if feature.startswith("path__") or feature.startswith("path_"):
            return "pathology"
        return "unknown"

    def _ontology_concept(feature: str) -> str:
        f = str(feature)
        for prefix in ["mri__group__", "path__group__", "weighted__", "group__", "group_status__", "group_count__", "group_binary__", "phrase__", "mri__", "path__"]:
            if f.startswith(prefix):
                f = f[len(prefix):]
        f = re.sub(r"^(present|count|negated_count|uncertain_count)__", "", f)
        for concept in SHARED_CONCEPT_ONTOLOGY:
            if concept in f:
                return concept
        return f

    out["feature_prevalence_train"] = out["feature"].astype(str).map(train_prev)
    out["feature_prevalence_test"] = out["feature"].astype(str).map(test_prev)
    out["feature_mean_train"] = out["feature"].astype(str).map(train_mean)
    out["feature_modality"] = out["feature"].astype(str).map(_feature_modality)
    out["ontology_concept"] = out["feature"].astype(str).map(_ontology_concept)
    out["coef_sign"] = np.where(out["coef"].astype(float) > COEF_ZERO_TOL, "positive", np.where(out["coef"].astype(float) < -COEF_ZERO_TOL, "negative", "zero"))
    return out


def should_skip_model_fit(
    y_train: pd.Series,
    y_test: pd.Series,
    task_type: str,
    split_id: str,
    dataset_key: str,
    representation: str,
    model_key: str,
    target_name: str,
) -> bool:
    if len(y_train) == 0 or len(y_test) == 0:
        print(
            f"[SKIP] split={split_id} dataset={dataset_key} representation={representation} "
            f"model={model_key} target={target_name}: empty train/test after target and MRI-missing filtering."
        )
        return True
    if task_type == "classification":
        counts = pd.Series(y_train).value_counts()
        if len(counts) < 2:
            print(
                f"[SKIP] split={split_id} dataset={dataset_key} representation={representation} "
                f"model={model_key} target={target_name}: training labels contain only one class."
            )
            return True
        if int(counts.min()) < 2:
            print(
                f"[SKIP] split={split_id} dataset={dataset_key} representation={representation} "
                f"model={model_key} target={target_name}: minority class has <2 training cases, "
                f"so inner StratifiedKFold would be invalid. class_counts={counts.to_dict()}"
            )
            return True
    else:
        if len(y_train) < 3 or len(np.unique(y_train.astype(float).values)) < 2:
            print(
                f"[SKIP] split={split_id} dataset={dataset_key} representation={representation} "
                f"model={model_key} target={target_name}: insufficient continuous target variation."
            )
            return True
    return False


def add_bootstrap_metric_cis(
    base_metrics: Dict[str, Any],
    pred_df: pd.DataFrame,
    task_type: str,
    n_bootstrap: int,
    random_seed: int,
) -> Dict[str, Any]:
    """Add simple case-level bootstrap CIs for aggregate held-out metrics."""
    if n_bootstrap <= 0 or len(pred_df) < 2:
        return base_metrics

    rng = np.random.default_rng(int(random_seed))
    metric_names = [
        "auroc", "auprc", "brier", "accuracy", "balanced_accuracy", "f1", "precision",
        "precision_ppv", "recall_sensitivity", "specificity", "npv",
    ] if task_type == "classification" else ["mae", "rmse", "r2", "spearman_rho", "pearson_r"]
    samples: Dict[str, List[float]] = {m: [] for m in metric_names}
    df = pred_df.reset_index(drop=True).copy()

    for _ in range(int(n_bootstrap)):
        idx = rng.integers(0, len(df), len(df))
        boot = df.iloc[idx].copy()
        try:
            m = classification_metrics(boot) if task_type == "classification" else regression_metrics(boot)
        except Exception:
            continue
        for name in metric_names:
            val = m.get(name, np.nan)
            if pd.notna(val):
                samples[name].append(float(val))

    out = dict(base_metrics)
    for name, vals in samples.items():
        if vals:
            out[f"{name}_ci_low"] = float(np.percentile(vals, 2.5))
            out[f"{name}_ci_high"] = float(np.percentile(vals, 97.5))
        else:
            out[f"{name}_ci_low"] = np.nan
            out[f"{name}_ci_high"] = np.nan
    out["bootstrap_n"] = int(n_bootstrap)
    return out

def extract_fitted_feature_coefficients(best_estimator: Pipeline, original_feature_names: List[str]) -> pd.DataFrame:
    try:
        selected_features = list(
            best_estimator.named_steps["lowinfo"].get_feature_names_out()
        )
    except Exception:
        selected_features = list(original_feature_names)

    model = best_estimator.named_steps["model"]
    coef = None

    if hasattr(model, "coef_"):
        coef = np.asarray(model.coef_).reshape(-1)
    elif hasattr(model, "feature_importances_"):
        coef = np.asarray(model.feature_importances_).reshape(-1)

    if coef is None:
        return pd.DataFrame(columns=["feature", "coef", "abs_coef"])

    n = min(len(selected_features), len(coef))
    out = pd.DataFrame({
        "feature": selected_features[:n],
        "coef": coef[:n],
    })
    out["abs_coef"] = out["coef"].abs()
    return out


# -----------------------------
# Outer-loop model fitting
# -----------------------------

def fit_one_outer_model(
    spec: ModelSpec,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    split_random_seed: int,
    ml_n_jobs: int,
) -> Tuple[pd.DataFrame, Dict[str, Any], pd.DataFrame]:
    X_train = pd.DataFrame(X_train).copy()
    X_test = pd.DataFrame(X_test).copy()
    feature_names = list(X_train.columns)

    pipe, param_grid = build_pipeline_and_grid(spec, n_features=X_train.shape[1], n_train=len(X_train))
    if spec.task_type == "classification":
        inner_cv = make_inner_cv_classification(y_train.values, split_random_seed)
    else:
        inner_cv = make_inner_cv_regression(len(X_train), split_random_seed)

    from joblib import parallel_backend
    search = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring=spec.scoring,
        cv=inner_cv,
        refit=True,
        n_jobs=ml_n_jobs,
        error_score=np.nan,
    )
    try:
        with parallel_backend("threading", n_jobs=ml_n_jobs):
            search.fit(X_train, y_train)
    except Exception as e:
        print(
            f"[WARN] Threaded GridSearchCV failed for model; "
            f"falling back to serial fit. Error={e}"
        )
        search = GridSearchCV(
            estimator=pipe,
            param_grid=param_grid,
            scoring=spec.scoring,
            cv=inner_cv,
            refit=True,
            n_jobs=1,
            error_score=np.nan,
        )
        search.fit(X_train, y_train)

    best_estimator = search.best_estimator_
    best_params = dict(search.best_params_)
    best_score = float(search.best_score_) if search.best_score_ is not None else np.nan
    coef_df = extract_fitted_feature_coefficients(best_estimator, feature_names)

    if spec.task_type == "classification":
        y_prob = best_estimator.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        pred_df = pd.DataFrame({
            "y_prob": y_prob,
            "y_pred": y_pred,
        })
    else:
        y_pred_value = best_estimator.predict(X_test)
        y_pred_value = np.asarray(y_pred_value).reshape(-1)
        pred_df = pd.DataFrame({
            "y_pred_value": y_pred_value,
        })

    hyper = {
        "best_score_inner_cv": best_score,
        "best_params_json": json.dumps(best_params, sort_keys=True),
        "n_features_input": X_train.shape[1],
        "n_train": len(X_train),
        "n_test": len(X_test),
    }
    return pred_df, hyper, coef_df


# -----------------------------
# Teacher-student MRI model
# -----------------------------

def _fixed_ridge_pipeline(alpha: float = 10.0) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("lowinfo", LowInfoFeatureFilter()),
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=alpha)),
    ])


def _make_oof_teacher_score(X_path: pd.DataFrame, y: pd.Series, random_seed: int, alpha: float) -> np.ndarray:
    if len(X_path) < 5:
        model = _fixed_ridge_pipeline(alpha=alpha)
        model.fit(X_path, y)
        return np.asarray(model.predict(X_path)).reshape(-1)
    cv = make_inner_cv_regression(len(X_path), random_seed)
    model = _fixed_ridge_pipeline(alpha=alpha)
    try:
        pred = cross_val_predict(model, X_path, y, cv=cv, n_jobs=1)
    except Exception as e:
        print(f"[WARN] cross_val_predict teacher failed; using in-sample teacher scores. Error={e}")
        model.fit(X_path, y)
        pred = model.predict(X_path)
    return np.asarray(pred).reshape(-1)


def fit_teacher_student_mri_model(
    X_mri_train: pd.DataFrame,
    X_mri_test: pd.DataFrame,
    X_path_train: pd.DataFrame,
    y_train: pd.Series,
    y_test_cont: pd.Series,
    y_test_binary: pd.Series,
    path_concept_train: pd.DataFrame,
    split_random_seed: int,
    alpha: float,
    lambda_dispersion: float,
    lambda_teacher_score: float,
    lambda_path_concepts: float,
    y_train_relapse: Optional[pd.Series] = None,
    y_test_relapse: Optional[pd.Series] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame], Dict[str, Any]]:
    """Simple multi-output ridge student.

    Inputs are MRI-only. Training targets include the true continuous dispersion,
    a pathology-teacher OOF continuous score, and pathology concept presences.
    The returned primary prediction is the first output, i.e. continuous dispersion.
    """
    teacher_score_train = _make_oof_teacher_score(X_path_train, y_train, split_random_seed, alpha=alpha)

    y_scaler = StandardScaler()
    y_primary_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1)).reshape(-1)
    teacher_scaler = StandardScaler()
    teacher_scaled = teacher_scaler.fit_transform(teacher_score_train.reshape(-1, 1)).reshape(-1)

    concept_cols = [c for c in path_concept_train.columns if c not in META_COLS]
    if concept_cols:
        path_concepts = path_concept_train[concept_cols].copy().fillna(0.0)
        # Keep small-sample outputs bounded and comparable to the scaled y targets.
        path_concepts_scaled = StandardScaler(with_mean=True, with_std=True).fit_transform(path_concepts)
    else:
        path_concepts_scaled = np.zeros((len(y_train), 0))

    targets = [lambda_dispersion * y_primary_scaled.reshape(-1, 1)]
    if lambda_teacher_score > 0:
        targets.append(lambda_teacher_score * teacher_scaled.reshape(-1, 1))
    if lambda_path_concepts > 0 and path_concepts_scaled.shape[1] > 0:
        targets.append(lambda_path_concepts * path_concepts_scaled)
    Y_multi = np.hstack(targets)

    pipe = _fixed_ridge_pipeline(alpha=alpha)
    pipe.fit(X_mri_train, Y_multi)
    pred_multi = np.asarray(pipe.predict(X_mri_test))
    if pred_multi.ndim == 1:
        pred_multi = pred_multi.reshape(-1, 1)
    y_pred_scaled = pred_multi[:, 0] / max(lambda_dispersion, EPS)
    y_pred_value = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).reshape(-1)
    y_prob = np.clip((y_pred_value - DISPERSION_TRUE_HIGH_THRESHOLD + 20.0) / 40.0, 0.0, 1.0)
    y_pred_binary = (y_pred_value >= DISPERSION_TRUE_HIGH_THRESHOLD).astype(int)

    reg_pred = pd.DataFrame({
        "y_true": y_test_cont.values,
        "y_pred_value": y_pred_value,
    })
    cls_pred = pd.DataFrame({
        "y_true": y_test_binary.values,
        "y_prob": y_prob,
        "y_pred": y_pred_binary,
    })

    relapse_pred: Optional[pd.DataFrame] = None
    if y_train_relapse is not None and y_test_relapse is not None:
        y_tr = y_train_relapse.dropna().astype(int)
        y_te = y_test_relapse.dropna().astype(int)
        if len(y_tr) >= 3 and len(np.unique(y_tr.values)) >= 2 and len(y_te) > 0:
            train_idx = y_train_relapse.notna()
            test_idx = y_test_relapse.notna()
            clf = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("lowinfo", LowInfoFeatureFilter()),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(
                    penalty="l2",
                    solver="liblinear",
                    class_weight="balanced",
                    max_iter=5000,
                    random_state=split_random_seed,
                )),
            ])
            clf.fit(X_mri_train.loc[train_idx], y_tr)
            y_prob_rel = clf.predict_proba(X_mri_test.loc[test_idx])[:, 1]
            relapse_pred = pd.DataFrame({
                "y_true": y_te.values,
                "y_prob": y_prob_rel,
                "y_pred": (y_prob_rel >= 0.5).astype(int),
            })
        else:
            print("[TEACHER_STUDENT] Skipping relapse head: insufficient class counts in train/test.")

    hyper = {
        "teacher_student_alpha": alpha,
        "lambda_dispersion": lambda_dispersion,
        "lambda_teacher_score": lambda_teacher_score,
        "lambda_path_concepts": lambda_path_concepts,
        "n_mri_features": X_mri_train.shape[1],
        "n_path_concept_targets": len(concept_cols),
        "n_train": len(X_mri_train),
        "n_test": len(X_mri_test),
        "relapse_head_fitted": relapse_pred is not None,
    }
    coef_df = extract_fitted_feature_coefficients(pipe, list(X_mri_train.columns))
    return reg_pred, cls_pred, relapse_pred, {"hyper": hyper, "coef_df": coef_df}
