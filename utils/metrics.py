"""Evaluation metrics utilities for ML Visualizer Lab.

Provides convenience wrappers around scikit-learn metrics for
classification, regression, and cross-validation scoring.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def get_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute standard classification metrics.

    Parameters
    ----------
    y_true:
        Ground-truth labels.
    y_pred:
        Predicted labels.
    y_proba:
        Predicted probabilities (shape ``(n_samples,)`` for binary or
        ``(n_samples, n_classes)`` for multiclass).  May be ``None``.

    Returns
    -------
    dict[str, float]
        Keys: ``accuracy``, ``precision``, ``recall``, ``f1``, ``roc_auc``.
        ``roc_auc`` is ``None`` when it cannot be computed.
    """
    n_classes = len(np.unique(y_true))
    average = "binary" if n_classes == 2 else "weighted"

    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=average, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
    }

    # ROC-AUC requires probability estimates
    roc_auc: float | None = None
    if y_proba is not None:
        try:
            if n_classes == 2:
                # Use probability of the positive class
                proba = (
                    y_proba[:, 1]
                    if y_proba.ndim == 2
                    else y_proba
                )
                roc_auc = float(roc_auc_score(y_true, proba))
            else:
                roc_auc = float(
                    roc_auc_score(
                        y_true,
                        y_proba,
                        multi_class="ovr",
                        average="weighted",
                    )
                )
        except (ValueError, IndexError):
            roc_auc = None

    metrics["roc_auc"] = roc_auc
    return metrics


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------

def get_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Compute standard regression metrics.

    Parameters
    ----------
    y_true:
        Ground-truth target values.
    y_pred:
        Predicted target values.

    Returns
    -------
    dict[str, float]
        Keys: ``mae``, ``mse``, ``rmse``, ``r2``.
    """
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(y_true, y_pred)),
    }


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def get_cv_scores(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    cv_type: str,
    n_folds: int,
    scoring: str,
) -> dict[str, Any]:
    """Perform cross-validation and return summary statistics.

    Parameters
    ----------
    model:
        An unfitted scikit-learn estimator (or compatible object).
    X:
        Feature matrix.
    y:
        Target vector.
    cv_type:
        ``'kfold'`` or ``'stratified'``.  Stratified is recommended for
        classification tasks.
    n_folds:
        Number of CV folds.
    scoring:
        Scoring metric name accepted by ``sklearn.model_selection.cross_val_score``
        (e.g. ``'accuracy'``, ``'f1_weighted'``, ``'r2'``).

    Returns
    -------
    dict
        ``scores`` (np.ndarray of per-fold scores), ``mean`` (float),
        ``std`` (float).

    Raises
    ------
    ValueError
        If *cv_type* is not ``'kfold'`` or ``'stratified'``.
    """
    if cv_type == "stratified":
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    elif cv_type == "kfold":
        cv = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    else:
        raise ValueError(
            f"Unknown cv_type '{cv_type}'. Use 'kfold' or 'stratified'."
        )

    scores: np.ndarray = cross_val_score(model, X, y, cv=cv, scoring=scoring)
    return {
        "scores": scores,
        "mean": float(scores.mean()),
        "std": float(scores.std()),
    }
