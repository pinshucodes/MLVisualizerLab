"""Data preprocessing utilities for ML Visualizer Lab.

Handles missing values, categorical encoding, feature scaling, and
train/test splitting with stratification support.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
)


# ---------------------------------------------------------------------------
# Missing values
# ---------------------------------------------------------------------------

def handle_missing_values(df: pd.DataFrame, strategy: str) -> pd.DataFrame:
    """Handle missing values in a DataFrame.

    Parameters
    ----------
    df:
        Input DataFrame (not modified in-place).
    strategy:
        One of ``'drop'``, ``'mean'``, ``'median'``, or ``'most_frequent'``.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with missing values handled.

    Raises
    ------
    ValueError
        If *strategy* is not recognised.
    """
    valid_strategies = ("drop", "mean", "median", "most_frequent")
    if strategy not in valid_strategies:
        raise ValueError(
            f"Unknown strategy '{strategy}'. Choose from {valid_strategies}."
        )

    if strategy == "drop":
        return df.dropna().reset_index(drop=True)

    # For imputation, split numeric and non-numeric columns
    result = df.copy()
    numeric_cols = result.select_dtypes(include="number").columns.tolist()
    non_numeric_cols = result.select_dtypes(exclude="number").columns.tolist()

    if strategy in ("mean", "median"):
        # Apply mean/median only to numeric; use most_frequent for others
        if numeric_cols:
            imp_num = SimpleImputer(strategy=strategy)
            result[numeric_cols] = imp_num.fit_transform(result[numeric_cols])
        if non_numeric_cols:
            imp_cat = SimpleImputer(strategy="most_frequent")
            result[non_numeric_cols] = imp_cat.fit_transform(result[non_numeric_cols])
    else:  # most_frequent
        imp = SimpleImputer(strategy="most_frequent")
        result = pd.DataFrame(
            imp.fit_transform(result), columns=result.columns
        )
        # Restore original dtypes where possible
        for col in result.columns:
            try:
                result[col] = result[col].astype(df[col].dtype)
            except (ValueError, TypeError):
                pass

    return result


# ---------------------------------------------------------------------------
# Categorical encoding
# ---------------------------------------------------------------------------

def encode_categoricals(
    df: pd.DataFrame,
    columns: list[str],
    method: str,
) -> pd.DataFrame:
    """Encode categorical columns.

    Parameters
    ----------
    df:
        Input DataFrame (not modified in-place).
    columns:
        Column names to encode.
    method:
        ``'label'`` for ordinal label encoding, ``'onehot'`` for one-hot
        (dummy) encoding.

    Returns
    -------
    pd.DataFrame
        DataFrame with specified columns encoded.

    Raises
    ------
    ValueError
        If *method* is not ``'label'`` or ``'onehot'``.
    """
    if method not in ("label", "onehot"):
        raise ValueError(f"Unknown encoding method '{method}'. Use 'label' or 'onehot'.")

    result = df.copy()

    if method == "label":
        for col in columns:
            le = LabelEncoder()
            result[col] = le.fit_transform(result[col].astype(str))
    else:  # onehot
        result = pd.get_dummies(result, columns=columns, drop_first=False)
        # Ensure all columns are numeric (bool -> int)
        for col in result.columns:
            if result[col].dtype == "bool":
                result[col] = result[col].astype(int)

    return result


# ---------------------------------------------------------------------------
# Feature scaling
# ---------------------------------------------------------------------------

def scale_features(
    X: np.ndarray,
    method: str,
    feature_names: list[str],
) -> tuple[np.ndarray, Any]:
    """Scale feature matrix.

    Parameters
    ----------
    X:
        Feature array of shape ``(n_samples, n_features)``.
    method:
        ``'none'``, ``'standard'``, ``'minmax'``, or ``'robust'``.
    feature_names:
        Column names (currently informational; reserved for future use).

    Returns
    -------
    tuple[np.ndarray, Any]
        ``(scaled_X, scaler)`` where *scaler* is the fitted scaler object
        or ``None`` when ``method='none'``.

    Raises
    ------
    ValueError
        If *method* is not recognised.
    """
    if method == "none":
        return X, None

    scalers = {
        "standard": StandardScaler,
        "minmax": MinMaxScaler,
        "robust": RobustScaler,
    }
    if method not in scalers:
        raise ValueError(
            f"Unknown scaling method '{method}'. "
            f"Choose from {list(scalers.keys()) + ['none']}."
        )

    scaler = scalers[method]()
    X_scaled: np.ndarray = scaler.fit_transform(X)
    return X_scaled, scaler


# ---------------------------------------------------------------------------
# Train / test split
# ---------------------------------------------------------------------------

def split_data(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    test_size: float,
    task_type: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split data into train and test sets.

    Parameters
    ----------
    X:
        Feature matrix.
    y:
        Target vector.
    test_size:
        Fraction of data reserved for testing (0 < test_size < 1).
    task_type:
        ``'classification'`` or ``'regression'``.  Stratified splitting is
        used for classification tasks.

    Returns
    -------
    tuple
        ``(X_train, X_test, y_train, y_test)`` as NumPy arrays.
    """
    stratify = y if task_type == "classification" else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
        stratify=stratify,
    )

    return (
        np.asarray(X_train),
        np.asarray(X_test),
        np.asarray(y_train),
        np.asarray(y_test),
    )
