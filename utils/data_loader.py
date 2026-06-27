"""Data loading utilities for ML Visualizer Lab.

Provides functions to load uploaded files, sample datasets from sklearn,
and inspect DataFrame metadata.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st
from sklearn.datasets import (
    load_diabetes,
    load_iris,
    load_wine,
)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def load_uploaded_file(uploaded_file: st.runtime.uploaded_file_manager.UploadedFile) -> pd.DataFrame:
    """Read an uploaded CSV or XLSX file into a DataFrame.

    Parameters
    ----------
    uploaded_file:
        A Streamlit ``UploadedFile`` object (from ``st.file_uploader``).

    Returns
    -------
    pd.DataFrame
        The parsed tabular data.

    Raises
    ------
    ValueError
        If the file extension is not ``.csv`` or ``.xlsx``.
    """
    name: str = uploaded_file.name.lower()

    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    elif name.endswith(".xlsx"):
        return pd.read_excel(uploaded_file, engine="openpyxl")
    else:
        raise ValueError(
            f"Unsupported file type: '{name.split('.')[-1]}'. "
            "Please upload a .csv or .xlsx file."
        )


@st.cache_data
def load_sample_dataset(name: str) -> pd.DataFrame:
    """Load a built-in sample dataset by name.

    Parameters
    ----------
    name:
        One of ``'iris'``, ``'wine'``, ``'diabetes'``, or
        ``'california_housing'``.

    Returns
    -------
    pd.DataFrame
        The dataset with feature columns and a ``target`` column.

    Raises
    ------
    ValueError
        If *name* is not a recognised dataset.
    """
    key = name.lower().strip()

    if key == "iris":
        data = load_iris()
        df = pd.DataFrame(data.data, columns=["sepal_length", "sepal_width",
                                                "petal_length", "petal_width"])
        df["species"] = pd.Series(data.target).map(
            {i: c for i, c in enumerate(data.target_names)}
        )
        return df

    if key == "wine":
        data = load_wine()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df["target"] = data.target
        return df

    if key == "diabetes":
        data = load_diabetes()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df["target"] = data.target
        return df

    if key in ("california_housing", "california housing"):
        # Lazy import – heavier download on first call
        from sklearn.datasets import fetch_california_housing

        data = fetch_california_housing()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df["target"] = data.target
        return df

    raise ValueError(
        f"Unknown dataset '{name}'. Choose from: "
        "'iris', 'wine', 'diabetes', 'california_housing'."
    )


@st.cache_data
def get_dataset_info(df: pd.DataFrame) -> dict[str, Any]:
    """Return summary metadata for a DataFrame.

    Parameters
    ----------
    df:
        The DataFrame to inspect.

    Returns
    -------
    dict
        Keys: ``n_rows``, ``n_cols``, ``missing_values`` (total count),
        ``dtypes`` (dict mapping column → dtype string),
        ``memory_usage`` (human-readable string, e.g. ``'1.2 MB'``).
    """
    mem_bytes: int = df.memory_usage(deep=True).sum()
    if mem_bytes < 1024:
        mem_str = f"{mem_bytes} B"
    elif mem_bytes < 1024 ** 2:
        mem_str = f"{mem_bytes / 1024:.1f} KB"
    else:
        mem_str = f"{mem_bytes / 1024 ** 2:.1f} MB"

    return {
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "missing_values": int(df.isna().sum().sum()),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "memory_usage": mem_str,
    }
