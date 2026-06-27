"""Export utilities for ML Visualizer Lab.

Serialize trained models, metrics, and experiment histories for download.
"""

from __future__ import annotations

import io
from typing import Any

import joblib
import pandas as pd


def export_model(model: Any) -> bytes:
    """Serialize a trained model to bytes using joblib.

    Parameters
    ----------
    model:
        A fitted scikit-learn (or compatible) estimator.

    Returns
    -------
    bytes
        The serialized model bytes, suitable for ``st.download_button``.
    """
    buffer = io.BytesIO()
    joblib.dump(model, buffer)
    buffer.seek(0)
    return buffer.read()


def export_metrics_csv(metrics: dict[str, Any]) -> str:
    """Convert a metrics dictionary to a CSV-formatted string.

    Parameters
    ----------
    metrics:
        Dictionary mapping metric names to their values.

    Returns
    -------
    str
        CSV string with columns ``metric`` and ``value``.
    """
    df = pd.DataFrame(
        list(metrics.items()),
        columns=["metric", "value"],
    )
    return df.to_csv(index=False)


def export_experiment_history(experiments: list[dict[str, Any]]) -> str:
    """Convert a list of experiment records to a CSV-formatted string.

    Parameters
    ----------
    experiments:
        List of dictionaries, each representing one experiment run.
        Typical keys include ``model_name``, ``accuracy``, ``f1``, etc.

    Returns
    -------
    str
        CSV string with one row per experiment.
    """
    if not experiments:
        return "No experiments recorded.\n"

    df = pd.DataFrame(experiments)
    return df.to_csv(index=False)
