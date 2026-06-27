"""Clustering model factories for ML Visualizer Lab."""

from __future__ import annotations

from typing import Any

from sklearn.cluster import DBSCAN, KMeans


_MODELS: dict[str, type] = {
    "K-Means": KMeans,
    "DBSCAN": DBSCAN,
}

_DEFAULT_PARAMS: dict[str, dict[str, Any]] = {
    "K-Means": {"n_clusters": 3, "n_init": 10, "random_state": 42},
    "DBSCAN": {"eps": 0.5, "min_samples": 5},
}


def get_clustering_names() -> list[str]:
    """Return available clustering algorithm names."""
    return list(_MODELS.keys())


def get_clustering_model(name: str, params: dict[str, Any]) -> Any:
    """Create and return a configured clustering model."""
    if name not in _MODELS:
        raise ValueError(f"Unknown clustering algorithm '{name}'.")
    cls = _MODELS[name]
    # Always set random_state for K-Means reproducibility
    if name == "K-Means":
        params = {**params, "n_init": 10, "random_state": 42}
    return cls(**params)
