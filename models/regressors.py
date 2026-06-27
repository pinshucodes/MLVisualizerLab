"""Regression model factories for ML Visualizer Lab.

Provides factory functions to create, configure, and retrieve metadata for
regression algorithms supported by the platform.
"""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor

# ---------------------------------------------------------------------------
# Optional boosting libraries
# ---------------------------------------------------------------------------
try:
    from xgboost import XGBRegressor
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

try:
    from lightgbm import LGBMRegressor
    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGRESSORS: dict[str, type] = {
    "Linear Regression": LinearRegression,
    "Ridge Regression": Ridge,
    "Lasso Regression": Lasso,
    "Decision Tree Regressor": DecisionTreeRegressor,
    "Random Forest Regressor": RandomForestRegressor,
    "Gradient Boosting Regressor": GradientBoostingRegressor,
}

if _HAS_XGB:
    _REGRESSORS["XGBoost Regressor"] = XGBRegressor
if _HAS_LGB:
    _REGRESSORS["LightGBM Regressor"] = LGBMRegressor


_DEFAULT_PARAMS: dict[str, dict[str, Any]] = {
    "Linear Regression": {},
    "Ridge Regression": {"alpha": 1.0},
    "Lasso Regression": {"alpha": 1.0},
    "Decision Tree Regressor": {
        "max_depth": 5,
        "min_samples_split": 2,
    },
    "Random Forest Regressor": {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 2,
    },
    "Gradient Boosting Regressor": {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 3,
    },
}

if _HAS_XGB:
    _DEFAULT_PARAMS["XGBoost Regressor"] = {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 3,
    }
if _HAS_LGB:
    _DEFAULT_PARAMS["LightGBM Regressor"] = {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": -1,
        "verbose": -1,
    }


_PARAM_GRIDS: dict[str, dict[str, list]] = {
    "Linear Regression": {},
    "Ridge Regression": {
        "alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
    },
    "Lasso Regression": {
        "alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
    },
    "Decision Tree Regressor": {
        "max_depth": [3, 5, 10, 15, 20, None],
        "min_samples_split": [2, 5, 10],
    },
    "Random Forest Regressor": {
        "n_estimators": [50, 100, 200, 300],
        "max_depth": [3, 5, 10, 15, 20, None],
        "min_samples_split": [2, 5, 10],
    },
    "Gradient Boosting Regressor": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "max_depth": [3, 5, 7],
    },
}

if _HAS_XGB:
    _PARAM_GRIDS["XGBoost Regressor"] = {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "max_depth": [3, 5, 7],
    }
if _HAS_LGB:
    _PARAM_GRIDS["LightGBM Regressor"] = {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "max_depth": [-1, 3, 5, 7],
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_regressor_names() -> list[str]:
    """Return the list of available regressor names."""
    return list(_REGRESSORS.keys())


def get_regressor(name: str, params: dict[str, Any]) -> Any:
    """Create and return a configured regressor instance."""
    if name not in _REGRESSORS:
        raise ValueError(f"Unknown regressor '{name}'. Available: {get_regressor_names()}")

    cls = _REGRESSORS[name]
    import inspect
    valid_keys = set(inspect.signature(cls.__init__).parameters.keys()) - {"self"}
    filtered = {k: v for k, v in params.items() if k in valid_keys}
    return cls(**filtered)


def get_default_params(name: str) -> dict[str, Any]:
    """Return default hyperparameters for *name*."""
    if name not in _DEFAULT_PARAMS:
        raise ValueError(f"Unknown regressor '{name}'.")
    return dict(_DEFAULT_PARAMS[name])


def get_param_grid(name: str) -> dict[str, list]:
    """Return the hyperparameter search grid for *name*."""
    if name not in _PARAM_GRIDS:
        raise ValueError(f"Unknown regressor '{name}'.")
    return dict(_PARAM_GRIDS[name])
