"""Classification model factories for ML Visualizer Lab.

Provides factory functions to create, configure, and retrieve metadata for
classification algorithms supported by the platform.
"""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# ---------------------------------------------------------------------------
# Optional boosting libraries
# ---------------------------------------------------------------------------
try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_CLASSIFIERS: dict[str, type] = {
    "K-Nearest Neighbors": KNeighborsClassifier,
    "Logistic Regression": LogisticRegression,
    "Support Vector Machine": SVC,
    "Decision Tree": DecisionTreeClassifier,
    "Random Forest": RandomForestClassifier,
    "Gradient Boosting": GradientBoostingClassifier,
}

if _HAS_XGB:
    _CLASSIFIERS["XGBoost"] = XGBClassifier
if _HAS_LGB:
    _CLASSIFIERS["LightGBM"] = LGBMClassifier


_DEFAULT_PARAMS: dict[str, dict[str, Any]] = {
    "K-Nearest Neighbors": {
        "n_neighbors": 5,
        "weights": "uniform",
        "metric": "minkowski",
    },
    "Logistic Regression": {
        "C": 1.0,
        "penalty": "l2",
        "solver": "lbfgs",
        "max_iter": 1000,
    },
    "Support Vector Machine": {
        "C": 1.0,
        "kernel": "rbf",
        "gamma": "scale",
        "probability": True,
    },
    "Decision Tree": {
        "max_depth": 5,
        "criterion": "gini",
        "min_samples_split": 2,
    },
    "Random Forest": {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 2,
    },
    "Gradient Boosting": {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 3,
    },
}

if _HAS_XGB:
    _DEFAULT_PARAMS["XGBoost"] = {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 3,
        "use_label_encoder": False,
        "eval_metric": "logloss",
    }
if _HAS_LGB:
    _DEFAULT_PARAMS["LightGBM"] = {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": -1,
        "verbose": -1,
    }


_PARAM_GRIDS: dict[str, dict[str, list]] = {
    "K-Nearest Neighbors": {
        "n_neighbors": [3, 5, 7, 9, 11, 15, 21],
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan", "minkowski"],
    },
    "Logistic Regression": {
        "C": [0.01, 0.1, 1.0, 10.0, 100.0],
        "penalty": ["l2"],
        "solver": ["lbfgs", "liblinear"],
        "max_iter": [1000],
    },
    "Support Vector Machine": {
        "C": [0.1, 1.0, 10.0],
        "kernel": ["rbf", "linear", "poly"],
        "gamma": ["scale", "auto"],
    },
    "Decision Tree": {
        "max_depth": [3, 5, 10, 15, 20, None],
        "criterion": ["gini", "entropy"],
        "min_samples_split": [2, 5, 10],
    },
    "Random Forest": {
        "n_estimators": [50, 100, 200, 300],
        "max_depth": [3, 5, 10, 15, 20, None],
        "min_samples_split": [2, 5, 10],
    },
    "Gradient Boosting": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "max_depth": [3, 5, 7],
    },
}

if _HAS_XGB:
    _PARAM_GRIDS["XGBoost"] = {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "max_depth": [3, 5, 7],
    }
if _HAS_LGB:
    _PARAM_GRIDS["LightGBM"] = {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "max_depth": [-1, 3, 5, 7],
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_classifier_names() -> list[str]:
    """Return the list of available classifier names."""
    return list(_CLASSIFIERS.keys())


def get_classifier(name: str, params: dict[str, Any]) -> Any:
    """Create and return a configured classifier instance.

    Parameters
    ----------
    name:
        Algorithm name (must be in ``get_classifier_names()``).
    params:
        Hyperparameter dict passed to the constructor.

    Returns
    -------
    A scikit-learn compatible estimator.
    """
    if name not in _CLASSIFIERS:
        raise ValueError(f"Unknown classifier '{name}'. Available: {get_classifier_names()}")

    cls = _CLASSIFIERS[name]
    # Filter params to only those accepted by the constructor
    import inspect
    valid_keys = set(inspect.signature(cls.__init__).parameters.keys()) - {"self"}
    filtered = {k: v for k, v in params.items() if k in valid_keys}

    # SVM always needs probability=True for predict_proba
    if name == "Support Vector Machine":
        filtered["probability"] = True

    return cls(**filtered)


def get_default_params(name: str) -> dict[str, Any]:
    """Return default hyperparameters for *name*."""
    if name not in _DEFAULT_PARAMS:
        raise ValueError(f"Unknown classifier '{name}'.")
    return dict(_DEFAULT_PARAMS[name])


def get_param_grid(name: str) -> dict[str, list]:
    """Return the hyperparameter search grid for *name*."""
    if name not in _PARAM_GRIDS:
        raise ValueError(f"Unknown classifier '{name}'.")
    return dict(_PARAM_GRIDS[name])
