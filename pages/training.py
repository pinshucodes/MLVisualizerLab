"""
ML Visualizer Lab — Training
Select algorithms, tune hyperparameters, train models, and log experiments.
"""

import time
import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

# ─── Guard ─────────────────────────────────────────────────────────────────────
if st.session_state.get("X_train") is None:
    st.warning("⚠️ No preprocessed data found. Complete **Preprocessing** first.")
    st.stop()

task = st.session_state["task_type"]
X_train = st.session_state["X_train"]
X_test = st.session_state["X_test"]
y_train = st.session_state["y_train"]
y_test = st.session_state["y_test"]

st.markdown("## 🎯 Model Training")
st.caption(f"Task: **{task}** · Train {X_train.shape[0]} / Test {X_test.shape[0]} samples · {X_train.shape[1]} features")

# ─── Algorithm registry ───────────────────────────────────────────────────────
_CLF_ALGOS = {
    "K-Nearest Neighbors": {"icon": "👥", "desc": "Instance-based learning, classifies by nearest neighbors"},
    "Logistic Regression": {"icon": "📐", "desc": "Linear model for binary/multiclass classification"},
    "SVM": {"icon": "✂️", "desc": "Finds optimal hyperplane for classification"},
    "Decision Tree": {"icon": "🌳", "desc": "Tree-based splits for interpretable decisions"},
    "Random Forest": {"icon": "🌲", "desc": "Ensemble of decision trees for robust predictions"},
    "Gradient Boosting": {"icon": "🚀", "desc": "Sequential boosting for high accuracy"},
}

_REG_ALGOS = {
    "K-Nearest Neighbors": {"icon": "👥", "desc": "Predicts by averaging nearest neighbors"},
    "Linear Regression": {"icon": "📏", "desc": "Basic linear model, no regularization"},
    "Ridge": {"icon": "📐", "desc": "L2-regularized linear regression"},
    "Lasso": {"icon": "🎯", "desc": "L1-regularized, promotes sparsity"},
    "ElasticNet": {"icon": "🔗", "desc": "Combined L1+L2 regularization"},
    "SVR": {"icon": "✂️", "desc": "Support Vector Regression"},
    "Decision Tree": {"icon": "🌳", "desc": "Tree-based regression"},
    "Random Forest": {"icon": "🌲", "desc": "Ensemble regression with bagging"},
    "Gradient Boosting": {"icon": "🚀", "desc": "Sequential boosting for regression"},
}

algos = _CLF_ALGOS if task == "classification" else _REG_ALGOS
algo_names = list(algos.keys())

# ─── Step 1: Algorithm Selection (MAIN AREA) ──────────────────────────────────
st.markdown("### 1️⃣ Choose Algorithm")

# Show algorithm cards in a grid
ncols = 3
rows = [algo_names[i:i+ncols] for i in range(0, len(algo_names), ncols)]

# Use radio buttons styled as cards for clear selection
algo = st.radio(
    "Select an algorithm to train:",
    algo_names,
    index=0,
    horizontal=True,
    format_func=lambda x: f"{algos[x]['icon']} {x}",
    help="Choose which ML algorithm to train on your data.",
)

st.caption(f"💡 {algos[algo]['desc']}")

st.markdown("---")

# ─── Step 2: Hyperparameters ──────────────────────────────────────────────────
st.markdown("### 2️⃣ Tune Hyperparameters")

params: dict = {}
hp1, hp2 = st.columns(2)

try:
    if task == "classification":
        if algo == "K-Nearest Neighbors":
            with hp1:
                params["n_neighbors"] = st.slider("n_neighbors", 1, 50, 5)
                params["weights"] = st.selectbox("weights", ["uniform", "distance"])
            with hp2:
                params["metric"] = st.selectbox("metric", ["euclidean", "manhattan", "minkowski"])
        elif algo == "Logistic Regression":
            with hp1:
                params["C"] = st.slider("C (regularization)", 0.01, 100.0, 1.0, step=0.01)
                params["penalty"] = st.selectbox("penalty", ["l2", "l1", "elasticnet", "none"])
            with hp2:
                params["solver"] = st.selectbox("solver", ["lbfgs", "liblinear", "saga"])
                params["max_iter"] = st.slider("max_iter", 100, 5000, 1000, step=100)
        elif algo == "SVM":
            with hp1:
                params["C"] = st.slider("C", 0.01, 100.0, 1.0, step=0.01)
                params["kernel"] = st.selectbox("kernel", ["rbf", "linear", "poly", "sigmoid"])
            with hp2:
                params["gamma"] = st.selectbox("gamma", ["scale", "auto"])
        elif algo == "Decision Tree":
            with hp1:
                params["max_depth"] = st.slider("max_depth", 1, 50, 5)
                params["criterion"] = st.selectbox("criterion", ["gini", "entropy"])
            with hp2:
                params["min_samples_split"] = st.slider("min_samples_split", 2, 20, 2)
        elif algo == "Random Forest":
            with hp1:
                params["n_estimators"] = st.slider("n_estimators", 10, 500, 100, step=10)
                params["max_depth"] = st.slider("max_depth", 1, 50, 10)
            with hp2:
                params["min_samples_split"] = st.slider("min_samples_split", 2, 20, 2)
        elif algo == "Gradient Boosting":
            with hp1:
                params["n_estimators"] = st.slider("n_estimators", 10, 500, 100, step=10)
                params["learning_rate"] = st.slider("learning_rate", 0.001, 1.0, 0.1, step=0.001)
            with hp2:
                params["max_depth"] = st.slider("max_depth", 1, 20, 3)
    else:  # regression
        if algo == "K-Nearest Neighbors":
            with hp1:
                params["n_neighbors"] = st.slider("n_neighbors", 1, 50, 5)
                params["weights"] = st.selectbox("weights", ["uniform", "distance"])
            with hp2:
                params["metric"] = st.selectbox("metric", ["euclidean", "manhattan", "minkowski"])
        elif algo == "Linear Regression":
            st.caption("No tuneable hyperparameters for Linear Regression.")
        elif algo == "Ridge":
            with hp1:
                params["alpha"] = st.slider("alpha", 0.01, 100.0, 1.0, step=0.01)
        elif algo == "Lasso":
            with hp1:
                params["alpha"] = st.slider("alpha", 0.001, 10.0, 1.0, step=0.001)
            with hp2:
                params["max_iter"] = st.slider("max_iter", 100, 10000, 1000, step=100)
        elif algo == "ElasticNet":
            with hp1:
                params["alpha"] = st.slider("alpha", 0.001, 10.0, 1.0, step=0.001)
                params["l1_ratio"] = st.slider("l1_ratio", 0.0, 1.0, 0.5, step=0.05)
            with hp2:
                params["max_iter"] = st.slider("max_iter", 100, 10000, 1000, step=100)
        elif algo == "SVR":
            with hp1:
                params["C"] = st.slider("C", 0.01, 100.0, 1.0, step=0.01)
                params["kernel"] = st.selectbox("kernel", ["rbf", "linear", "poly", "sigmoid"])
            with hp2:
                params["gamma"] = st.selectbox("gamma", ["scale", "auto"])
        elif algo == "Decision Tree":
            with hp1:
                params["max_depth"] = st.slider("max_depth", 1, 50, 5)
                params["criterion"] = st.selectbox("criterion", ["squared_error", "friedman_mse", "absolute_error"])
            with hp2:
                params["min_samples_split"] = st.slider("min_samples_split", 2, 20, 2)
        elif algo == "Random Forest":
            with hp1:
                params["n_estimators"] = st.slider("n_estimators", 10, 500, 100, step=10)
                params["max_depth"] = st.slider("max_depth", 1, 50, 10)
            with hp2:
                params["min_samples_split"] = st.slider("min_samples_split", 2, 20, 2)
        elif algo == "Gradient Boosting":
            with hp1:
                params["n_estimators"] = st.slider("n_estimators", 10, 500, 100, step=10)
                params["learning_rate"] = st.slider("learning_rate", 0.001, 1.0, 0.1, step=0.001)
            with hp2:
                params["max_depth"] = st.slider("max_depth", 1, 20, 3)
except Exception as e:
    st.error(f"❌ Error rendering hyperparameters: {e}")

# ─── Model builder ────────────────────────────────────────────────────────────
def _build_model(name: str, task_type: str, hp: dict):
    """Return an sklearn estimator configured with *hp*."""
    if task_type == "classification":
        mapping = {
            "K-Nearest Neighbors": KNeighborsClassifier,
            "Logistic Regression": LogisticRegression,
            "SVM": SVC,
            "Decision Tree": DecisionTreeClassifier,
            "Random Forest": RandomForestClassifier,
            "Gradient Boosting": GradientBoostingClassifier,
        }
    else:
        mapping = {
            "K-Nearest Neighbors": KNeighborsRegressor,
            "Linear Regression": LinearRegression,
            "Ridge": Ridge,
            "Lasso": Lasso,
            "ElasticNet": ElasticNet,
            "SVR": SVR,
            "Decision Tree": DecisionTreeRegressor,
            "Random Forest": RandomForestRegressor,
            "Gradient Boosting": GradientBoostingRegressor,
        }
    cls = mapping[name]
    if name == "SVM" and task_type == "classification":
        hp = {**hp, "probability": True}
    valid = cls().get_params()
    filtered = {k: v for k, v in hp.items() if k in valid}
    return cls(**filtered)

# ─── Step 3: Train ────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 3️⃣ Train Model")

train_btn = st.button(f"🚀 Train {algo}", type="primary", use_container_width=True)

if train_btn:
    try:
        with st.spinner(f"Training **{algo}**…"):
            model = _build_model(algo, task, params)

            t0 = time.perf_counter()
            model.fit(X_train, y_train)
            train_time = time.perf_counter() - t0

            y_pred = model.predict(X_test)
            y_proba = None
            if task == "classification" and hasattr(model, "predict_proba"):
                try:
                    y_proba = model.predict_proba(X_test)
                except Exception:
                    y_proba = None

        # Store in session state
        st.session_state["model"] = model
        st.session_state["model_name"] = algo
        st.session_state["y_pred"] = y_pred
        st.session_state["y_proba"] = y_proba

        st.success(f"✅ **{algo}** trained in **{train_time:.3f}s**")

        # ── Metrics ───────────────────────────────────────────────────────────
        st.markdown("### 📊 Quick Metrics")

        if task == "classification":
            avg = "weighted" if len(np.unique(y_test)) > 2 else "binary"
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average=avg, zero_division=0)
            rec = recall_score(y_test, y_pred, average=avg, zero_division=0)
            f1 = f1_score(y_test, y_pred, average=avg, zero_division=0)

            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Accuracy", f"{acc:.4f}")
            mc2.metric("Precision", f"{prec:.4f}")
            mc3.metric("Recall", f"{rec:.4f}")
            mc4.metric("F1 Score", f"{f1:.4f}")

            metrics_dict = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

            if y_proba is not None:
                try:
                    if y_proba.shape[1] == 2:
                        auc = roc_auc_score(y_test, y_proba[:, 1])
                    else:
                        auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="weighted")
                    st.metric("ROC AUC", f"{auc:.4f}")
                    metrics_dict["roc_auc"] = auc
                except Exception:
                    pass
        else:
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)

            mr1, mr2, mr3, mr4 = st.columns(4)
            mr1.metric("MAE", f"{mae:.4f}")
            mr2.metric("MSE", f"{mse:.4f}")
            mr3.metric("RMSE", f"{rmse:.4f}")
            mr4.metric("R²", f"{r2:.4f}")

            metrics_dict = {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2}

        # ── Log experiment ────────────────────────────────────────────────────
        experiment = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "dataset": st.session_state.get("dataset_name", "Unknown"),
            "algorithm": algo,
            "task_type": task,
            "hyperparameters": params.copy(),
            "metrics": metrics_dict,
            "training_time": round(train_time, 4),
            "train_size": X_train.shape[0],
            "test_size": X_test.shape[0],
        }
        if "experiments" not in st.session_state or st.session_state["experiments"] is None:
            st.session_state["experiments"] = []
        st.session_state["experiments"].append(experiment)
        st.caption(f"💾 Experiment #{len(st.session_state['experiments'])} logged.")

    except Exception as e:
        st.error(
            f"❌ **Training failed**: {str(e)}\n\n"
            "**Suggestions:**\n"
            "- Try a different algorithm\n"
            "- Adjust the hyperparameters\n"
            "- Check that your data was preprocessed correctly"
        )
