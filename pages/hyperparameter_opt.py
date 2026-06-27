"""ML Visualizer Lab — Hyperparameter Optimization (Grid Search / Random Search)."""

import time
import streamlit as st
import numpy as np
import pandas as pd

st.header("⚡ Hyperparameter Optimization")

# Guard
if st.session_state.get("X_train") is None:
    st.warning("⚠️ No preprocessed data. Go to **Preprocessing** first.")
    st.stop()

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from utils.metrics import get_classification_metrics, get_regression_metrics

task_type = st.session_state["task_type"]
X_train = st.session_state["X_train"]
X_test = st.session_state["X_test"]
y_train = st.session_state["y_train"]
y_test = st.session_state["y_test"]

if task_type == "classification":
    from models.classifiers import (get_classifier, get_classifier_names,
                                     get_default_params as get_defaults,
                                     get_param_grid as get_grid)
    algo_names = get_classifier_names()
    build_model = get_classifier
else:
    from models.regressors import (get_regressor, get_regressor_names,
                                    get_default_params as get_defaults,
                                    get_param_grid as get_grid)
    algo_names = get_regressor_names()
    build_model = get_regressor

# ─── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.subheader("⚡ Optimization Settings")
algorithm = st.sidebar.selectbox("Algorithm", algo_names, key="hpo_algo")
search_method = st.sidebar.selectbox("Search Method", ["Grid Search", "Random Search"])
n_iter = 50
if search_method == "Random Search":
    n_iter = st.sidebar.slider("n_iter", 10, 200, 50)
cv_folds = st.sidebar.slider("CV folds", 2, 10, 5, key="hpo_cv")

if task_type == "classification":
    scoring_options = ["accuracy", "f1_weighted", "precision_weighted"]
else:
    scoring_options = ["r2", "neg_mean_absolute_error", "neg_mean_squared_error"]
scoring = st.sidebar.selectbox("Scoring", scoring_options, key="hpo_scoring")

# ─── Parameter Grid ───────────────────────────────────────────────────────────
param_grid = get_grid(algorithm)

st.markdown(f"**Algorithm:** {algorithm} &nbsp;|&nbsp; **Method:** {search_method}")

with st.expander("📋 Parameter Grid"):
    for param, values in param_grid.items():
        st.write(f"**{param}**: {values}")

    total = 1
    for v in param_grid.values():
        total *= len(v)
    st.caption(f"Total combinations: {total}")

if not param_grid:
    st.info(f"ℹ️ {algorithm} has no hyperparameters to optimize.")
    st.stop()

# ─── Run Search ───────────────────────────────────────────────────────────────
if st.button("🔍 Start Search", type="primary", use_container_width=True):
    with st.spinner(f"Running {search_method}... This may take a while."):
        try:
            base_model = build_model(algorithm, get_defaults(algorithm))

            start = time.time()
            if search_method == "Grid Search":
                search = GridSearchCV(base_model, param_grid, cv=cv_folds,
                                      scoring=scoring, n_jobs=-1, return_train_score=True)
            else:
                search = RandomizedSearchCV(base_model, param_grid, n_iter=n_iter,
                                            cv=cv_folds, scoring=scoring, n_jobs=-1,
                                            random_state=42, return_train_score=True)
            search.fit(X_train, y_train)
            elapsed = time.time() - start

            st.session_state["hpo_search"] = search
            st.session_state["hpo_elapsed"] = elapsed
            st.session_state["hpo_algo_name"] = algorithm
        except Exception as e:
            st.error(
                f"❌ Error during hyperparameter search: {e}\n\n"
                "**Suggestions:** Try reducing the parameter grid size, lowering "
                "CV folds, or switching to Random Search for faster execution."
            )

if "hpo_search" not in st.session_state:
    st.info("Click **Start Search** to begin hyperparameter optimization.")
    st.stop()

search = st.session_state["hpo_search"]
elapsed = st.session_state["hpo_elapsed"]
algo_searched = st.session_state["hpo_algo_name"]

# ─── Results ──────────────────────────────────────────────────────────────────
try:
    st.subheader("🏆 Best Parameters")

    col1, col2 = st.columns(2)
    col1.metric("Best CV Score", f"{search.best_score_:.4f}")
    col2.metric("Search Time", f"{elapsed:.1f}s")

    st.json(search.best_params_)

    # Before vs After
    st.subheader("📊 Before vs After")
    default_params = get_defaults(algo_searched)
    default_model = build_model(algo_searched, default_params)
    default_model.fit(X_train, y_train)
    y_pred_default = default_model.predict(X_test)

    optimized_model = search.best_estimator_
    y_pred_opt = optimized_model.predict(X_test)

    if task_type == "classification":
        m_default = get_classification_metrics(y_test, y_pred_default)
        m_opt = get_classification_metrics(y_test, y_pred_opt)
        key_metric = "accuracy"
    else:
        m_default = get_regression_metrics(y_test, y_pred_default)
        m_opt = get_regression_metrics(y_test, y_pred_opt)
        key_metric = "r2"

    comparison_data = {
        "Setting": ["Default", "Optimized"],
    }
    for metric_name in m_default.keys():
        val_d = m_default[metric_name]
        val_o = m_opt[metric_name]
        comparison_data[metric_name] = [
            f"{val_d:.4f}" if val_d is not None else "N/A",
            f"{val_o:.4f}" if val_o is not None else "N/A",
        ]

    comp_df = pd.DataFrame(comparison_data)
    st.dataframe(comp_df, use_container_width=True)

    # Improvement
    if m_default[key_metric] is not None and m_opt[key_metric] is not None:
        delta = m_opt[key_metric] - m_default[key_metric]
        st.metric(f"Improvement ({key_metric})", f"{delta:+.4f}")

    # Top parameter combinations
    st.subheader("📋 Top 10 Combinations")
    cv_results = pd.DataFrame(search.cv_results_)
    cols_to_show = [c for c in cv_results.columns if c.startswith("param_") or c in
                    ("mean_test_score", "std_test_score", "rank_test_score")]
    top10 = cv_results[cols_to_show].sort_values("rank_test_score").head(10)
    st.dataframe(top10.style.format(precision=4), use_container_width=True)
except Exception as e:
    st.error(
        f"❌ Error displaying optimization results: {e}\n\n"
        "**Suggestions:** Try running the search again with different settings."
    )

# Apply best model
if st.button("✅ Apply Best Model", use_container_width=True):
    try:
        optimized_model = search.best_estimator_
        st.session_state["model"] = optimized_model
        st.session_state["model_name"] = f"{algo_searched} (Optimized)"
        y_pred = optimized_model.predict(X_test)
        st.session_state["y_pred"] = y_pred
        if hasattr(optimized_model, "predict_proba"):
            st.session_state["y_proba"] = optimized_model.predict_proba(X_test)
        st.success(f"✅ Applied optimized {algo_searched} as the current model!")
    except Exception as e:
        st.error(f"❌ Error applying optimized model: {e}")
