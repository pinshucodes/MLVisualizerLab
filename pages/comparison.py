"""ML Visualizer Lab — Model Comparison Dashboard."""

import time
import streamlit as st
import numpy as np
import pandas as pd

st.header("⚔️ Model Comparison")

# Guard
if st.session_state.get("X_train") is None:
    st.warning("⚠️ No preprocessed data. Go to **Preprocessing** first.")
    st.stop()

from utils.metrics import get_classification_metrics, get_regression_metrics
from utils.plots import plot_comparison_bar, plot_radar
from models.classifiers import get_classifier, get_classifier_names, get_default_params as get_clf_defaults
from models.regressors import get_regressor, get_regressor_names, get_default_params as get_reg_defaults

task_type = st.session_state["task_type"]
X_train = st.session_state["X_train"]
X_test = st.session_state["X_test"]
y_train = st.session_state["y_train"]
y_test = st.session_state["y_test"]

# ─── Algorithm Selection ──────────────────────────────────────────────────────
if task_type == "classification":
    available = get_classifier_names()
    default_sel = available[:3] if len(available) >= 3 else available
else:
    available = get_regressor_names()
    default_sel = available[:3] if len(available) >= 3 else available

selected = st.multiselect("Select algorithms to compare", available, default=default_sel)

if not selected:
    st.info("Select at least one algorithm to compare.")
    st.stop()

# ─── Train All ────────────────────────────────────────────────────────────────
if st.button("🚀 Train & Compare All", type="primary", use_container_width=True):
    results = []
    progress = st.progress(0, text="Training models...")

    for i, algo_name in enumerate(selected):
        progress.progress((i) / len(selected), text=f"Training {algo_name}...")
        try:
            if task_type == "classification":
                params = get_clf_defaults(algo_name)
                model = get_classifier(algo_name, params)
            else:
                params = get_reg_defaults(algo_name)
                model = get_regressor(algo_name, params)

            start = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - start

            y_pred = model.predict(X_test)

            if task_type == "classification":
                y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
                metrics = get_classification_metrics(y_test, y_pred, y_proba)
                results.append({
                    "Model": algo_name,
                    "Accuracy": round(metrics["accuracy"], 4),
                    "Precision": round(metrics["precision"], 4),
                    "Recall": round(metrics["recall"], 4),
                    "F1": round(metrics["f1"], 4),
                    "ROC AUC": round(metrics["roc_auc"], 4) if metrics["roc_auc"] else "N/A",
                    "Time (s)": round(train_time, 3),
                })
            else:
                metrics = get_regression_metrics(y_test, y_pred)
                results.append({
                    "Model": algo_name,
                    "MAE": round(metrics["mae"], 4),
                    "MSE": round(metrics["mse"], 4),
                    "RMSE": round(metrics["rmse"], 4),
                    "R²": round(metrics["r2"], 4),
                    "Time (s)": round(train_time, 3),
                })

        except Exception as e:
            st.warning(f"⚠️ {algo_name} failed: {e}")

    progress.progress(1.0, text="Done!")

    if not results:
        st.error("All models failed to train.")
        st.stop()

    results_df = pd.DataFrame(results)
    st.session_state["comparison_results"] = results_df

# ─── Display Results ──────────────────────────────────────────────────────────
if "comparison_results" in st.session_state:
    results_df = st.session_state["comparison_results"]

    st.subheader("📊 Comparison Table")

    # Highlight best
    if task_type == "classification":
        metric_cols = ["Accuracy", "Precision", "Recall", "F1"]
        best_col = "Accuracy"
    else:
        metric_cols = ["R²"]
        best_col = "R²"

    st.dataframe(
        results_df.style.highlight_max(subset=metric_cols, color="rgba(124,58,237,0.3)")
            .format(precision=4),
        use_container_width=True,
    )

    # Best model
    best_idx = results_df[best_col].astype(float, errors="ignore").idxmax()
    best_model = results_df.loc[best_idx, "Model"]
    best_score = results_df.loc[best_idx, best_col]

    col1, col2 = st.columns(2)
    with col1:
        st.metric("🏆 Best Model", best_model)
    with col2:
        st.metric(f"Best {best_col}", f"{best_score}")

    # Charts
    st.subheader("📈 Visual Comparison")
    tab_bar, tab_radar, tab_time = st.tabs(["Bar Chart", "Radar Chart", "Training Time"])

    with tab_bar:
        fig = plot_comparison_bar(results_df, metric_cols)
        st.plotly_chart(fig, use_container_width=True)

    with tab_radar:
        # Radar only with 0-1 metrics
        radar_metrics = [m for m in metric_cols if m in results_df.columns]
        if radar_metrics:
            radar_df = results_df.copy()
            # Normalize for radar if regression
            if task_type == "regression":
                for m in radar_metrics:
                    vals = radar_df[m].astype(float)
                    if vals.max() != vals.min():
                        radar_df[m] = (vals - vals.min()) / (vals.max() - vals.min())
            fig = plot_radar(radar_df, radar_metrics)
            st.plotly_chart(fig, use_container_width=True)

    with tab_time:
        import plotly.graph_objects as go
        fig = go.Figure(go.Bar(
            x=results_df["Model"], y=results_df["Time (s)"],
            marker_color="#7C3AED",
        ))
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)",
                          title="Training Time", yaxis_title="Seconds")
        st.plotly_chart(fig, use_container_width=True)
