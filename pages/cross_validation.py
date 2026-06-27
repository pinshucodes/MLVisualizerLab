"""ML Visualizer Lab — Cross Validation."""

import streamlit as st
import numpy as np

st.header("📐 Cross Validation")

# Guard
if st.session_state.get("X_train") is None or st.session_state.get("model") is None:
    st.warning("⚠️ Train a model first on the **Training** page.")
    st.stop()

from sklearn.base import clone
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from utils.plots import plot_cv_scores
import pandas as pd

model = st.session_state["model"]
model_name = st.session_state["model_name"]
task_type = st.session_state["task_type"]
X_train = st.session_state["X_train"]
X_test = st.session_state["X_test"]
y_train = st.session_state["y_train"]
y_test = st.session_state["y_test"]

# Full data for proper CV
X_full = np.vstack([X_train, X_test])
y_full = np.concatenate([y_train, y_test])

# ─── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.subheader("📐 CV Settings")

cv_options = ["K-Fold", "Stratified K-Fold"] if task_type == "classification" else ["K-Fold"]
cv_type = st.sidebar.selectbox("CV Type", cv_options)
n_folds = st.sidebar.slider("Number of folds", 2, 20, 5)

if task_type == "classification":
    scoring_options = ["accuracy", "f1_weighted", "precision_weighted", "recall_weighted"]
else:
    scoring_options = ["r2", "neg_mean_absolute_error", "neg_mean_squared_error"]

scoring = st.sidebar.selectbox("Scoring Metric", scoring_options)

# ─── Run CV ───────────────────────────────────────────────────────────────────
st.markdown(f"**Model:** {model_name} &nbsp;|&nbsp; **Scoring:** `{scoring}`")

if st.button("🔄 Run Cross Validation", type="primary", use_container_width=True):
    with st.spinner(f"Running {n_folds}-fold cross validation..."):
        try:
            if cv_type == "Stratified K-Fold":
                cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
            else:
                cv = KFold(n_splits=n_folds, shuffle=True, random_state=42)

            model_clone = clone(model)
            scores = cross_val_score(model_clone, X_full, y_full, cv=cv, scoring=scoring)

            st.session_state["cv_scores"] = scores
            st.session_state["cv_scoring"] = scoring
        except Exception as e:
            st.error(
                f"❌ Error during cross validation: {e}\n\n"
                "**Suggestions:** Try reducing the number of folds (the dataset may be "
                "too small for the selected fold count), or choose a different scoring metric."
            )

if "cv_scores" not in st.session_state:
    st.info("Click **Run Cross Validation** to start.")
    st.stop()

scores = st.session_state["cv_scores"]
scoring_name = st.session_state["cv_scoring"]

# ─── Results ──────────────────────────────────────────────────────────────────
try:
    st.subheader("📊 Results")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Mean Score", f"{scores.mean():.4f}")
    col2.metric("Std Deviation", f"{scores.std():.4f}")
    col3.metric("Min Score", f"{scores.min():.4f}")
    col4.metric("Max Score", f"{scores.max():.4f}")

    # Per-fold table
    st.subheader("Per-Fold Scores")
    fold_df = pd.DataFrame({
        "Fold": [f"Fold {i+1}" for i in range(len(scores))],
        "Score": scores,
    })
    fold_df["Deviation from Mean"] = scores - scores.mean()
    st.dataframe(fold_df.style.format({"Score": "{:.4f}", "Deviation from Mean": "{:+.4f}"}),
                 use_container_width=True)

    # Box plot
    st.subheader("Score Distribution")
    fig = plot_cv_scores(scores)
    st.plotly_chart(fig, use_container_width=True)

    # Histogram
    import plotly.graph_objects as go
    fig_hist = go.Figure(go.Histogram(
        x=scores, nbinsx=max(5, n_folds // 2),
        marker_color="#7C3AED",
    ))
    fig_hist.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                           plot_bgcolor="rgba(0,0,0,0)",
                           title="Score Distribution", xaxis_title="Score",
                           yaxis_title="Count", height=350)
    st.plotly_chart(fig_hist, use_container_width=True)
except Exception as e:
    st.error(f"❌ Error rendering cross validation results: {e}")
