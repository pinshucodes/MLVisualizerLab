"""ML Visualizer Lab — Learning Curves."""

import streamlit as st
import numpy as np

st.header("📚 Learning Curves")

# Guard
if st.session_state.get("X_train") is None or st.session_state.get("model") is None:
    st.warning("⚠️ Train a model first on the **Training** page.")
    st.stop()

from sklearn.base import clone
from sklearn.model_selection import learning_curve
from utils.plots import plot_learning_curve_fig
import pandas as pd

model = st.session_state["model"]
model_name = st.session_state["model_name"]
task_type = st.session_state["task_type"]
X_train = st.session_state["X_train"]
X_test = st.session_state["X_test"]
y_train = st.session_state["y_train"]
y_test = st.session_state["y_test"]

X_full = np.vstack([X_train, X_test])
y_full = np.concatenate([y_train, y_test])

# ─── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.subheader("📚 Learning Curve Settings")
n_points = st.sidebar.slider("Number of training sizes", 5, 20, 10)
cv_folds = st.sidebar.slider("CV folds", 2, 10, 5, key="lc_cv")

if task_type == "classification":
    scoring_options = ["accuracy", "f1_weighted", "precision_weighted"]
else:
    scoring_options = ["r2", "neg_mean_absolute_error", "neg_mean_squared_error"]

scoring = st.sidebar.selectbox("Scoring Metric", scoring_options, key="lc_scoring")

# ─── Generate ─────────────────────────────────────────────────────────────────
st.markdown(f"**Model:** {model_name} &nbsp;|&nbsp; **Scoring:** `{scoring}`")

if st.button("📈 Generate Learning Curves", type="primary", use_container_width=True):
    with st.spinner("Generating learning curves... This may take a moment."):
        try:
            model_clone = clone(model)
            train_sizes, train_scores, val_scores = learning_curve(
                model_clone, X_full, y_full,
                cv=cv_folds,
                train_sizes=np.linspace(0.1, 1.0, n_points),
                scoring=scoring,
                n_jobs=-1,
            )

            st.session_state["lc_train_sizes"] = train_sizes
            st.session_state["lc_train_scores"] = train_scores
            st.session_state["lc_val_scores"] = val_scores
        except Exception as e:
            st.error(
                f"❌ Error generating learning curves: {e}\n\n"
                "**Suggestions:** Try reducing the number of training sizes or CV folds. "
                "The dataset may be too small for the current settings."
            )

if "lc_train_sizes" not in st.session_state:
    st.info("Click **Generate Learning Curves** to start.")
    st.stop()

train_sizes = st.session_state["lc_train_sizes"]
train_scores = st.session_state["lc_train_scores"]
val_scores = st.session_state["lc_val_scores"]

# ─── Plot ─────────────────────────────────────────────────────────────────────
try:
    fig = plot_learning_curve_fig(train_sizes, train_scores, val_scores)
    st.plotly_chart(fig, use_container_width=True)
except Exception as e:
    st.error(f"❌ Error rendering learning curve plot: {e}")

# ─── Diagnostic ───────────────────────────────────────────────────────────────
try:
    train_final = train_scores[-1].mean()
    val_final = val_scores[-1].mean()
    gap = abs(train_final - val_final)

    st.subheader("🔍 Diagnostic")
    if train_final > 0.9 and gap > 0.15:
        st.warning("⚠️ **Possible Overfitting**: Training score is high but validation score is "
                   "significantly lower. Consider reducing model complexity, adding regularization, "
                   "or getting more data.")
    elif train_final < 0.6 and val_final < 0.6:
        st.warning("⚠️ **Possible Underfitting**: Both training and validation scores are low. "
                   "Consider increasing model complexity, adding more features, or tuning hyperparameters.")
    elif gap < 0.05:
        st.success("✅ **Good Generalization**: Training and validation scores are close and converging. "
                   "The model generalizes well.")
    else:
        st.info("ℹ️ Model shows moderate fit. Consider tuning hyperparameters for improvement.")
except Exception as e:
    st.error(f"❌ Error computing diagnostic: {e}")

# ─── Raw Values ───────────────────────────────────────────────────────────────
with st.expander("📋 Raw Values"):
    try:
        lc_df = pd.DataFrame({
            "Training Size": train_sizes,
            "Train Mean": train_scores.mean(axis=1),
            "Train Std": train_scores.std(axis=1),
            "Val Mean": val_scores.mean(axis=1),
            "Val Std": val_scores.std(axis=1),
        })
        st.dataframe(lc_df.style.format(precision=4), use_container_width=True)
    except Exception as e:
        st.error(f"❌ Error displaying raw values: {e}")
