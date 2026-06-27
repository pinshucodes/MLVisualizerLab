"""ML Visualizer Lab — Explainability (SHAP integration)."""

import streamlit as st
import numpy as np
import pandas as pd

st.header("🧠 Explainability")

# Guard
if st.session_state.get("model") is None:
    st.warning("⚠️ No model trained yet. Go to **Training** first.")
    st.stop()

# Try SHAP import
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# Try streamlit-shap
try:
    from streamlit_shap import st_shap
    ST_SHAP_AVAILABLE = True
except ImportError:
    ST_SHAP_AVAILABLE = False

model = st.session_state["model"]
model_name = st.session_state["model_name"]
X_train = st.session_state["X_train"]
X_test = st.session_state["X_test"]
feature_names = st.session_state["feature_names"]
task_type = st.session_state["task_type"]

if not SHAP_AVAILABLE:
    st.warning(
        "⚠️ **SHAP is not installed.** Install it with:\n\n"
        "```bash\npip install shap\n```\n\n"
        "Falling back to basic feature importance."
    )
    from utils.plots import plot_feature_importance
    try:
        if hasattr(model, "feature_importances_") or hasattr(model, "coef_"):
            fig = plot_feature_importance(model, feature_names)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(f"ℹ️ {model_name} does not expose feature importances.")
    except Exception as e:
        st.error(
            f"❌ **Error**: {str(e)}\n\n"
            "Please check your data and try again."
        )
    st.stop()

# ─── SHAP Computation ─────────────────────────────────────────────────────────
st.markdown(f"**Model:** {model_name}")

# Subsample for performance
max_samples = st.sidebar.slider("Max samples for SHAP", 50, 1000, 200, step=50)
X_shap = X_test[:min(max_samples, len(X_test))]
X_bg = X_train[:min(100, len(X_train))]

X_display = pd.DataFrame(X_shap, columns=feature_names)

@st.cache_data(show_spinner=False)
def compute_shap(_model, _X_shap, _X_bg, _model_name):
    """Compute SHAP values with auto-selected explainer."""
    tree_models = ("DecisionTree", "RandomForest", "GradientBoosting",
                   "XGB", "LGBM", "ExtraTrees")
    linear_models = ("LogisticRegression", "LinearRegression", "Ridge", "Lasso",
                     "SGD", "Perceptron")

    cls_name = type(_model).__name__

    if any(t in cls_name for t in tree_models):
        explainer = shap.TreeExplainer(_model)
        shap_values = explainer.shap_values(_X_shap)
    elif any(t in cls_name for t in linear_models):
        try:
            explainer = shap.LinearExplainer(_model, _X_bg)
            shap_values = explainer.shap_values(_X_shap)
        except Exception:
            explainer = shap.KernelExplainer(_model.predict, _X_bg)
            shap_values = explainer.shap_values(_X_shap)
    else:
        predict_fn = _model.predict
        explainer = shap.KernelExplainer(predict_fn, _X_bg)
        shap_values = explainer.shap_values(_X_shap)

    return explainer, shap_values


with st.spinner("Computing SHAP values... This may take a moment."):
    try:
        explainer, shap_values = compute_shap(model, X_shap, X_bg, model_name)
    except Exception as e:
        st.error(
            f"❌ **Error**: {str(e)}\n\n"
            "SHAP analysis failed. Try a different model or smaller dataset."
        )
        st.stop()

# Handle multiclass: shap_values may be a list of arrays
if isinstance(shap_values, list):
    shap_values_for_summary = shap_values
    shap_values_single = shap_values[0]  # first class
    is_multiclass = True
else:
    shap_values_for_summary = shap_values
    shap_values_single = shap_values
    is_multiclass = False

# ─── SHAP Tabs ─────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Summary Plot", "📈 Feature Importance", "💧 Waterfall", "⚡ Force Plot"]
)

with tab1:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    st.subheader("SHAP Summary Plot")
    try:
        fig, ax = plt.subplots(figsize=(12, 8))
        plt.sca(ax)
        shap.summary_plot(shap_values_for_summary, X_display, show=False, plot_size=None)
        fig = plt.gcf()
        fig.patch.set_facecolor("#0E1117")
        for a in fig.axes:
            a.set_facecolor("#0E1117")
            a.tick_params(colors="white")
            a.xaxis.label.set_color("white")
            a.yaxis.label.set_color("white")
            a.title.set_color("white")
        st.pyplot(fig, use_container_width=True)
        plt.close("all")
    except Exception as e:
        st.error(
            f"❌ **Error**: {str(e)}\n\n"
            "SHAP analysis failed. Try a different model or smaller dataset."
        )

with tab2:
    st.subheader("SHAP Feature Importance")
    try:
        if is_multiclass:
            mean_abs = np.abs(np.array(shap_values_for_summary)).mean(axis=(0, 1))
        else:
            mean_abs = np.abs(shap_values_single).mean(axis=0)

        import plotly.graph_objects as go
        idx = np.argsort(mean_abs)[::-1]
        sorted_names = [feature_names[i] for i in idx]
        sorted_vals = mean_abs[idx]

        fig = go.Figure(go.Bar(
            y=sorted_names[::-1], x=sorted_vals[::-1], orientation="h",
            marker=dict(color=sorted_vals[::-1],
                        colorscale=[[0, "#06B6D4"], [1, "#7C3AED"]]),
        ))
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)",
                          title="Mean |SHAP Value|", height=max(400, len(feature_names) * 25),
                          xaxis_title="Mean |SHAP|", yaxis_title="Feature")
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(
            f"❌ **Error**: {str(e)}\n\n"
            "SHAP analysis failed. Try a different model or smaller dataset."
        )

with tab3:
    st.subheader("SHAP Waterfall Plot")
    sample_idx = st.slider("Select test sample", 0, len(X_shap) - 1, 0)

    try:
        if is_multiclass:
            class_idx = st.selectbox("Class to explain", range(len(shap_values_for_summary)), index=0)
            sv = shap_values_for_summary[class_idx][sample_idx]
            base = explainer.expected_value[class_idx] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
        else:
            sv = shap_values_single[sample_idx]
            base = explainer.expected_value
            if isinstance(base, (list, np.ndarray)):
                base = base[0]

        explanation = shap.Explanation(
            values=sv, base_values=base,
            data=X_shap[sample_idx], feature_names=feature_names,
        )

        fig_w, ax_w = plt.subplots(figsize=(12, 8))
        plt.sca(ax_w)
        shap.plots.waterfall(explanation, show=False)
        fig_w = plt.gcf()
        fig_w.patch.set_facecolor("#0E1117")
        for a in fig_w.axes:
            a.set_facecolor("#0E1117")
            a.tick_params(colors="white")
            a.xaxis.label.set_color("white")
            a.yaxis.label.set_color("white")
        st.pyplot(fig_w, use_container_width=True)
        plt.close("all")
    except Exception as e:
        st.error(
            f"❌ **Error**: {str(e)}\n\n"
            "SHAP analysis failed. Try a different model or smaller dataset."
        )

with tab4:
    st.subheader("SHAP Force Plot")
    if not ST_SHAP_AVAILABLE:
        st.warning(
            "⚠️ `streamlit-shap` not installed. Install with:\n\n"
            "```bash\npip install streamlit-shap\n```"
        )
    else:
        try:
            sample_idx_f = st.slider("Sample for force plot", 0, len(X_shap) - 1, 0,
                                     key="force_sample")
            if is_multiclass:
                sv_f = shap_values_for_summary[0][sample_idx_f]
                base_f = explainer.expected_value[0] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
            else:
                sv_f = shap_values_single[sample_idx_f]
                base_f = explainer.expected_value
                if isinstance(base_f, (list, np.ndarray)):
                    base_f = base_f[0]

            force = shap.force_plot(base_f, sv_f, X_display.iloc[sample_idx_f])
            st_shap(force, height=200)
        except Exception as e:
            st.error(
                f"❌ **Error**: {str(e)}\n\n"
                "SHAP analysis failed. Try a different model or smaller dataset."
            )
