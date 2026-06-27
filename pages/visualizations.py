"""ML Visualizer Lab — Visualizations (Decision Boundary & Feature Importance)."""

import streamlit as st
import numpy as np

st.header("🎨 Visualizations")

# Guard
if st.session_state.get("model") is None:
    st.warning("⚠️ No model trained yet. Go to **Training** to train a model first.")
    st.stop()

model = st.session_state["model"]
model_name = st.session_state["model_name"]
X_train = st.session_state["X_train"]
y_train = st.session_state["y_train"]
feature_names = st.session_state["feature_names"]
task_type = st.session_state["task_type"]

tab1, tab2 = st.tabs(["🗺️ Decision Boundary", "📊 Feature Importance"])

# ─── Decision Boundary ────────────────────────────────────────────────────────
with tab1:
    if task_type != "classification":
        st.info("ℹ️ Decision boundary visualization is only available for classification tasks.")
    else:
        from utils.plots import plot_decision_boundary

        st.markdown(f"**Model:** {model_name} &nbsp;|&nbsp; **Features:** {X_train.shape[1]}")
        if X_train.shape[1] > 2:
            st.caption("Features reduced to 2D using PCA for visualization.")

        with st.spinner("Generating decision boundary..."):
            try:
                fig = plot_decision_boundary(model, X_train, y_train, feature_names)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Could not generate decision boundary: {e}")

# ─── Feature Importance ───────────────────────────────────────────────────────
with tab2:
    from utils.plots import plot_feature_importance

    has_importance = hasattr(model, "feature_importances_") or hasattr(model, "coef_")

    if not has_importance:
        st.info(
            f"ℹ️ **{model_name}** does not expose feature importances.\n\n"
            "Try a tree-based model (Decision Tree, Random Forest, Gradient Boosting) "
            "or a linear model (Logistic Regression, Ridge, Lasso)."
        )
    else:
        try:
            fig = plot_feature_importance(model, feature_names)
            st.plotly_chart(fig, use_container_width=True)

            # Raw values table
            with st.expander("📋 Raw Values"):
                if hasattr(model, "feature_importances_"):
                    importance = model.feature_importances_
                else:
                    coef = model.coef_
                    importance = np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef)

                import pandas as pd
                imp_df = pd.DataFrame({
                    "Feature": feature_names,
                    "Importance": importance,
                }).sort_values("Importance", ascending=False).reset_index(drop=True)
                st.dataframe(imp_df, use_container_width=True)
        except Exception as e:
            st.error(f"❌ Error computing feature importance: {e}\n\nTry retraining the model or selecting a different algorithm.")
