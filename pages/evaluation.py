"""
ML Visualizer Lab — Evaluation
Classification and regression evaluation with interactive Plotly charts.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

# ─── Plotly defaults ──────────────────────────────────────────────────────────
_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter"),
)

# ─── Guard ─────────────────────────────────────────────────────────────────────
if st.session_state.get("model") is None or st.session_state.get("y_pred") is None:
    st.warning("⚠️ No trained model found. Go to **Training** first.")
    st.stop()

task = st.session_state["task_type"]
y_test = st.session_state["y_test"]
y_pred = st.session_state["y_pred"]
y_proba = st.session_state.get("y_proba")
model_name = st.session_state.get("model_name", "Model")

st.markdown(f"## 📈 Evaluation — *{model_name}*")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CLASSIFICATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if task == "classification":
    try:
        avg = "weighted" if len(np.unique(y_test)) > 2 else "binary"

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average=avg, zero_division=0)
        rec = recall_score(y_test, y_pred, average=avg, zero_division=0)
        f1 = f1_score(y_test, y_pred, average=avg, zero_division=0)

        # ROC AUC
        roc_auc = None
        if y_proba is not None:
            try:
                if y_proba.shape[1] == 2:
                    roc_auc = roc_auc_score(y_test, y_proba[:, 1])
                else:
                    roc_auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="weighted")
            except Exception:
                roc_auc = None

        # Metric cards
        cols = st.columns(5 if roc_auc is not None else 4)
        cols[0].metric("Accuracy", f"{acc:.4f}")
        cols[1].metric("Precision", f"{prec:.4f}")
        cols[2].metric("Recall", f"{rec:.4f}")
        cols[3].metric("F1 Score", f"{f1:.4f}")
        if roc_auc is not None:
            cols[4].metric("ROC AUC", f"{roc_auc:.4f}")

        st.markdown("---")

        # Build tab list dynamically
        tab_names = ["🔢 Confusion Matrix"]
        if y_proba is not None:
            tab_names += ["📈 ROC Curve", "📉 Precision–Recall"]
        tab_names.append("📋 Classification Report")
        tabs = st.tabs(tab_names)
        tab_idx = 0

        # ── Confusion Matrix ──────────────────────────────────────────────────
        with tabs[tab_idx]:
            try:
                cm = confusion_matrix(y_test, y_pred)
                labels = sorted(np.unique(y_test))
                label_strs = [str(l) for l in labels]
                fig_cm = px.imshow(
                    cm,
                    text_auto=True,
                    x=label_strs,
                    y=label_strs,
                    color_continuous_scale="Purp",
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    title="Confusion Matrix",
                )
                fig_cm.update_layout(**_LAYOUT, height=500)
                st.plotly_chart(fig_cm, use_container_width=True)
            except Exception as e:
                st.error(f"❌ Error rendering confusion matrix: {e}")
        tab_idx += 1

        # ── ROC Curve ─────────────────────────────────────────────────────────
        if y_proba is not None:
            with tabs[tab_idx]:
                try:
                    classes = sorted(np.unique(y_test))
                    fig_roc = go.Figure()
                    if len(classes) == 2:
                        fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
                        fig_roc.add_trace(
                            go.Scatter(x=fpr, y=tpr, mode="lines", name=f"AUC = {roc_auc:.3f}",
                                       line=dict(color="#7C3AED", width=2.5))
                        )
                    else:
                        from sklearn.preprocessing import label_binarize
                        y_bin = label_binarize(y_test, classes=classes)
                        colors = px.colors.qualitative.Vivid
                        for i, cls in enumerate(classes):
                            fpr_i, tpr_i, _ = roc_curve(y_bin[:, i], y_proba[:, i])
                            auc_i = roc_auc_score(y_bin[:, i], y_proba[:, i])
                            fig_roc.add_trace(
                                go.Scatter(
                                    x=fpr_i, y=tpr_i, mode="lines",
                                    name=f"Class {cls} (AUC={auc_i:.3f})",
                                    line=dict(color=colors[i % len(colors)], width=2),
                                )
                            )
                    fig_roc.add_trace(
                        go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                   line=dict(dash="dash", color="gray"), showlegend=False)
                    )
                    fig_roc.update_layout(
                        **_LAYOUT,
                        title="ROC Curve",
                        xaxis_title="False Positive Rate",
                        yaxis_title="True Positive Rate",
                        height=500,
                    )
                    st.plotly_chart(fig_roc, use_container_width=True)
                except Exception as e:
                    st.error(f"❌ Error rendering ROC curve: {e}")
            tab_idx += 1

            # ── Precision–Recall ──────────────────────────────────────────────
            with tabs[tab_idx]:
                try:
                    fig_pr = go.Figure()
                    if len(classes) == 2:
                        prec_arr, rec_arr, _ = precision_recall_curve(y_test, y_proba[:, 1])
                        ap = average_precision_score(y_test, y_proba[:, 1])
                        fig_pr.add_trace(
                            go.Scatter(x=rec_arr, y=prec_arr, mode="lines",
                                       name=f"AP = {ap:.3f}", line=dict(color="#06B6D4", width=2.5))
                        )
                    else:
                        from sklearn.preprocessing import label_binarize
                        y_bin = label_binarize(y_test, classes=classes)
                        colors = px.colors.qualitative.Vivid
                        for i, cls in enumerate(classes):
                            prec_i, rec_i, _ = precision_recall_curve(y_bin[:, i], y_proba[:, i])
                            ap_i = average_precision_score(y_bin[:, i], y_proba[:, i])
                            fig_pr.add_trace(
                                go.Scatter(
                                    x=rec_i, y=prec_i, mode="lines",
                                    name=f"Class {cls} (AP={ap_i:.3f})",
                                    line=dict(color=colors[i % len(colors)], width=2),
                                )
                            )
                    fig_pr.update_layout(
                        **_LAYOUT,
                        title="Precision–Recall Curve",
                        xaxis_title="Recall",
                        yaxis_title="Precision",
                        height=500,
                    )
                    st.plotly_chart(fig_pr, use_container_width=True)
                except Exception as e:
                    st.error(f"❌ Error rendering precision-recall curve: {e}")
            tab_idx += 1

        # ── Classification Report ─────────────────────────────────────────────
        with tabs[tab_idx]:
            try:
                report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
                report_df = pd.DataFrame(report).T
                st.dataframe(report_df.style.format(precision=4), use_container_width=True, height=400)
            except Exception as e:
                st.error(f"❌ Error generating classification report: {e}")

    except Exception as e:
        st.error(
            f"❌ Error computing classification metrics: {e}\n\n"
            "**Suggestions:** Ensure the target column has valid class labels and the model "
            "was trained correctly on the Training page."
        )

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# REGRESSION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
else:
    try:
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("MAE", f"{mae:.4f}")
        mc2.metric("MSE", f"{mse:.4f}")
        mc3.metric("RMSE", f"{rmse:.4f}")
        mc4.metric("R²", f"{r2:.4f}")

        st.markdown("---")

        tab_pva, tab_res, tab_dist = st.tabs(
            ["📐 Predicted vs Actual", "📊 Residuals", "📉 Error Distribution"]
        )

        # ── Predicted vs Actual ───────────────────────────────────────────────
        with tab_pva:
            try:
                fig_pva = go.Figure()
                fig_pva.add_trace(
                    go.Scatter(
                        x=y_test, y=y_pred, mode="markers",
                        marker=dict(color="#7C3AED", opacity=0.6, size=6),
                        name="Predictions",
                    )
                )
                mn, mx = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
                fig_pva.add_trace(
                    go.Scatter(x=[mn, mx], y=[mn, mx], mode="lines",
                               line=dict(dash="dash", color="#06B6D4"), name="Ideal")
                )
                fig_pva.update_layout(
                    **_LAYOUT,
                    title="Predicted vs Actual",
                    xaxis_title="Actual",
                    yaxis_title="Predicted",
                    height=500,
                )
                st.plotly_chart(fig_pva, use_container_width=True)
            except Exception as e:
                st.error(f"❌ Error rendering Predicted vs Actual chart: {e}")

        # ── Residuals ─────────────────────────────────────────────────────────
        with tab_res:
            try:
                residuals = y_test - y_pred
                fig_res = go.Figure()
                fig_res.add_trace(
                    go.Scatter(
                        x=y_pred, y=residuals, mode="markers",
                        marker=dict(color="#06B6D4", opacity=0.6, size=6),
                        name="Residuals",
                    )
                )
                fig_res.add_hline(y=0, line_dash="dash", line_color="gray")
                fig_res.update_layout(
                    **_LAYOUT,
                    title="Residual Plot",
                    xaxis_title="Predicted",
                    yaxis_title="Residual (Actual − Predicted)",
                    height=500,
                )
                st.plotly_chart(fig_res, use_container_width=True)
            except Exception as e:
                st.error(f"❌ Error rendering residual plot: {e}")

        # ── Error Distribution ────────────────────────────────────────────────
        with tab_dist:
            try:
                residuals = y_test - y_pred
                fig_hist = px.histogram(
                    x=residuals,
                    nbins=40,
                    title="Residual Distribution",
                    labels={"x": "Residual", "y": "Count"},
                    color_discrete_sequence=["#7C3AED"],
                    marginal="box",
                )
                fig_hist.update_layout(**_LAYOUT, height=500)
                st.plotly_chart(fig_hist, use_container_width=True)
            except Exception as e:
                st.error(f"❌ Error rendering error distribution: {e}")

    except Exception as e:
        st.error(
            f"❌ Error computing regression metrics: {e}\n\n"
            "**Suggestions:** Ensure the target column is numeric and the model "
            "was trained correctly on the Training page."
        )
