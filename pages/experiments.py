"""ML Visualizer Lab — Experiment Tracking & Export."""

import json
import streamlit as st
import pandas as pd

st.header("🧪 Experiments")

from utils.export import export_model, export_metrics_csv, export_experiment_history

experiments = st.session_state.get("experiments", [])

if not experiments:
    st.info(
        "📭 No experiments recorded yet.\n\n"
        "Train models on the **Training** page and they will appear here."
    )
    st.stop()

# ─── Summary Stats ────────────────────────────────────────────────────────────
try:
    st.subheader("📊 Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Experiments", len(experiments))
    col2.metric("Unique Algorithms", len(set(e.get("algorithm", "") for e in experiments)))

    # Find best
    task_type = st.session_state.get("task_type", "classification")
    if task_type == "classification":
        best = max(experiments, key=lambda e: e.get("metrics", {}).get("accuracy", 0))
        best_score = best.get("metrics", {}).get("accuracy", 0)
        col3.metric("Best Accuracy", f"{best_score:.4f}")
    else:
        best = max(experiments, key=lambda e: e.get("metrics", {}).get("r2", -999))
        best_score = best.get("metrics", {}).get("r2", 0)
        col3.metric("Best R²", f"{best_score:.4f}")

    st.success(f"🏆 Best: **{best.get('algorithm', 'Unknown')}**")
except Exception as e:
    st.error(
        f"❌ **Error**: {str(e)}\n\n"
        "Please check your data and try again."
    )

# ─── Experiments Table ────────────────────────────────────────────────────────
try:
    st.subheader("📋 Experiment History")

    # Flatten for display
    display_data = []
    for exp in experiments:
        row = {
            "Timestamp": exp.get("timestamp", ""),
            "Dataset": exp.get("dataset", ""),
            "Algorithm": exp.get("algorithm", ""),
            "Task": exp.get("task_type", ""),
            "Training Time (s)": round(exp.get("training_time", 0), 3),
            "Hyperparameters": json.dumps(exp.get("hyperparameters", {}), default=str)[:80],
        }
        metrics = exp.get("metrics", {})
        for k, v in metrics.items():
            row[k] = round(v, 4) if isinstance(v, (int, float)) and v is not None else v
        display_data.append(row)

    display_df = pd.DataFrame(display_data)
    st.dataframe(display_df, use_container_width=True)
except Exception as e:
    st.error(
        f"❌ **Error**: {str(e)}\n\n"
        "Please check your data and try again."
    )

# ─── Metric Trends ────────────────────────────────────────────────────────────
if len(experiments) >= 2:
    try:
        st.subheader("📈 Metric Trends")
        import plotly.graph_objects as go

        if task_type == "classification":
            metric_key = "accuracy"
        else:
            metric_key = "r2"

        values = [e.get("metrics", {}).get(metric_key, 0) for e in experiments]
        labels = [f"#{i+1} {e.get('algorithm', '')[:15]}" for i, e in enumerate(experiments)]

        fig = go.Figure(go.Scatter(
            x=labels, y=values, mode="lines+markers",
            line=dict(color="#7C3AED", width=2),
            marker=dict(size=8),
        ))
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)",
                          title=f"{metric_key.upper()} Across Experiments",
                          yaxis_title=metric_key, height=400)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(
            f"❌ **Error**: {str(e)}\n\n"
            "Please check your data and try again."
        )

# ─── Export ───────────────────────────────────────────────────────────────────
st.subheader("💾 Export")

try:
    col1, col2, col3 = st.columns(3)

    with col1:
        csv_data = export_experiment_history(experiments)
        st.download_button(
            "📥 Download History (CSV)", csv_data,
            file_name="experiment_history.csv", mime="text/csv",
            use_container_width=True,
        )

    with col2:
        if st.session_state.get("model") is not None:
            model_bytes = export_model(st.session_state["model"])
            st.download_button(
                "📥 Download Model (.pkl)", model_bytes,
                file_name="trained_model.pkl", mime="application/octet-stream",
                use_container_width=True,
            )
        else:
            st.button("📥 Download Model (.pkl)", disabled=True, use_container_width=True)

    with col3:
        if experiments:
            latest_metrics = experiments[-1].get("metrics", {})
            metrics_csv = export_metrics_csv(latest_metrics)
            st.download_button(
                "📥 Download Metrics (CSV)", metrics_csv,
                file_name="metrics_report.csv", mime="text/csv",
                use_container_width=True,
            )
except Exception as e:
    st.error(
        f"❌ **Error**: {str(e)}\n\n"
        "Please check your data and try again."
    )

# ─── Clear History ────────────────────────────────────────────────────────────
st.markdown("---")
if st.button("🗑️ Clear Experiment History", type="secondary"):
    st.session_state["experiments"] = []
    st.rerun()
