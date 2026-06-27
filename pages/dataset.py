"""
ML Visualizer Lab — Dataset Overview
Upload or select a sample dataset, preview data, view statistics, types, and correlations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.datasets import load_iris, load_wine, load_diabetes, fetch_california_housing

# ─── Helpers ───────────────

_SAMPLE_DATASETS = {
    "Iris": load_iris,
    "Wine": load_wine,
    "Diabetes": load_diabetes,
    "California Housing": fetch_california_housing,
}


def _sklearn_to_df(loader_fn) -> pd.DataFrame:
    """Convert a scikit-learn Bunch to a DataFrame with a 'target' column."""
    data = loader_fn()
    feature_names = (
        data.feature_names
        if hasattr(data, "feature_names")
        else [f"feature_{i}" for i in range(data.data.shape[1])]
    )
    df = pd.DataFrame(data.data, columns=feature_names)
    df["target"] = data.target
    return df


# ─── Sidebar — File Upload & Sample Selector ────
with st.sidebar:
    st.header("📂 Load Data")
    uploaded_file = st.file_uploader(
        "Upload CSV or XLSX",
        type=["csv", "xlsx"],
        help="Drag-and-drop or click to upload your dataset.",
    )
    st.markdown("**— or —**")
    sample_name = st.selectbox(
        "Sample dataset", options=["—"] + list(_SAMPLE_DATASETS.keys())
    )
    load_sample_btn = st.button("Load Sample", use_container_width=True)

# ─── Loading Logic ─────
if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        st.session_state["df"] = df
        st.session_state["dataset_name"] = uploaded_file.name
    except Exception as exc:
        st.error(f"Failed to read file: {exc}")

if load_sample_btn and sample_name != "—":
    try:
        df = _sklearn_to_df(_SAMPLE_DATASETS[sample_name])
        st.session_state["df"] = df
        st.session_state["dataset_name"] = sample_name
    except Exception as e:
        st.error(
            f"❌ **Error**: {str(e)}\n\n"
            "Please check your data and try again."
        )

# ─── Guard ───────────────────────────────────────
df: pd.DataFrame | None = st.session_state.get("df")

if df is None:
    st.warning("⬆️ Upload a file or select a sample dataset from the sidebar.")
    st.stop()

# ─── Page Title ────────────────────────────
st.markdown(
    f"## 📊 Dataset Overview — *{st.session_state.get('dataset_name', 'Untitled')}*"
)

# ─── Top Metrics Row ─────────────────────
try:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Rows", f"{df.shape[0]:,}")
    m2.metric("Columns", f"{df.shape[1]}")
    m3.metric("Missing Values", f"{int(df.isnull().sum().sum()):,}")
    mem_mb = df.memory_usage(deep=True).sum() / 1024 ** 2
    m4.metric("Memory", f"{mem_mb:.2f} MB")
except Exception as e:
    st.error(
        f"❌ **Error**: {str(e)}\n\n"
        "Please check your data and try again."
    )

st.markdown("---")

# ─── Tabs ──────────────────────────────────────────────────────────────────────
tab_preview, tab_stats, tab_types, tab_corr = st.tabs(
    ["📋 Preview", "📊 Statistics", "🏷️ Data Types", "🔗 Correlation"]
)

# --- Preview ----------------------------------------------------------------
with tab_preview:
    try:
        st.dataframe(df.head(50), use_container_width=True, height=480)
    except Exception as e:
        st.error(
            f"❌ **Error**: {str(e)}\n\n"
            "Please check your data and try again."
        )

# --- Statistics --------------------------------------------------------------
with tab_stats:
    try:
        desc = df.describe(include="all").T
        desc.index.name = "Column"
        st.dataframe(desc.style.format(precision=3), use_container_width=True, height=480)
    except Exception as e:
        st.error(
            f"❌ **Error**: {str(e)}\n\n"
            "Please check your data and try again."
        )

# --- Data Types --------------------------------------------------------------
with tab_types:
    try:
        type_info = pd.DataFrame(
            {
                "Column": df.columns,
                "Dtype": df.dtypes.astype(str).values,
                "Non-Null": df.notnull().sum().values,
                "Null": df.isnull().sum().values,
                "Unique": df.nunique().values,
                "Missing %": (df.isnull().mean() * 100).round(2).values,
            }
        )
        st.dataframe(type_info, use_container_width=True, hide_index=True, height=480)
    except Exception as e:
        st.error(
            f"❌ **Error**: {str(e)}\n\n"
            "Please check your data and try again."
        )

# --- Correlation -------------------------------------------------------------
with tab_corr:
    try:
        numeric_df = df.select_dtypes(include=np.number)
        if numeric_df.shape[1] < 2:
            st.info("Need at least 2 numeric columns to compute a correlation matrix.")
        else:
            corr = numeric_df.corr()
            fig = px.imshow(
                corr,
                text_auto=".2f",
                color_continuous_scale="RdBu_r",
                aspect="auto",
                title="Pearson Correlation Matrix",
            )
            fig.update_layout(
                height=600,
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter"),
            )
            st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(
            f"❌ **Error**: {str(e)}\n\n"
            "Please check your data and try again."
        )
