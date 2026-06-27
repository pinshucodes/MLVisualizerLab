"""
ML Visualizer Lab — Main Entry Point
Interactive Machine Learning Exploration Platform
"""

import streamlit as st

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ML Visualizer Lab",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="st-"] { font-family: 'Inter', sans-serif; }

    .main .block-container { padding-top: 2rem; max-width: 1200px; }

    /* Gradient header */
    .stApp > header { background: linear-gradient(135deg, #7C3AED 0%, #06B6D4 100%); }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(124,58,237,0.1) 0%, rgba(6,182,212,0.1) 100%);
        border: 1px solid rgba(124,58,237,0.3);
        border-radius: 12px;
        padding: 16px;
    }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 16px;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0E1117 0%, #1A1D23 100%);
    }

    /* ====== FIX: Hide Material Icon ligature text in sidebar ====== */
    /* These render as 'expand_more', 'keyboard_double_arrow_left' text */
    section[data-testid="stSidebar"] .material-symbols-rounded,
    section[data-testid="stSidebar"] .material-icons,
    section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] span,
    section[data-testid="stSidebar"] [data-testid="collapsedControl"] span {
        font-size: 0 !important;
        visibility: hidden !important;
        width: 0 !important;
        height: 0 !important;
        overflow: hidden !important;
        position: absolute !important;
    }

    /* Hide the collapse/expand button entirely for cleaner look */
    [data-testid="stSidebarCollapseButton"],
    [data-testid="collapsedControl"] {
        display: none !important;
    }

    /* Fix nav section headers — hide expand_more icon */
    section[data-testid="stSidebar"] [data-testid="stSidebarNavSectionHeader"] button {
        pointer-events: none;
    }
    section[data-testid="stSidebar"] [data-testid="stSidebarNavSectionHeader"] button span[class*="icon"],
    section[data-testid="stSidebar"] [data-testid="stSidebarNavSectionHeader"] button svg {
        display: none !important;
    }
    /* Also hide any raw text that reads 'expand_more' or 'View less' */
    section[data-testid="stSidebar"] [data-testid="stSidebarNavSectionHeader"] {
        overflow: hidden !important;
    }

    /* Make sidebar nav items always expanded (no collapsing) */
    section[data-testid="stSidebar"] [data-testid="stSidebarNavItems"] {
        max-height: none !important;
        overflow: visible !important;
    }

    /* Success/info boxes */
    .stAlert { border-radius: 12px; }

    /* Smooth page transitions */
    .main .block-container {
        animation: fadeIn 0.3s ease-in;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(8px); }
        to { opacity: 1; transform: translateY(0); }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─── Session State Defaults ────────────────────────────────────────────────────
_defaults = {
    "df": None,
    "df_processed": None,
    "X_train": None,
    "X_test": None,
    "y_train": None,
    "y_test": None,
    "task_type": None,
    "model": None,
    "model_name": None,
    "feature_names": None,
    "target_name": None,
    "experiments": [],
    "scaler": None,
    "dataset_name": None,
    "y_pred": None,
    "y_proba": None,
}

for key, val in _defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ─── Navigation ────────────────────────────────────────────────────────────────
pg = st.navigation(
    {
        "📁 Data": [
            st.Page("pages/home.py", title="Home", icon="🏠", default=True),
            st.Page("pages/dataset.py", title="Dataset Overview", icon="📊"),
            st.Page("pages/preprocessing.py", title="Preprocessing", icon="🔧"),
        ],
        "🤖 Modeling": [
            st.Page("pages/training.py", title="Training", icon="🎯"),
            st.Page("pages/evaluation.py", title="Evaluation", icon="📈"),
            st.Page("pages/visualizations.py", title="Visualizations", icon="🎨"),
        ],
        "🧠 Advanced": [
            st.Page("pages/explainability.py", title="Explainability", icon="🧠"),
            st.Page("pages/comparison.py", title="Model Comparison", icon="⚔️"),
            st.Page("pages/dimensionality.py", title="Dimensionality Reduction", icon="📉"),
            st.Page("pages/clustering.py", title="Clustering", icon="🔮"),
        ],
        "📐 Analysis": [
            st.Page("pages/cross_validation.py", title="Cross Validation", icon="📐"),
            st.Page("pages/learning_curves.py", title="Learning Curves", icon="📚"),
            st.Page("pages/hyperparameter_opt.py", title="Hyperparameter Optimization", icon="⚡"),
            st.Page("pages/experiments.py", title="Experiments", icon="🧪"),
        ],
    }
)

# ─── Shared Sidebar Status ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("---")
    if st.session_state.get("df") is not None:
        st.success(f"✅ Dataset: {st.session_state.get('dataset_name', 'Uploaded')}")
        st.caption(
            f"{st.session_state['df'].shape[0]} rows × "
            f"{st.session_state['df'].shape[1]} cols"
        )
    else:
        st.info("📂 No dataset loaded")

    if st.session_state.get("model") is not None:
        st.success(f"🤖 Model: {st.session_state.get('model_name', 'Trained')}")
    else:
        st.info("🤖 No model trained")

pg.run()
