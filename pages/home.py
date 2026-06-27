"""
ML Visualizer Lab — Home (Landing Page)
Beautiful welcome page with feature cards and getting-started guide.
"""

import streamlit as st

# ─── Hero Section ──────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style="text-align:center; padding: 2rem 0 1rem;">
        <h1 style="
            font-size: 3.2rem;
            font-weight: 700;
            background: linear-gradient(135deg, #7C3AED 0%, #06B6D4 50%, #10B981 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.3rem;
        ">🧪 ML Visualizer Lab</h1>
        <p style="
            font-size: 1.25rem;
            color: #94A3B8;
            font-weight: 300;
            letter-spacing: 0.02em;
        ">Interactive Machine Learning Exploration Platform</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("")  # spacer

# ─── Feature Cards ─────────────────────────────────────────────────────────────
_card_css = """
<style>
.feature-card {
    background: linear-gradient(135deg, rgba(124,58,237,0.08) 0%, rgba(6,182,212,0.08) 100%);
    border: 1px solid rgba(124,58,237,0.25);
    border-radius: 16px;
    padding: 28px 24px;
    text-align: center;
    transition: transform 0.25s ease, box-shadow 0.25s ease;
    height: 100%;
}
.feature-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 32px rgba(124,58,237,0.15);
}
.feature-card .card-icon {
    font-size: 2.8rem;
    margin-bottom: 12px;
}
.feature-card .card-title {
    font-size: 1.15rem;
    font-weight: 600;
    color: #E2E8F0;
    margin-bottom: 8px;
}
.feature-card .card-desc {
    font-size: 0.92rem;
    color: #94A3B8;
    line-height: 1.55;
}
</style>
"""
st.markdown(_card_css, unsafe_allow_html=True)

c1, c2, c3 = st.columns(3, gap="large")

with c1:
    st.markdown(
        """
        <div class="feature-card">
            <div class="card-icon">📂</div>
            <div class="card-title">Upload &amp; Explore Datasets</div>
            <div class="card-desc">
                Load CSV / XLSX files or pick from built-in sample datasets.
                Instantly preview statistics, distributions, and correlations.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c2:
    st.markdown(
        """
        <div class="feature-card">
            <div class="card-icon">🤖</div>
            <div class="card-title">Train &amp; Tune ML Models</div>
            <div class="card-desc">
                Choose from classifiers and regressors with real-time
                hyperparameter tuning, cross-validation, and grid search.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c3:
    st.markdown(
        """
        <div class="feature-card">
            <div class="card-icon">📊</div>
            <div class="card-title">Visualize &amp; Compare Results</div>
            <div class="card-desc">
                Rich Plotly charts — confusion matrices, ROC curves,
                SHAP explanations, learning curves, and experiment tracking.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# ─── Getting Started ──────────────────────────────────────────────────────────
st.markdown(
    """
    <div style="
        background: linear-gradient(135deg, rgba(16,185,129,0.08) 0%, rgba(6,182,212,0.08) 100%);
        border: 1px solid rgba(16,185,129,0.25);
        border-radius: 16px;
        padding: 28px 32px;
        margin-bottom: 1.5rem;
    ">
        <h3 style="color:#E2E8F0; margin-top:0;">🚀 Getting Started</h3>
        <ol style="color:#94A3B8; line-height:2; padding-left:20px; margin-bottom:0;">
            <li><strong style="color:#E2E8F0;">Load a dataset</strong> — Go to <em>Dataset Overview</em> and upload a file or select a sample dataset.</li>
            <li><strong style="color:#E2E8F0;">Preprocess your data</strong> — Select features &amp; target, handle missing values, encode categoricals, and split into train/test.</li>
            <li><strong style="color:#E2E8F0;">Train a model</strong> — Pick an algorithm, tune hyperparameters, and hit <em>Train</em>.</li>
            <li><strong style="color:#E2E8F0;">Evaluate &amp; visualize</strong> — Explore metrics, confusion matrices, ROC curves, and SHAP explanations.</li>
            <li><strong style="color:#E2E8F0;">Compare &amp; iterate</strong> — Track experiments, compare models, and optimize hyperparameters.</li>
        </ol>
    </div>
    """,
    unsafe_allow_html=True,
)

# ─── Quick Links ──────────────────────────────────────────────────────────────
st.markdown("### 🔗 Quick Links")

_quick_links = [
    ("📊", "Dataset Overview", "pages/dataset.py"),
    ("🔧", "Preprocessing", "pages/preprocessing.py"),
    ("🎯", "Training", "pages/training.py"),
    ("📈", "Evaluation", "pages/evaluation.py"),
    ("🎨", "Visualizations", "pages/visualizations.py"),
    ("⚔️", "Model Comparison", "pages/comparison.py"),
    ("🧠", "Explainability", "pages/explainability.py"),
    ("🧪", "Experiments", "pages/experiments.py"),
]

row1 = st.columns(4, gap="medium")
row2 = st.columns(4, gap="medium")
all_cols = row1 + row2

for col, (icon, label, page) in zip(all_cols, _quick_links):
    with col:
        st.page_link(page, label=f"{icon}  {label}", use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── Developers ───────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .dev-card {
        flex:1; min-width:220px;
        background: rgba(30,33,40,0.5);
        border: 1px solid rgba(124,58,237,0.2);
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        position: relative;
    }
    .dev-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 24px rgba(124,58,237,0.2);
    }
    .dev-avatar {
        width: 80px; height: 80px;
        border-radius: 50%;
        margin: 0 auto 12px;
        display: flex; align-items: center; justify-content: center;
        font-size: 2.2rem;
        cursor: pointer;
        transition: transform 0.3s ease;
    }
    .dev-avatar:hover { transform: scale(1.1); }
    .dev-avatar-ps { background: linear-gradient(135deg, #7C3AED, #06B6D4); }
    .dev-avatar-sk { background: linear-gradient(135deg, #EC4899, #F59E0B); }
    .dev-name { font-weight: 600; color: #E2E8F0; font-size: 1.1rem; margin-bottom: 4px; }
    .dev-role { color: #94A3B8; font-size: 0.85rem; margin-bottom: 12px; }
    .dev-links { display: flex; justify-content: center; gap: 12px; }
    .dev-links a {
        display: inline-flex; align-items: center; gap: 6px;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 500;
        text-decoration: none;
        transition: all 0.2s ease;
    }
    .dev-links a.gh {
        background: rgba(124,58,237,0.15); color: #7C3AED;
        border: 1px solid rgba(124,58,237,0.3);
    }
    .dev-links a.gh:hover { background: rgba(124,58,237,0.3); }
    .dev-links a.tw {
        background: rgba(6,182,212,0.15); color: #06B6D4;
        border: 1px solid rgba(6,182,212,0.3);
    }
    .dev-links a.tw:hover { background: rgba(6,182,212,0.3); }
    </style>

    <div style="
        background: linear-gradient(135deg, rgba(124,58,237,0.06) 0%, rgba(236,72,153,0.06) 100%);
        border: 1px solid rgba(124,58,237,0.2);
        border-radius: 16px;
        padding: 28px 32px;
        margin-bottom: 1.5rem;
    ">
        <h3 style="color:#E2E8F0; margin-top:0;">👨💻 Meet the Developers</h3>
        <div style="display:flex; gap:24px; flex-wrap:wrap; margin-top:12px;">
            <div class="dev-card">
                <div class="dev-avatar dev-avatar-ps">PS</div>
                <div class="dev-name">Priyanshu Shekhar</div>
                <div class="dev-role">Full Stack Developer</div>
                <div class="dev-links">
                    <a href="https://github.com/pinshucodes" target="_blank" class="gh">⚡ GitHub</a>
                    <a href="https://x.com/hightableclock" target="_blank" class="tw">𝕏 Twitter</a>
                </div>
            </div>
            <div class="dev-card">
                <div class="dev-avatar dev-avatar-sk">SK</div>
                <div class="dev-name">Supriya Kumari</div>
                <div class="dev-role">ML Engineer</div>
                <div class="dev-links">
                    <a href="https://github.com/suppcodes" target="_blank" class="gh">⚡ GitHub</a>
                    <a href="https://x.com" target="_blank" class="tw">𝕏 Twitter</a>
                </div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ─── Tech Stack Footer ────────────────────────────────────────────────────────
st.markdown(
    """
    <div style="text-align:center; padding:1.5rem 0; border-top:1px solid rgba(124,58,237,0.15);">
        <p style="color:#64748B; font-size:0.85rem; margin-bottom:10px;">Built with</p>
        <div style="display:flex; justify-content:center; gap:12px; flex-wrap:wrap; margin-bottom:14px;">
            <span style="background:rgba(255,75,75,0.15); color:#FF4B4B; padding:4px 14px; border-radius:20px; font-size:0.82rem; font-weight:500;">Streamlit</span>
            <span style="background:rgba(6,182,212,0.15); color:#06B6D4; padding:4px 14px; border-radius:20px; font-size:0.82rem; font-weight:500;">Scikit-learn</span>
            <span style="background:rgba(99,102,241,0.15); color:#818CF8; padding:4px 14px; border-radius:20px; font-size:0.82rem; font-weight:500;">Plotly</span>
            <span style="background:rgba(16,185,129,0.15); color:#10B981; padding:4px 14px; border-radius:20px; font-size:0.82rem; font-weight:500;">Pandas</span>
            <span style="background:rgba(245,158,11,0.15); color:#F59E0B; padding:4px 14px; border-radius:20px; font-size:0.82rem; font-weight:500;">NumPy</span>
        </div>
        <a href="https://github.com/pinshucodes/MLVisualizerLab" target="_blank" style="
            color:#7C3AED; text-decoration:none; font-size:0.88rem; font-weight:500;
        ">⭐ Star us on GitHub</a>
    </div>
    """,
    unsafe_allow_html=True,
)
