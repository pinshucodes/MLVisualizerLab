"""ML Visualizer Lab — Clustering (K-Means & DBSCAN)."""

import streamlit as st
import numpy as np

st.header("🔮 Clustering")

# Guard
if st.session_state.get("X_train") is None:
    st.warning("⚠️ No preprocessed data. Go to **Preprocessing** first.")
    st.stop()

from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from models.clustering import get_clustering_model, get_clustering_names
from utils.plots import plot_cluster_scatter, plot_elbow, plot_silhouette_chart

X_train = st.session_state["X_train"]
X_test = st.session_state["X_test"]
X_all = np.vstack([X_train, X_test])

# ─── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.subheader("🔮 Clustering Settings")
algorithm = st.sidebar.selectbox("Algorithm", get_clustering_names())

if algorithm == "K-Means":
    n_clusters = st.sidebar.slider("Number of clusters", 2, 15, 3)
    params = {"n_clusters": n_clusters}
else:  # DBSCAN
    eps = st.sidebar.slider("eps (neighborhood radius)", 0.1, 5.0, 0.5, step=0.1)
    min_samples = st.sidebar.slider("min_samples", 2, 20, 5)
    params = {"eps": eps, "min_samples": min_samples}

# ─── Run Clustering ───────────────────────────────────────────────────────────
if st.button("🚀 Run Clustering", type="primary", use_container_width=True):
    with st.spinner("Clustering..."):
        try:
            model = get_clustering_model(algorithm, params)
            labels = model.fit_predict(X_all)
            st.session_state["cluster_labels"] = labels
            st.session_state["cluster_algo"] = algorithm
        except Exception as e:
            st.error(
                f"❌ Error during clustering: {e}\n\n"
                "**Suggestions:** Try adjusting parameters (e.g., increase eps for DBSCAN, "
                "or reduce the number of clusters for K-Means)."
            )

if "cluster_labels" not in st.session_state:
    st.info("Configure settings and click **Run Clustering** to start.")
    st.stop()

labels = st.session_state["cluster_labels"]
algo = st.session_state["cluster_algo"]

n_clusters_found = len(set(labels) - {-1})
n_noise = int(np.sum(labels == -1))

# ─── Metrics ──────────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)
col1.metric("Clusters Found", n_clusters_found)
col2.metric("Noise Points", n_noise)

try:
    if n_clusters_found >= 2:
        sil = silhouette_score(X_all, labels)
        col3.metric("Silhouette Score", f"{sil:.4f}")
    else:
        col3.metric("Silhouette Score", "N/A")
except Exception as e:
    col3.metric("Silhouette Score", "Error")
    st.warning(f"⚠️ Could not compute silhouette score: {e}")

# ─── Visualizations ──────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊 Cluster Scatter", "📉 Elbow Method", "🔍 Silhouette"])

with tab1:
    try:
        if X_all.shape[1] > 2:
            pca = PCA(n_components=2)
            X_2d = pca.fit_transform(X_all)
            st.caption("Reduced to 2D using PCA for visualization.")
        else:
            X_2d = X_all[:, :2]
        fig = plot_cluster_scatter(X_2d, labels)
        st.plotly_chart(fig, use_container_width=True)

        # Cluster size distribution
        import pandas as pd
        import plotly.graph_objects as go
        cluster_counts = pd.Series(labels).value_counts().sort_index()
        fig_bar = go.Figure(go.Bar(
            x=[f"Cluster {c}" if c != -1 else "Noise" for c in cluster_counts.index],
            y=cluster_counts.values,
            marker_color=["gray" if c == -1 else "#7C3AED" for c in cluster_counts.index],
        ))
        fig_bar.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)",
                              title="Cluster Size Distribution", yaxis_title="Count")
        st.plotly_chart(fig_bar, use_container_width=True)
    except Exception as e:
        st.error(f"❌ Error rendering cluster scatter plot: {e}")

with tab2:
    if algo == "K-Means":
        st.subheader("Elbow Method")
        try:
            k_range = range(2, 16)
            inertias = []
            for k in k_range:
                km = KMeans(n_clusters=k, n_init=10, random_state=42)
                km.fit(X_all)
                inertias.append(km.inertia_)
            fig = plot_elbow(inertias, k_range)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"❌ Error computing elbow method: {e}")
    else:
        st.info("Elbow method is only applicable for K-Means.")

with tab3:
    if n_clusters_found >= 2:
        try:
            fig = plot_silhouette_chart(X_all, labels)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"❌ Error rendering silhouette chart: {e}")
    else:
        st.info("Need at least 2 clusters for silhouette analysis.")
