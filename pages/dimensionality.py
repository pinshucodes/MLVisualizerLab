"""ML Visualizer Lab — Dimensionality Reduction (PCA, t-SNE, UMAP)."""

import streamlit as st
import numpy as np

st.header("📉 Dimensionality Reduction")

# Guard
if st.session_state.get("X_train") is None:
    st.warning("⚠️ No preprocessed data. Go to **Preprocessing** first.")
    st.stop()

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from utils.plots import plot_scree, plot_dimensionality_scatter

X_train = st.session_state["X_train"]
X_test = st.session_state["X_test"]
y_train = st.session_state["y_train"]
y_test = st.session_state["y_test"]
feature_names = st.session_state["feature_names"]

# Combine for visualization
X_all = np.vstack([X_train, X_test])
y_all = np.concatenate([y_train, y_test])

n_features = X_all.shape[1]

# Try UMAP import
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

tab_pca, tab_tsne, tab_umap = st.tabs(["📐 PCA", "🌀 t-SNE", "🔬 UMAP"])

# ─── PCA ──────────────────────────────────────────────────────────────────────
with tab_pca:
    st.subheader("Principal Component Analysis")
    max_comp = min(n_features, 20, X_all.shape[0])
    n_components = st.slider("Number of components", 2, max_comp, min(2, max_comp), key="pca_comp")

    try:
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_all)

        # Scree plot
        fig_scree = plot_scree(pca.explained_variance_ratio_)
        st.plotly_chart(fig_scree, use_container_width=True)

        # Explained variance table
        with st.expander("📋 Explained Variance Details"):
            import pandas as pd
            ev_df = pd.DataFrame({
                "Component": [f"PC{i+1}" for i in range(n_components)],
                "Explained Variance Ratio": pca.explained_variance_ratio_,
                "Cumulative": np.cumsum(pca.explained_variance_ratio_),
            })
            st.dataframe(ev_df.style.format({"Explained Variance Ratio": "{:.4f}",
                                              "Cumulative": "{:.4f}"}),
                          use_container_width=True)

        # Scatter
        st.subheader("Projection")
        use_3d = st.checkbox("3D view", value=False, key="pca_3d") if n_components >= 3 else False
        X_plot = X_pca[:, :3] if use_3d else X_pca[:, :2]
        fig = plot_dimensionality_scatter(X_plot, y_all, "PCA")
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        col1.metric("Total Variance Explained", f"{sum(pca.explained_variance_ratio_):.1%}")
        col2.metric("Components", n_components)
    except Exception as e:
        st.error(
            f"❌ Error during PCA computation: {e}\n\n"
            "**Suggestions:** Try reducing the number of components or check that "
            "the dataset has no constant/empty features."
        )

# ─── t-SNE ────────────────────────────────────────────────────────────────────
with tab_tsne:
    st.subheader("t-Distributed Stochastic Neighbor Embedding")

    max_samples_tsne = 5000
    if len(X_all) > max_samples_tsne:
        st.warning(f"⚠️ Dataset has {len(X_all)} samples. Subsampling to {max_samples_tsne} for speed.")
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X_all), max_samples_tsne, replace=False)
        X_tsne_input = X_all[idx]
        y_tsne = y_all[idx]
    else:
        X_tsne_input = X_all
        y_tsne = y_all

    perplexity = st.slider("Perplexity", 5, 50, 30, key="tsne_perp")
    n_iter = st.slider("Iterations", 250, 1000, 300, step=50, key="tsne_iter")

    if st.button("🔄 Run t-SNE", key="tsne_run"):
        with st.spinner("Running t-SNE... This may take a moment."):
            try:
                tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter,
                            random_state=42)
                X_tsne = tsne.fit_transform(X_tsne_input)
                fig = plot_dimensionality_scatter(X_tsne, y_tsne, "t-SNE")
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(
                    f"❌ Error during t-SNE computation: {e}\n\n"
                    "**Suggestions:** Try lowering the perplexity value or reducing "
                    "the number of iterations."
                )

# ─── UMAP ─────────────────────────────────────────────────────────────────────
with tab_umap:
    st.subheader("Uniform Manifold Approximation and Projection")

    if not UMAP_AVAILABLE:
        st.warning(
            "⚠️ **UMAP is not installed.** Install with:\n\n"
            "```bash\npip install umap-learn\n```"
        )
    else:
        max_samples_umap = 5000
        if len(X_all) > max_samples_umap:
            st.warning(f"⚠️ Subsampling to {max_samples_umap} samples for speed.")
            rng = np.random.RandomState(42)
            idx = rng.choice(len(X_all), max_samples_umap, replace=False)
            X_umap_input = X_all[idx]
            y_umap = y_all[idx]
        else:
            X_umap_input = X_all
            y_umap = y_all

        n_neighbors = st.slider("n_neighbors", 2, 100, 15, key="umap_nn")
        min_dist = st.slider("min_dist", 0.0, 1.0, 0.1, step=0.05, key="umap_md")

        if st.button("🔄 Run UMAP", key="umap_run"):
            with st.spinner("Running UMAP..."):
                try:
                    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors,
                                        min_dist=min_dist, random_state=42)
                    X_umap = reducer.fit_transform(X_umap_input)
                    fig = plot_dimensionality_scatter(X_umap, y_umap, "UMAP")
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(
                        f"❌ Error during UMAP computation: {e}\n\n"
                        "**Suggestions:** Try adjusting n_neighbors or min_dist parameters."
                    )
