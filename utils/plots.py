"""Visualization utilities for ML Visualizer Lab.

All public functions return Plotly figures using the ``plotly_dark`` template
with a consistent violet/cyan color palette.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from sklearn.preprocessing import label_binarize

# ---------------------------------------------------------------------------
# Shared palette & layout helpers
# ---------------------------------------------------------------------------

COLORS = ["#7C3AED", "#06B6D4", "#10B981", "#F59E0B", "#EF4444", "#EC4899",
          "#8B5CF6", "#14B8A6", "#F97316", "#64748B"]

_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif"),
    margin=dict(l=40, r=40, t=50, b=40),
)


def _base_fig(**kwargs: Any) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(**_LAYOUT, **kwargs)
    return fig


# ---------------------------------------------------------------------------
# Correlation Matrix
# ---------------------------------------------------------------------------

def plot_correlation_matrix(df: pd.DataFrame) -> go.Figure:
    """Plotly heatmap of the correlation matrix."""
    numeric = df.select_dtypes(include="number")
    corr = numeric.corr()
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.columns.tolist(),
        colorscale=[[0, "#06B6D4"], [0.5, "#0E1117"], [1, "#7C3AED"]],
        text=np.round(corr.values, 2),
        texttemplate="%{text}",
        textfont=dict(size=10),
        hoverongaps=False,
    ))
    fig.update_layout(**_LAYOUT, title="Correlation Matrix",
                      height=max(400, len(corr) * 35))
    return fig


# ---------------------------------------------------------------------------
# Confusion Matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                          labels: list | None = None) -> go.Figure:
    """Annotated confusion matrix heatmap."""
    cm = sk_confusion_matrix(y_true, y_pred, labels=labels)
    if labels is None:
        labels = [str(c) for c in sorted(np.unique(np.concatenate([y_true, y_pred])))]
    labels_str = [str(l) for l in labels]

    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=labels_str,
        y=labels_str,
        colorscale=[[0, "rgba(124,58,237,0.1)"], [1, "#7C3AED"]],
        text=cm,
        texttemplate="%{text}",
        textfont=dict(size=14),
        hoverongaps=False,
        showscale=False,
    ))
    fig.update_layout(**_LAYOUT, title="Confusion Matrix",
                      xaxis_title="Predicted", yaxis_title="Actual",
                      height=450)
    fig.update_yaxes(autorange="reversed")
    return fig


# ---------------------------------------------------------------------------
# ROC Curve
# ---------------------------------------------------------------------------

def plot_roc_curve(y_true: np.ndarray, y_proba: np.ndarray,
                   classes: list | None = None) -> go.Figure:
    """ROC curve with multi-class One-vs-Rest support."""
    from sklearn.metrics import roc_curve, auc

    fig = _base_fig(title="ROC Curve", xaxis_title="False Positive Rate",
                    yaxis_title="True Positive Rate", height=500)

    unique = sorted(np.unique(y_true))
    if len(unique) == 2:
        proba = y_proba[:, 1] if y_proba.ndim == 2 else y_proba
        fpr, tpr, _ = roc_curve(y_true, proba)
        roc_auc = auc(fpr, tpr)
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                 name=f"AUC = {roc_auc:.3f}",
                                 line=dict(color=COLORS[0], width=2)))
    else:
        y_bin = label_binarize(y_true, classes=unique)
        for i, cls in enumerate(unique):
            if y_proba.ndim == 2 and i < y_proba.shape[1]:
                fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
                roc_auc = auc(fpr, tpr)
                name = classes[i] if classes and i < len(classes) else str(cls)
                fig.add_trace(go.Scatter(
                    x=fpr, y=tpr, mode="lines",
                    name=f"{name} (AUC={roc_auc:.3f})",
                    line=dict(color=COLORS[i % len(COLORS)], width=2),
                ))

    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                             name="Random", line=dict(color="gray", dash="dash")))
    return fig


# ---------------------------------------------------------------------------
# Precision-Recall Curve
# ---------------------------------------------------------------------------

def plot_precision_recall_curve(y_true: np.ndarray, y_proba: np.ndarray,
                                classes: list | None = None) -> go.Figure:
    """Precision-Recall curve with multi-class support."""
    from sklearn.metrics import precision_recall_curve, average_precision_score

    fig = _base_fig(title="Precision-Recall Curve",
                    xaxis_title="Recall", yaxis_title="Precision", height=500)

    unique = sorted(np.unique(y_true))
    if len(unique) == 2:
        proba = y_proba[:, 1] if y_proba.ndim == 2 else y_proba
        prec, rec, _ = precision_recall_curve(y_true, proba)
        ap = average_precision_score(y_true, proba)
        fig.add_trace(go.Scatter(x=rec, y=prec, mode="lines",
                                 name=f"AP = {ap:.3f}",
                                 line=dict(color=COLORS[0], width=2)))
    else:
        y_bin = label_binarize(y_true, classes=unique)
        for i, cls in enumerate(unique):
            if y_proba.ndim == 2 and i < y_proba.shape[1]:
                prec, rec, _ = precision_recall_curve(y_bin[:, i], y_proba[:, i])
                ap = average_precision_score(y_bin[:, i], y_proba[:, i])
                name = classes[i] if classes and i < len(classes) else str(cls)
                fig.add_trace(go.Scatter(
                    x=rec, y=prec, mode="lines",
                    name=f"{name} (AP={ap:.3f})",
                    line=dict(color=COLORS[i % len(COLORS)], width=2),
                ))
    return fig


# ---------------------------------------------------------------------------
# Predicted vs Actual (Regression)
# ---------------------------------------------------------------------------

def plot_predicted_vs_actual(y_true: np.ndarray, y_pred: np.ndarray) -> go.Figure:
    """Scatter plot of predicted vs actual values with perfect-fit line."""
    fig = _base_fig(title="Predicted vs Actual", height=500,
                    xaxis_title="Actual", yaxis_title="Predicted")
    fig.add_trace(go.Scatter(x=y_true, y=y_pred, mode="markers",
                             marker=dict(color=COLORS[0], size=6, opacity=0.6),
                             name="Predictions"))
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines",
                             name="Perfect Fit",
                             line=dict(color=COLORS[1], dash="dash", width=2)))
    return fig


# ---------------------------------------------------------------------------
# Residual Plot (Regression)
# ---------------------------------------------------------------------------

def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray) -> go.Figure:
    """Residuals vs predicted values."""
    residuals = y_true - y_pred
    fig = _base_fig(title="Residual Plot", height=500,
                    xaxis_title="Predicted", yaxis_title="Residual")
    fig.add_trace(go.Scatter(x=y_pred, y=residuals, mode="markers",
                             marker=dict(color=COLORS[0], size=6, opacity=0.6),
                             name="Residuals"))
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    return fig


# ---------------------------------------------------------------------------
# Decision Boundary
# ---------------------------------------------

def plot_decision_boundary(model: Any, X: np.ndarray, y: np.ndarray,
                           feature_names: list[str] | None = None) -> go.Figure:
    """2D decision boundary plot (uses PCA if >2 features)."""
    from sklearn.decomposition import PCA
    from sklearn.base import clone

    if X.shape[1] > 2:
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X)
        xlabel, ylabel = "PC 1", "PC 2"
    else:
        X_2d = X[:, :2].copy()
        xlabel = feature_names[0] if feature_names else "Feature 1"
        ylabel = feature_names[1] if feature_names else "Feature 2"

    # Train a clone on 2D data
    model_2d = clone(model)
    model_2d.fit(X_2d, y)

    margin = 0.5
    x_min, x_max = X_2d[:, 0].min() - margin, X_2d[:, 0].max() + margin
    y_min, y_max = X_2d[:, 1].min() - margin, X_2d[:, 1].max() + margin
    res = 150
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, res),
                         np.linspace(y_min, y_max, res))
    Z = model_2d.predict(np.c_[xx.ravel(), yy.ravel()])

    # Encode labels to integers for contour
    unique_labels = sorted(np.unique(y))
    label_map = {l: i for i, l in enumerate(unique_labels)}
    Z_int = np.array([label_map.get(z, 0) for z in Z]).reshape(xx.shape)

    fig = _base_fig(title="Decision Boundary", height=550,
                    xaxis_title=xlabel, yaxis_title=ylabel)

    fig.add_trace(go.Contour(
        x=np.linspace(x_min, x_max, res),
        y=np.linspace(y_min, y_max, res),
        z=Z_int, showscale=False, opacity=0.25,
        colorscale=[[i / max(len(unique_labels) - 1, 1), c]
                     for i, c in enumerate(COLORS[:len(unique_labels)])],
        contours=dict(showlines=False),
    ))

    for i, cls in enumerate(unique_labels):
        mask = y == cls
        fig.add_trace(go.Scatter(
            x=X_2d[mask, 0], y=X_2d[mask, 1], mode="markers",
            name=f"Class {cls}",
            marker=dict(color=COLORS[i % len(COLORS)], size=6,
                        line=dict(width=0.5, color="white")),
        ))
    return fig


# ----------------------------------------------------------------
# Feature Importance
# --------------------------------------------------

def plot_feature_importance(model: Any, feature_names: list[str]) -> go.Figure:
    """Bar chart of feature importances or coefficients."""
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
        title = "Feature Importance"
    elif hasattr(model, "coef_"):
        coef = model.coef_
        importance = np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef)
        title = "Coefficient Importance (|coef|)"
    else:
        fig = _base_fig(title="Feature Importance (unavailable)", height=400)
        fig.add_annotation(text="This model does not expose feature importances.",
                           xref="paper", yref="paper", x=0.5, y=0.5,
                           showarrow=False, font=dict(size=16))
        return fig

    # Sort descending
    idx = np.argsort(importance)[::-1]
    names_sorted = [feature_names[i] if i < len(feature_names) else f"f{i}" for i in idx]
    vals_sorted = importance[idx]

    fig = _base_fig(title=title, height=max(400, len(names_sorted) * 25),
                    xaxis_title="Importance", yaxis_title="Feature")
    fig.add_trace(go.Bar(
        y=names_sorted[::-1], x=vals_sorted[::-1], orientation="h",
        marker=dict(color=vals_sorted[::-1],
                    colorscale=[[0, "#06B6D4"], [1, "#7C3AED"]]),
    ))
    return fig


# -------------------------------------+++++++++++++++++++++++++++++++++++
# Learning Curve
# ---------------------------------------------------------------------------

def plot_learning_curve_fig(train_sizes: np.ndarray, train_scores: np.ndarray,
                            val_scores: np.ndarray) -> go.Figure:
    """Training & validation score curves with confidence bands."""
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    fig = _base_fig(title="Learning Curves", height=500,
                    xaxis_title="Training Set Size", yaxis_title="Score")

    # Training band
    fig.add_trace(go.Scatter(
        x=np.concatenate([train_sizes, train_sizes[::-1]]),
        y=np.concatenate([train_mean + train_std, (train_mean - train_std)[::-1]]),
        fill="toself", fillcolor="rgba(124,58,237,0.15)",
        line=dict(color="rgba(0,0,0,0)"), showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(x=train_sizes, y=train_mean, mode="lines+markers",
                             name="Training Score",
                             line=dict(color=COLORS[0], width=2)))

    # Validation band
    fig.add_trace(go.Scatter(
        x=np.concatenate([train_sizes, train_sizes[::-1]]),
        y=np.concatenate([val_mean + val_std, (val_mean - val_std)[::-1]]),
        fill="toself", fillcolor="rgba(6,182,212,0.15)",
        line=dict(color="rgba(0,0,0,0)"), showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(x=train_sizes, y=val_mean, mode="lines+markers",
                             name="Validation Score",
                             line=dict(color=COLORS[1], width=2)))
    return fig


# ---------------------------------------------------------------------------
# Elbow Plot (K-Means)
# ---------------------Kya be chutiye---------------------

def plot_elbow(inertias: list[float], k_range: range) -> go.Figure:
    """Elbow plot for K-Means cluster selection."""
    fig = _base_fig(title="Elbow Method", height=450,
                    xaxis_title="Number of Clusters (k)", yaxis_title="Inertia")
    fig.add_trace(go.Scatter(x=list(k_range), y=inertias, mode="lines+markers",
                             line=dict(color=COLORS[0], width=2),
                             marker=dict(size=8)))
    return fig


# ---------------------------------------------------------
# Cluster Scatter
# --------------------------------------------------------------

def plot_cluster_scatter(X_2d: np.ndarray, labels: np.ndarray) -> go.Figure:
    """2D scatter plot colored by cluster label."""
    fig = _base_fig(title="Cluster Visualization", height=500,
                    xaxis_title="Component 1", yaxis_title="Component 2")
    unique = sorted(set(labels))
    for i, cls in enumerate(unique):
        mask = labels == cls
        name = "Noise" if cls == -1 else f"Cluster {cls}"
        color = "gray" if cls == -1 else COLORS[i % len(COLORS)]
        fig.add_trace(go.Scatter(
            x=X_2d[mask, 0], y=X_2d[mask, 1], mode="markers", name=name,
            marker=dict(color=color, size=6, opacity=0.7,
                        line=dict(width=0.5, color="white")),
        ))
    return fig


# ---------------------------------------------------------------------------
# Silhouette Chart
# ---------------------------------------------------------------------------

def plot_silhouette_chart(X: np.ndarray, labels: np.ndarray) -> go.Figure:
    """Bar chart of per-cluster silhouette scores."""
    from sklearn.metrics import silhouette_samples, silhouette_score

    unique = sorted(set(labels))
    if len(unique) < 2:
        fig = _base_fig(title="Silhouette Analysis", height=400)
        fig.add_annotation(text="Need at least 2 clusters for silhouette analysis.",
                           xref="paper", yref="paper", x=0.5, y=0.5,
                           showarrow=False, font=dict(size=14))
        return fig

    sil_avg = silhouette_score(X, labels)
    sample_scores = silhouette_samples(X, labels)

    fig = _base_fig(title=f"Silhouette Analysis (avg = {sil_avg:.3f})", height=500,
                    xaxis_title="Silhouette Coefficient", yaxis_title="Cluster")

    y_lower = 0
    for i, cls in enumerate(unique):
        if cls == -1:
            continue
        cluster_scores = np.sort(sample_scores[labels == cls])
        y_upper = y_lower + len(cluster_scores)
        fig.add_trace(go.Bar(
            x=cluster_scores, y=list(range(y_lower, y_upper)),
            orientation="h", name=f"Cluster {cls}",
            marker=dict(color=COLORS[i % len(COLORS)]),
            showlegend=True,
        ))
        y_lower = y_upper + 5

    fig.add_vline(x=sil_avg, line_dash="dash", line_color="white",
                  annotation_text=f"Avg: {sil_avg:.3f}")
    fig.update_yaxes(showticklabels=False)
    return fig


# ---------------------------------------------------------------------------
# Scree Plot (PCA)
# ---------------------------------------------------------------------------

def plot_scree(explained_variance_ratio: np.ndarray) -> go.Figure:
    """Scree plot with cumulative explained variance."""
    n = len(explained_variance_ratio)
    cumulative = np.cumsum(explained_variance_ratio)
    components = list(range(1, n + 1))

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(
        x=components, y=explained_variance_ratio, name="Individual",
        marker=dict(color=COLORS[0]),
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=components, y=cumulative, mode="lines+markers", name="Cumulative",
        line=dict(color=COLORS[1], width=2),
    ), secondary_y=True)

    fig.update_layout(**_LAYOUT, title="Scree Plot (Explained Variance)",
                      height=450)
    fig.update_xaxes(title_text="Principal Component")
    fig.update_yaxes(title_text="Explained Variance Ratio", secondary_y=False)
    fig.update_yaxes(title_text="Cumulative", secondary_y=True)
    return fig


# ---------------------------------------------------------------------------
# Dimensionality Scatter
# ---------------------------------------------------------------------------

def plot_dimensionality_scatter(X_reduced: np.ndarray, y: np.ndarray | None = None,
                                method_name: str = "PCA") -> go.Figure:
    """2D or 3D scatter of dimensionality-reduced data."""
    is_3d = X_reduced.shape[1] >= 3

    if is_3d:
        fig = _base_fig(title=f"{method_name} — 3D Projection", height=600)
        if y is not None:
            unique = sorted(np.unique(y))
            for i, cls in enumerate(unique):
                mask = y == cls
                fig.add_trace(go.Scatter3d(
                    x=X_reduced[mask, 0], y=X_reduced[mask, 1], z=X_reduced[mask, 2],
                    mode="markers", name=str(cls),
                    marker=dict(color=COLORS[i % len(COLORS)], size=3),
                ))
        else:
            fig.add_trace(go.Scatter3d(
                x=X_reduced[:, 0], y=X_reduced[:, 1], z=X_reduced[:, 2],
                mode="markers", marker=dict(color=COLORS[0], size=3),
            ))
    else:
        fig = _base_fig(title=f"{method_name} — 2D Projection", height=550,
                        xaxis_title="Component 1", yaxis_title="Component 2")
        if y is not None:
            unique = sorted(np.unique(y))
            for i, cls in enumerate(unique):
                mask = y == cls
                fig.add_trace(go.Scatter(
                    x=X_reduced[mask, 0], y=X_reduced[mask, 1],
                    mode="markers", name=str(cls),
                    marker=dict(color=COLORS[i % len(COLORS)], size=6, opacity=0.7),
                ))
        else:
            fig.add_trace(go.Scatter(
                x=X_reduced[:, 0], y=X_reduced[:, 1], mode="markers",
                marker=dict(color=COLORS[0], size=6, opacity=0.7),
            ))
    return fig


# ----------------
# CV Scores
# ----------

def plot_cv_scores(scores: np.ndarray) -> go.Figure:
    """Box plot of cross-validation fold scores."""
    fig = _base_fig(title="Cross-Validation Scores", height=400,
                    yaxis_title="Score")
    fig.add_trace(go.Box(y=scores, name="CV Scores",
                         marker_color=COLORS[0],
                         boxpoints="all", jitter=0.3, pointpos=-1.8))
    return fig


# -----------------------------------------
# Model Comparison Bar Chart
# -----------------------------------------------------

def plot_comparison_bar(results_df: pd.DataFrame, metrics: list[str]) -> go.Figure:
    """Grouped bar chart comparing models across metrics."""
    fig = _base_fig(title="Model Comparison", height=500,
                    xaxis_title="Model", yaxis_title="Score")
    for i, metric in enumerate(metrics):
        if metric in results_df.columns:
            fig.add_trace(go.Bar(
                x=results_df["Model"], y=results_df[metric], name=metric,
                marker_color=COLORS[i % len(COLORS)],
            ))
    fig.update_layout(barmode="group")
    return fig


# ---------------------------------------------------
# Radar Chart (Model Comparison)
# -------------------------------------------

def plot_radar(results_df: pd.DataFrame, metrics: list[str]) -> go.Figure:
    """Radar / spider chart for multi-metric model comparison."""
    fig = _base_fig(title="Model Comparison — Radar", height=550)

    for i, row in results_df.iterrows():
        values = [row.get(m, 0) for m in metrics]
        values.append(values[0])  # close the polygon
        fig.add_trace(go.Scatterpolar(
            r=values, theta=metrics + [metrics[0]],
            fill="toself", name=row["Model"],
            line=dict(color=COLORS[i % len(COLORS)]),
            fillcolor=f"rgba({int(COLORS[i % len(COLORS)][1:3], 16)},"
                      f"{int(COLORS[i % len(COLORS)][3:5], 16)},"
                      f"{int(COLORS[i % len(COLORS)][5:7], 16)},0.1)",
        ))
    fig.update_layout(polar=dict(
        bgcolor="rgba(0,0,0,0)",
        radialaxis=dict(visible=True, range=[0, 1]),
    ))
    return fig
