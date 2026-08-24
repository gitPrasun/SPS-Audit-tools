"""Matplotlib chart helpers shared by the Run Models and Chat Assistant pages."""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

from src.datasets import Dataset
from src.models import TrainResult


def plot_confusion_matrix(result: TrainResult):
    cm = confusion_matrix(result.y_true, result.y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    return fig


def plot_feature_importance(result: TrainResult):
    importances = pd.Series(result.feature_importances).sort_values(key=abs, ascending=True)
    fig, ax = plt.subplots(figsize=(6, max(3, 0.3 * len(importances))))
    importances.plot.barh(ax=ax)
    ax.set_xlabel("Importance / coefficient")
    ax.set_title("Feature Importance")
    return fig


def plot_clusters(dataset: Dataset, result: TrainResult):
    numeric = dataset.X.select_dtypes(include="number")
    coords = PCA(n_components=2, random_state=42).fit_transform(numeric)
    fig, ax = plt.subplots()
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=result.labels, cmap="tab10")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Clusters (PCA projection)")
    legend = ax.legend(*scatter.legend_elements(), title="Cluster")
    ax.add_artist(legend)
    return fig


def plot_result(dataset: Dataset, result: TrainResult):
    """Return the single most relevant chart for a TrainResult, or None."""
    if result.task == "classification" and result.y_true is not None:
        return plot_confusion_matrix(result)
    if result.task == "clustering" and result.labels is not None:
        return plot_clusters(dataset, result)
    if result.feature_importances:
        return plot_feature_importance(result)
    return None
