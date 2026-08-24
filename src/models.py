"""Model registry and training/evaluation helpers built on scikit-learn."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

CLASSIFIERS = {
    "Logistic Regression": LogisticRegression,
    "Random Forest": RandomForestClassifier,
    "Gradient Boosting": GradientBoostingClassifier,
    "Support Vector Machine": SVC,
    "K-Nearest Neighbors": KNeighborsClassifier,
    "Decision Tree": DecisionTreeClassifier,
}

REGRESSORS = {
    "Linear Regression": LinearRegression,
    "Ridge Regression": Ridge,
    "Random Forest": RandomForestRegressor,
    "Gradient Boosting": GradientBoostingRegressor,
    "Support Vector Machine": SVR,
    "Decision Tree": DecisionTreeRegressor,
}

CLUSTERERS = {
    "K-Means": KMeans,
    "DBSCAN": DBSCAN,
    "Agglomerative Clustering": AgglomerativeClustering,
}

REGISTRY = {"classification": CLASSIFIERS, "regression": REGRESSORS, "clustering": CLUSTERERS}


def list_models(task: str) -> list:
    if task not in REGISTRY:
        raise ValueError(f"Unknown task '{task}'. Choose from {list(REGISTRY)}")
    return list(REGISTRY[task].keys())


@dataclass
class TrainResult:
    task: str
    model_name: str
    metrics: dict
    y_true: Any = None
    y_pred: Any = None
    feature_importances: Optional[dict] = None
    labels: Any = None  # cluster assignments, only set when task == "clustering"


def train_and_evaluate(
    task: str,
    model_name: str,
    X: pd.DataFrame,
    y: Optional[pd.Series],
    test_size: float = 0.25,
    random_state: int = 42,
    model_params: Optional[dict] = None,
) -> TrainResult:
    if task not in REGISTRY:
        raise ValueError(f"Unknown task '{task}'. Choose from {list(REGISTRY)}")
    if model_name not in REGISTRY[task]:
        raise ValueError(f"Unknown model '{model_name}' for task '{task}'. Choose from {list_models(task)}")

    model_cls = REGISTRY[task][model_name]
    model = model_cls(**(model_params or {}))

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

    if task == "clustering":
        labels = model.fit_predict(X_scaled)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        metrics = {"n_clusters": float(n_clusters)}
        if n_clusters > 1:
            metrics["silhouette_score"] = float(silhouette_score(X_scaled, labels))
        return TrainResult(task=task, model_name=model_name, metrics=metrics, labels=labels)

    if y is None:
        raise ValueError("A target column is required for classification/regression.")

    stratify = y if task == "classification" and y.nunique() > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state, stratify=stratify
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if task == "classification":
        average = "binary" if y.nunique() == 2 else "macro"
        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, average=average, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, average=average, zero_division=0)),
            "f1_score": float(f1_score(y_test, y_pred, average=average, zero_division=0)),
        }
    else:
        metrics = {
            "mae": float(mean_absolute_error(y_test, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
            "r2_score": float(r2_score(y_test, y_pred)),
        }

    feature_importances = None
    if hasattr(model, "feature_importances_"):
        feature_importances = dict(zip(X.columns, (float(v) for v in model.feature_importances_)))
    elif hasattr(model, "coef_"):
        coef = np.asarray(model.coef_)
        coef = coef[0] if coef.ndim > 1 else coef
        feature_importances = dict(zip(X.columns, (float(v) for v in coef)))

    return TrainResult(
        task=task,
        model_name=model_name,
        metrics=metrics,
        y_true=y_test,
        y_pred=y_pred,
        feature_importances=feature_importances,
    )


__all__ = ["list_models", "train_and_evaluate", "TrainResult", "REGISTRY"]
