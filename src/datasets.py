"""Dataset loading utilities: built-in scikit-learn datasets and user-uploaded CSVs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd
from sklearn import datasets as sk_datasets

_BUILTIN_LOADERS = {
    "Iris": (sk_datasets.load_iris, "classification"),
    "Wine": (sk_datasets.load_wine, "classification"),
    "Breast Cancer": (sk_datasets.load_breast_cancer, "classification"),
    "Digits": (sk_datasets.load_digits, "classification"),
    "Diabetes": (sk_datasets.load_diabetes, "regression"),
}


@dataclass
class Dataset:
    name: str
    X: pd.DataFrame
    y: Optional[pd.Series]
    task: str  # "classification", "regression", or "clustering"
    target_names: Optional[list] = None


def list_builtin_datasets() -> list:
    return list(_BUILTIN_LOADERS.keys())


def load_builtin_dataset(name: str) -> Dataset:
    if name not in _BUILTIN_LOADERS:
        raise ValueError(f"Unknown built-in dataset: {name}. Choose from {list_builtin_datasets()}")
    loader, task = _BUILTIN_LOADERS[name]
    bunch = loader(as_frame=True)
    X = bunch.data
    y = bunch.target
    target_names = None
    if task == "classification" and hasattr(bunch, "target_names"):
        target_names = [str(t) for t in bunch.target_names]
    return Dataset(name=name, X=X, y=y, task=task, target_names=target_names)


def load_csv_dataset(file, target_column: Optional[str], task: str) -> Dataset:
    """Load a user-supplied CSV. `file` is a path or file-like object."""
    df = pd.read_csv(file)

    if target_column is None:
        if task != "clustering":
            raise ValueError("A target column is required for classification/regression.")
        numeric = df.select_dtypes(include="number")
        return Dataset(name="Uploaded CSV", X=numeric, y=None, task=task)

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found. Available columns: {list(df.columns)}")

    y = df[target_column]
    X = df.drop(columns=[target_column]).select_dtypes(include="number")
    if X.isna().any().any():
        X = X.fillna(X.mean(numeric_only=True))

    target_names = None
    if task == "classification":
        target_names = sorted(y.astype(str).unique().tolist())
    return Dataset(name="Uploaded CSV", X=X, y=y, task=task, target_names=target_names)
