"""Run Models page: pick a dataset and scikit-learn algorithm, then train and evaluate it."""
import pandas as pd
import streamlit as st

from src.datasets import list_builtin_datasets, load_builtin_dataset, load_csv_dataset
from src.models import list_models, train_and_evaluate
from src.plotting import plot_clusters, plot_confusion_matrix, plot_feature_importance

st.set_page_config(page_title="Run Models", page_icon="🔬", layout="wide")
st.title("🔬 Run a scikit-learn Model")

# --- 1. Dataset selection ------------------------------------------------
st.header("1. Choose a dataset")
source = st.radio("Data source", ["Built-in dataset", "Upload CSV"], horizontal=True)

dataset = None
if source == "Built-in dataset":
    name = st.selectbox("Dataset", list_builtin_datasets())
    dataset = load_builtin_dataset(name)
    st.caption(f"Task: **{dataset.task}** · {dataset.X.shape[0]} rows · {dataset.X.shape[1]} features")
else:
    uploaded = st.file_uploader("Upload a CSV file", type="csv")
    if uploaded is not None:
        preview = pd.read_csv(uploaded)
        uploaded.seek(0)
        st.dataframe(preview.head())

        task = st.selectbox("Task", ["classification", "regression", "clustering"])
        target_column = None
        if task != "clustering":
            target_column = st.selectbox("Target column", list(preview.columns))

        try:
            dataset = load_csv_dataset(uploaded, target_column, task)
        except ValueError as exc:
            st.error(str(exc))

if dataset is not None:
    st.dataframe(dataset.X.head())

    st.header("2. Choose a model")
    model_name = st.selectbox("Algorithm", list_models(dataset.task))

    st.header("3. Options")
    test_size = 0.25
    if dataset.task != "clustering":
        test_size = st.slider("Test set size", 0.1, 0.5, 0.25, 0.05)

    if st.button("Run", type="primary"):
        with st.spinner("Training model..."):
            try:
                result = train_and_evaluate(
                    dataset.task, model_name, dataset.X, dataset.y, test_size=test_size
                )
            except Exception as exc:
                st.error(f"Training failed: {exc}")
                st.stop()

        st.header("Results")
        cols = st.columns(len(result.metrics))
        for col, (metric, value) in zip(cols, result.metrics.items()):
            col.metric(metric.replace("_", " ").title(), f"{value:.3f}")

        if dataset.task == "classification" and result.y_true is not None:
            st.pyplot(plot_confusion_matrix(result))

        if dataset.task == "clustering" and result.labels is not None:
            st.pyplot(plot_clusters(dataset, result))

        if result.feature_importances:
            st.pyplot(plot_feature_importance(result))
