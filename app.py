"""Home page for the scikit-learn Workbench Streamlit app."""
import streamlit as st

st.set_page_config(page_title="scikit-learn Workbench", page_icon="🔬", layout="wide")

st.title("🔬 scikit-learn Workbench")
st.markdown(
    """
Welcome! This app lets you run [scikit-learn](https://github.com/scikit-learn/scikit-learn)
models against built-in or your own datasets, and includes a chat assistant that can run
the same workflows for you conversationally.

Use the sidebar to navigate:

- **Run Models** — pick a dataset (built-in or your own CSV), choose a task and algorithm,
  and see metrics and plots.
- **Chat Assistant** — describe what you want in plain English and let the assistant train
  and evaluate models for you.
"""
)

st.info(
    "The Chat Assistant page requires an `OPENAI_API_KEY` environment variable. "
    "Without it, use the Run Models page directly — no API key needed."
)
