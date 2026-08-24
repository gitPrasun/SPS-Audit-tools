"""Chat Assistant page: talk to Claude to drive the scikit-learn workflows conversationally."""
import os

import pandas as pd
import streamlit as st

from src.chat_engine import run_turn
from src.plotting import plot_result

st.set_page_config(page_title="Chat Assistant", page_icon="💬", layout="wide")
st.title("💬 Chat Assistant")
st.caption(
    "Ask me to explore a dataset or train a model, e.g. "
    "\"train a random forest on the wine dataset\" or \"cluster the iris dataset with k-means\"."
)

if "workspace" not in st.session_state:
    st.session_state.workspace = {}
if "anthropic_messages" not in st.session_state:
    st.session_state.anthropic_messages = []

with st.sidebar:
    st.subheader("Upload a dataset")
    uploaded = st.file_uploader("CSV file", type="csv", key="chat_uploader")
    if uploaded is not None:
        st.session_state.workspace["uploaded_df"] = pd.read_csv(uploaded)
        st.success(f"Loaded {uploaded.name} — refer to it as 'uploaded' in chat.")
    if st.button("Clear conversation"):
        st.session_state.anthropic_messages = []
        st.session_state.workspace.pop("last_result", None)
        st.session_state.workspace.pop("last_dataset", None)
        st.rerun()

if not os.environ.get("ANTHROPIC_API_KEY"):
    st.warning(
        "Set the `ANTHROPIC_API_KEY` environment variable to enable the chat assistant. "
        "Use the Run Models page in the meantime."
    )
    st.stop()

for message in st.session_state.anthropic_messages:
    if message["role"] == "assistant":
        text = "".join(b.text for b in message["content"] if getattr(b, "type", None) == "text")
        if text:
            with st.chat_message("assistant"):
                st.markdown(text)
    elif message["role"] == "user" and isinstance(message["content"], str):
        with st.chat_message("user"):
            st.markdown(message["content"])

prompt = st.chat_input("Ask about a dataset or model...")
if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.anthropic_messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                st.session_state.anthropic_messages = run_turn(
                    st.session_state.anthropic_messages, st.session_state.workspace
                )
            except Exception as exc:
                st.error(f"The assistant hit an error: {exc}")
                st.stop()

        last = st.session_state.anthropic_messages[-1]
        text = "".join(b.text for b in last["content"] if getattr(b, "type", None) == "text")
        st.markdown(text or "_(no response text)_")

    result = st.session_state.workspace.get("last_result")
    dataset = st.session_state.workspace.get("last_dataset")
    if result is not None and dataset is not None:
        fig = plot_result(dataset, result)
        if fig is not None:
            st.pyplot(fig)
