# scikit-learn Workbench

A small Streamlit app for running [scikit-learn](https://github.com/scikit-learn/scikit-learn)
models against built-in or user-supplied datasets, with two UIs:

- **Run Models** — a form-based UI: pick a dataset (a built-in scikit-learn dataset or your
  own CSV upload), pick a task (classification / regression / clustering) and an algorithm,
  and get back metrics and plots (confusion matrix, feature importance, or a PCA cluster plot).
- **Chat Assistant** — a chat UI backed by the OpenAI API. Describe what you want in plain
  English (e.g. *"train a random forest on the wine dataset"*) and the assistant calls the
  same scikit-learn workflows as tools and reports back the results, including a chart.

## Project layout

```
app.py                        # Streamlit entry point / home page
pages/
  1_🔬_Run_Models.py          # Dataset + model UI
  2_💬_Chat_Assistant.py      # Chat UI
src/
  datasets.py                 # Built-in + CSV dataset loading
  models.py                   # Model registry, training & evaluation
  plotting.py                 # Shared matplotlib chart helpers
  chat_tools.py                # Tool schemas + implementations for the chat assistant
  chat_engine.py               # OpenAI API tool-call loop
tests/                         # pytest unit tests (no network/API key required)
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running the app

```bash
streamlit run app.py
```

This opens the Home page; use the sidebar to switch between **Run Models** and
**Chat Assistant**.

The **Run Models** page works out of the box — no API key needed.

The **Chat Assistant** page requires an OpenAI API key, entered directly in its sidebar
(paste it into the "OpenAI API key" field) — there's no environment variable to set up.
The key lives only in that browser session's memory (Streamlit `session_state`); it's never
written to disk, logged, or hardcoded anywhere in the code, and you'll need to re-enter it
whenever you start a new session or reload the page. Until a key is entered, the Chat
Assistant page shows a warning and the Run Models page still works normally.

By default the Chat Assistant uses the `gpt-4o` model; override this with the
`OPENAI_MODEL` environment variable (e.g. `OPENAI_MODEL=gpt-4o-mini` for a cheaper/faster
model, or any other chat-completions model your account has access to). This is just a
model name, not a secret, so it's fine to set as a regular environment variable.

## Supported datasets & tasks

Built-in datasets (via `sklearn.datasets`): Iris, Wine, Breast Cancer, Digits (all
classification), Diabetes (regression). You can also upload your own CSV and pick a
target column for classification/regression, or run clustering on its numeric columns.

Algorithms by task:

| Task | Algorithms |
| --- | --- |
| Classification | Logistic Regression, Random Forest, Gradient Boosting, SVM, K-Nearest Neighbors, Decision Tree |
| Regression | Linear Regression, Ridge Regression, Random Forest, Gradient Boosting, SVM, Decision Tree |
| Clustering | K-Means, DBSCAN, Agglomerative Clustering |

## Tests

```bash
pytest
```

Tests exercise `src/datasets.py`, `src/models.py`, and the chat tool executor directly
(training on the built-in datasets); they don't call the OpenAI API.
