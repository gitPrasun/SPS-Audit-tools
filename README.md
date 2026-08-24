# scikit-learn Workbench

A small Streamlit app for running [scikit-learn](https://github.com/scikit-learn/scikit-learn)
models against built-in or user-supplied datasets, with two UIs:

- **Run Models** — a form-based UI: pick a dataset (a built-in scikit-learn dataset or your
  own CSV upload), pick a task (classification / regression / clustering) and an algorithm,
  and get back metrics and plots (confusion matrix, feature importance, or a PCA cluster plot).
- **Chat Assistant** — a chat UI backed by the Claude API. Describe what you want in plain
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
  chat_engine.py               # Claude API tool-use loop
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

The **Chat Assistant** page requires an Anthropic API key:

```bash
cp .env.example .env   # then edit .env and set ANTHROPIC_API_KEY
export $(cat .env | xargs)   # or use your preferred way of loading env vars
streamlit run app.py
```

Without `ANTHROPIC_API_KEY` set, the Chat Assistant page shows a warning and the Run
Models page still works normally.

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
(training on the built-in datasets); they don't call the Anthropic API.
