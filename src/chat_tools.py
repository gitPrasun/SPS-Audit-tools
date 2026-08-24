"""Tool definitions the chat assistant can call to drive the scikit-learn workflow.

Each tool is an OpenAI function-calling schema dict (chat.completions `tools`
format); `make_executor` binds the tool implementations to a per-session
workspace dict so the chat page can keep state (an uploaded CSV, the last
training result) across turns without any global mutable state.
"""
from __future__ import annotations

import json
from typing import Callable, Optional

from src.datasets import Dataset, list_builtin_datasets, load_builtin_dataset
from src.models import list_models, train_and_evaluate

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_datasets",
            "description": (
                "List the built-in scikit-learn datasets available (Iris, Wine, Breast Cancer, "
                "Digits, Diabetes), plus whether a CSV was uploaded by the user in this session."
            ),
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "describe_dataset",
            "description": "Get the shape, columns, and dtypes for a dataset by name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "dataset": {
                        "type": "string",
                        "description": "A built-in dataset name (e.g. 'Iris') or 'uploaded' for the user's uploaded CSV.",
                    }
                },
                "required": ["dataset"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_models",
            "description": "List the scikit-learn algorithms available for a given task.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {"type": "string", "enum": ["classification", "regression", "clustering"]}
                },
                "required": ["task"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "train_model",
            "description": (
                "Train and evaluate a scikit-learn model on a dataset and return its metrics. "
                "For dataset 'uploaded', target_column is required unless task is 'clustering'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "dataset": {
                        "type": "string",
                        "description": "A built-in dataset name (e.g. 'Iris') or 'uploaded'.",
                    },
                    "task": {
                        "type": "string",
                        "enum": ["classification", "regression", "clustering"],
                    },
                    "model": {"type": "string", "description": "Algorithm name, e.g. 'Random Forest'."},
                    "target_column": {
                        "type": "string",
                        "description": "Target column name; required for 'uploaded' on classification/regression.",
                    },
                    "test_size": {
                        "type": "number",
                        "description": "Fraction of the data held out for testing (default 0.25).",
                    },
                },
                "required": ["dataset", "task", "model"],
                "additionalProperties": False,
            },
        },
    },
]


def _resolve_dataset(
    name: str, workspace: dict, task: str, target_column: Optional[str]
) -> Dataset:
    if name.lower() == "uploaded":
        df = workspace.get("uploaded_df")
        if df is None:
            raise ValueError("No CSV has been uploaded in this session yet.")
        if task == "clustering":
            numeric = df.select_dtypes(include="number")
            return Dataset(name="Uploaded CSV", X=numeric, y=None, task=task)
        if not target_column:
            raise ValueError("target_column is required for the uploaded dataset on this task.")
        if target_column not in df.columns:
            raise ValueError(f"Column '{target_column}' not found. Available: {list(df.columns)}")
        y = df[target_column]
        X = df.drop(columns=[target_column]).select_dtypes(include="number")
        return Dataset(name="Uploaded CSV", X=X, y=y, task=task)

    return load_builtin_dataset(name)


def make_executor(workspace: dict) -> Callable[[str, dict], str]:
    """Build a tool executor bound to the given session workspace."""

    def execute(name: str, tool_input: dict) -> str:
        if name == "list_datasets":
            return json.dumps(
                {
                    "builtin_datasets": list_builtin_datasets(),
                    "uploaded_available": workspace.get("uploaded_df") is not None,
                }
            )

        if name == "describe_dataset":
            ds_name = tool_input["dataset"]
            if ds_name.lower() == "uploaded":
                df = workspace.get("uploaded_df")
                if df is None:
                    return json.dumps({"error": "No CSV uploaded yet."})
                return json.dumps(
                    {
                        "columns": list(df.columns),
                        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
                        "shape": list(df.shape),
                    }
                )
            ds = load_builtin_dataset(ds_name)
            return json.dumps(
                {
                    "task": ds.task,
                    "shape": list(ds.X.shape),
                    "columns": list(ds.X.columns),
                    "target_names": ds.target_names,
                }
            )

        if name == "list_models":
            return json.dumps({"models": list_models(tool_input["task"])})

        if name == "train_model":
            ds = _resolve_dataset(
                tool_input["dataset"],
                workspace,
                tool_input["task"],
                tool_input.get("target_column"),
            )
            result = train_and_evaluate(
                task=tool_input["task"],
                model_name=tool_input["model"],
                X=ds.X,
                y=ds.y,
                test_size=tool_input.get("test_size", 0.25),
            )
            workspace["last_result"] = result
            workspace["last_dataset"] = ds
            return json.dumps({"metrics": result.metrics})

        raise ValueError(f"Unknown tool: {name}")

    return execute
