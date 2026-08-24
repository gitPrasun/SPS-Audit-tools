import json

from src.chat_tools import make_executor


def test_list_datasets_tool():
    execute = make_executor({})
    payload = json.loads(execute("list_datasets", {}))
    assert "Iris" in payload["builtin_datasets"]
    assert payload["uploaded_available"] is False


def test_describe_dataset_tool():
    execute = make_executor({})
    payload = json.loads(execute("describe_dataset", {"dataset": "Wine"}))
    assert payload["task"] == "classification"
    assert payload["shape"][0] > 0


def test_train_model_tool():
    execute = make_executor({})
    payload = json.loads(
        execute(
            "train_model",
            {"dataset": "Iris", "task": "classification", "model": "Logistic Regression"},
        )
    )
    assert "accuracy" in payload["metrics"]


def test_train_model_tool_uploaded_without_upload_errors():
    execute = make_executor({})
    try:
        execute(
            "train_model",
            {"dataset": "uploaded", "task": "classification", "model": "Logistic Regression"},
        )
    except ValueError:
        return
    raise AssertionError("Expected ValueError when no CSV has been uploaded")
