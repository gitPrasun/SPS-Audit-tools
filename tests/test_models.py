from src.datasets import load_builtin_dataset
from src.models import list_models, train_and_evaluate


def test_list_models():
    assert "Random Forest" in list_models("classification")
    assert "Linear Regression" in list_models("regression")
    assert "K-Means" in list_models("clustering")


def test_train_classification():
    ds = load_builtin_dataset("Iris")
    result = train_and_evaluate("classification", "Logistic Regression", ds.X, ds.y)
    assert 0.0 <= result.metrics["accuracy"] <= 1.0
    assert result.feature_importances is not None


def test_train_regression():
    ds = load_builtin_dataset("Diabetes")
    result = train_and_evaluate("regression", "Linear Regression", ds.X, ds.y)
    assert "r2_score" in result.metrics


def test_train_clustering():
    ds = load_builtin_dataset("Iris")
    result = train_and_evaluate(
        "clustering", "K-Means", ds.X, None, model_params={"n_clusters": 3, "n_init": 10}
    )
    assert result.labels is not None
    assert result.metrics["n_clusters"] >= 1


def test_train_missing_target_raises():
    ds = load_builtin_dataset("Iris")
    try:
        train_and_evaluate("classification", "Logistic Regression", ds.X, None)
    except ValueError:
        return
    raise AssertionError("Expected ValueError when target is missing")
