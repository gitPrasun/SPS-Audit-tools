from src.datasets import list_builtin_datasets, load_builtin_dataset


def test_list_builtin_datasets():
    names = list_builtin_datasets()
    assert "Iris" in names
    assert "Diabetes" in names


def test_load_builtin_dataset_classification():
    ds = load_builtin_dataset("Iris")
    assert ds.task == "classification"
    assert ds.X.shape[0] == 150
    assert ds.y is not None
    assert ds.target_names is not None


def test_load_builtin_dataset_regression():
    ds = load_builtin_dataset("Diabetes")
    assert ds.task == "regression"
    assert ds.y is not None


def test_load_builtin_dataset_unknown_raises():
    try:
        load_builtin_dataset("Not A Dataset")
    except ValueError:
        return
    raise AssertionError("Expected ValueError for unknown dataset")
