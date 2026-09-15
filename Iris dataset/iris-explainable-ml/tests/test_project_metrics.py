import numpy as np

from src.data_loader import check_missing_values, get_raw_dataframe
from src.evaluate import compute_classification_metrics, compute_rss


def test_missing_values_are_zero():
    df = get_raw_dataframe()
    results = check_missing_values(df)
    assert results["total_missing"] == 0
    assert results["missing_by_column"] == {}


def test_classification_and_rss_metrics():
    y_true = np.array([0, 1, 1, 2, 0, 2, 1, 0])
    y_pred = np.array([0, 1, 0, 2, 0, 2, 1, 1])
    y_proba = np.array([
        [0.90, 0.05, 0.05],
        [0.10, 0.80, 0.10],
        [0.45, 0.40, 0.15],
        [0.08, 0.10, 0.82],
        [0.88, 0.08, 0.04],
        [0.05, 0.12, 0.83],
        [0.15, 0.70, 0.15],
        [0.81, 0.10, 0.09],
    ])

    metrics = compute_classification_metrics(y_true, y_pred, y_proba=y_proba)
    assert metrics["f1_weighted"] > 0.7
    assert metrics["roc_auc_macro"] > 0.7

    rss = compute_rss(np.array([1.0, 2.0, 3.0]), np.array([0.8, 2.1, 2.7]))
    assert rss > 0
