import json
from typing import Any, Dict, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)


def compute_rss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute residual sum of squares for regression-style error evaluation."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residuals = y_true - y_pred
    return float(np.sum(np.square(residuals)))


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    labels: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Return a richer set of classification metrics, including F1 and ROC-AUC."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0,
    )

    metrics: Dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(np.mean(precision)),
        "recall_macro": float(np.mean(recall)),
        "f1_macro": float(np.mean(f1)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }

    if y_proba is not None:
        y_proba = np.asarray(y_proba)
        if y_proba.ndim == 1:
            metrics["roc_auc_ovr"] = float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro"))
            metrics["brier_score"] = float(np.mean(np.square(y_proba - y_true)))
        else:
            metrics["roc_auc_macro"] = float(
                roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro", labels=labels)
            )
            try:
                metrics["log_loss"] = float(log_loss(y_true, y_proba, labels=labels))
            except ValueError:
                metrics["log_loss"] = None

    return metrics


def save_metrics(metrics: Dict[str, Any], output_path: str) -> str:
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return output_path
