import json
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

try:
    from src.data_loader import check_missing_values, load_data
    from src.evaluate import compute_classification_metrics, save_metrics
except ModuleNotFoundError:
    from data_loader import check_missing_values, load_data
    from evaluate import compute_classification_metrics, save_metrics

MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.joblib")
RESULTS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "reports", "evaluation_metrics.json"))


def plot_correlation_heatmap(df, output_path: str):
    corr = df.drop(columns=["species"], errors="ignore").corr(numeric_only=True)
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def train():
    print("Loading data...")
    X_train, X_test, y_train, y_test, target_names, df = load_data()
    missing_summary = check_missing_values(df)
    print("Missing value summary:", missing_summary)

    print("Training RandomForestClassifier...")
    model = RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42)
    model.fit(X_train, y_train)

    print("\n--- Model Evaluation ---")
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)
    metrics = compute_classification_metrics(y_test.to_numpy(), preds, y_proba=probs, labels=np.array([0, 1, 2]))
    print("Classification Report:\n", classification_report(y_test, preds, target_names=target_names, digits=4))
    print("Confusion Matrix:\n", confusion_matrix(y_test, preds))
    print("Classification metrics:", json.dumps(metrics, indent=2))

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    save_metrics(metrics, RESULTS_PATH)

    heatmap_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "reports", "feature_correlation_heatmap.png"))
    plot_correlation_heatmap(df, heatmap_path)
    print(f"\nModel written successfully to: {MODEL_PATH}")
    print(f"Evaluation metrics saved to: {RESULTS_PATH}")
    print(f"Heatmap saved to: {heatmap_path}")


if __name__ == "__main__":
    train()