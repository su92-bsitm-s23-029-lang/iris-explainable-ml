import os
import joblib
import shap
import matplotlib.pyplot as plt

try:
    from src.data_loader import load_data
except ModuleNotFoundError:
    from data_loader import load_data

MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "best_model.joblib"))

def explain():
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) == 0:
        raise FileNotFoundError("Trained weights missing or 0 bytes. Run train.py first!")

    model = joblib.load(MODEL_PATH)
    X_train, X_test, y_train, y_test, target_names, _ = load_data()

    print("\n--- Feature Importances ---")
    for feature, importance in zip(X_train.columns, model.feature_importances_):
        print(f"{feature:20s}: {importance:.4f}")

    print("\nGenerating SHAP explanations...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    plt.figure(figsize=(8, 5))
    shap.summary_plot(shap_values, X_test, class_names=target_names, show=False)
    plt.title("SHAP Global Feature Impact per Class", fontsize=12)
    plt.tight_layout()
    plt.savefig("shap_summary.png", dpi=300)
    print("SHAP summary plot saved to root as 'shap_summary.png'.")
    plt.show()

if __name__ == "__main__":
    explain()
    