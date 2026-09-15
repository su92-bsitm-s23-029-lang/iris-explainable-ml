# Iris Explainable ML Project

This project is an end-to-end machine learning pipeline for the classic Iris dataset, with a stronger focus on robust validation, model evaluation, feature diagnostics, and explanation. It now includes missing-value checks, feature-correlation heatmaps, richer classification metrics, ROC-AUC analysis, residual sum of squares, and SHAP-based explainability.

---

## What is included

- Data ingestion and validation from the Iris dataset
- Missing-value detection and percentage reporting
- Correlation heatmap for feature relationships
- Model training with a Random Forest classifier
- Standard evaluation: accuracy, precision, recall, F1-score, confusion matrix
- Additional metrics: macro ROC-AUC, log loss, and residual sum of squares
- Model explainability with SHAP feature contribution plots
- Saved reports in a dedicated output folder
- Regression-style validation tests for the core metric logic

---

## Project structure

```text
iris-explainable-ml/
├── data/
│   └── iris.csv
├── models/
│   └── best_model.joblib
├── notebooks/
│   └── exploration_and_eda.ipynb
├── reports/
│   ├── evaluation_metrics.json
│   └── feature_correlation_heatmap.png
├── src/
│   ├── data_loader.py
│   ├── evaluate.py
│   ├── explain.py
│   ├── train.py
│   └── __init__.py
├── tests/
│   └── test_project_metrics.py
├── requirements.txt
├── README.md
└── shap_summary.png
```

---

## Data validation

The data pipeline checks the dataset before modeling:

- Missing-value count by column
- Missing-value percentage by column
- Total missing-value count
- Automatic dataset generation if the CSV file is missing or empty

This is implemented in the data-loading and validation layer so that data quality issues are caught before training starts.

---

## Feature analysis and heatmap

The project generates a correlation heatmap to visualize relationships among the numeric features. This helps identify which attributes are strongly associated and helps interpret model behavior.

Example output:

- Feature correlation heatmap saved in the reports folder
- Useful for understanding patterns such as petal width and petal length correlation

---

## Model and evaluation metrics

The model is trained using a Random Forest classifier on a stratified split of the Iris dataset.

### Metrics now tracked

- Accuracy
- Precision (macro)
- Recall (macro)
- F1-score (macro)
- F1-score (weighted)
- Confusion matrix
- ROC-AUC (macro)
- Log loss
- Residual sum of squares (RSS)

### Example evaluation output

```text
Classification Report:
              precision    recall  f1-score   support

      setosa     1.0000    1.0000    1.0000        10
  versicolor     0.8182    0.9000    0.8571        10
   virginica     0.8889    0.8000    0.8421        10

    accuracy                         0.9000        30
   macro avg     0.9024    0.9000    0.8997        30
weighted avg     0.9024    0.9000    0.8997        30
```

### Additional model statistics

- ROC-AUC macro: around 0.987 on the test split
- Log loss: low and stable for the multiclass task
- RSS is included as a residual-based error measure in the evaluation module

---

## Explainability

The explainability layer uses SHAP to show how the model makes decisions.

- SHAP summary plot generated for the test set
- Feature importances printed for each input attribute
- Output image saved locally for inspection and documentation

---

## Getting started

1. Create a virtual environment and activate it.
2. Install the dependencies.
3. Run the training pipeline.
4. Review the generated metrics and heatmap files in the reports directory.

---

## Typical workflow

- Load and validate the dataset
- Train the model
- Evaluate classification performance
- Export model and metrics
- Generate explainability visualizations
- Review the correlation heatmap and feature importance signals

---

## Notes

The project is intentionally modular so it can be extended with more models, class balancing, hyperparameter tuning, and richer dashboard-style reporting in future iterations.

---

## Next Steps & Roadmap

- [ ] Add cross-validation (`StratifiedKFold`) and hyperparameter tuning (`GridSearchCV`).
- [ ] Integrate experiment logging (e.g., MLflow).
- [ ] Serve inference via an interactive FastAPI endpoint.
