import os
from typing import Dict

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "iris.csv"))


def get_raw_dataframe() -> pd.DataFrame:
    save_raw_data()
    return pd.read_csv(DATA_PATH)


def save_raw_data() -> str:
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    if not os.path.exists(DATA_PATH) or os.path.getsize(DATA_PATH) == 0:
        iris = load_iris(as_frame=True)
        df = iris.frame
        df["species"] = df["target"].map(dict(enumerate(iris.target_names)))
        df.to_csv(DATA_PATH, index=False)
    return DATA_PATH


def check_missing_values(df: pd.DataFrame) -> Dict[str, object]:
    missing_by_column = df.isna().sum()
    missing_by_column = missing_by_column[missing_by_column > 0].to_dict()
    return {
        "total_missing": int(df.isna().sum().sum()),
        "missing_by_column": missing_by_column,
        "missing_percentage": {k: round(v / len(df) * 100, 4) for k, v in missing_by_column.items()},
    }


def load_data(test_size: float = 0.2, random_state: int = 42):
    df = get_raw_dataframe()
    feature_cols = [col for col in df.columns if col not in ["target", "species"]]
    X = df[feature_cols]
    y = df["target"]
    target_names = df["species"].unique().tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test, target_names, df


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, targets, df = load_data()
    print(f"Data ready. Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(check_missing_values(df))