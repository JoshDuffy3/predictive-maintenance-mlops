"""Preprocessing and model setup for the predictive maintenance pipeline."""

import json
from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "ai4i2020.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed_ai4i2020.csv"
PREPROCESSING_REPORT_PATH = PROJECT_ROOT / "data" / "preprocessing_report.json"

MODEL_PATH = PROJECT_ROOT / "models" / "model.joblib"
METRICS_PATH = PROJECT_ROOT / "models" / "metrics.json"

TARGET_COLUMN = "Machine failure"

FEATURE_COLUMNS = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

NUMERIC_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

CATEGORICAL_FEATURES = ["Type"]

REQUIRED_COLUMNS = FEATURE_COLUMNS + [TARGET_COLUMN]


def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the raw dataset, clean it, and save a preprocessing report."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run: python src/download_data.py"
        )

    df = pd.read_csv(path)

    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    report = {
        "rows_before_cleaning": int(len(df)),
        "duplicates_before_cleaning": int(df[REQUIRED_COLUMNS].duplicated().sum()),
        "missing_values_before_cleaning": df[REQUIRED_COLUMNS].isna().sum().to_dict(),
    }

    # Keep only the columns used by the model.
    df = df[REQUIRED_COLUMNS].copy()

    # Remove duplicate training rows.
    df = df.drop_duplicates()

    # Clean the machine type column.
    df["Type"] = df["Type"].astype(str).str.strip().str.upper()
    df["Type"] = df["Type"].replace({"NAN": "M", "": "M"})

    # Convert numeric columns and fill missing values with the column median.
    for column in NUMERIC_FEATURES:
        df[column] = pd.to_numeric(df[column], errors="coerce")
        df[column] = df[column].fillna(df[column].median())

    # Clean the target column.
    df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
    df = df.dropna(subset=[TARGET_COLUMN])
    df = df[df[TARGET_COLUMN].isin([0, 1])]
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)

    report["rows_after_cleaning"] = int(len(df))
    report["missing_values_after_cleaning"] = df.isna().sum().to_dict()
    report["target_distribution"] = df[TARGET_COLUMN].value_counts().to_dict()

    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(PROCESSED_DATA_PATH, index=False)

    with open(PREPROCESSING_REPORT_PATH, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    return df


def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Split the cleaned dataset into features and target."""
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].astype(int).copy()

    return X, y


def build_preprocessor() -> ColumnTransformer:
    """Build preprocessing for categorical and numeric features."""
    return ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("numeric", "passthrough", NUMERIC_FEATURES),
        ]
    )


def build_model_pipeline(random_state: int = 42) -> Pipeline:
    """Create the full preprocessing and Random Forest model pipeline."""
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=random_state,
        class_weight="balanced",
    )

    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", model),
        ]
    )


def main() -> None:
    """Run preprocessing as a standalone stage."""
    df = load_dataset()

    print("Preprocessing completed.")
    print(f"Rows after cleaning: {len(df)}")
    print(f"Processed data saved to: {PROCESSED_DATA_PATH}")
    print(f"Preprocessing report saved to: {PREPROCESSING_REPORT_PATH}")


if __name__ == "__main__":
    main()