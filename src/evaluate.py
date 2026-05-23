"""Evaluate the saved predictive maintenance model."""

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from preprocess import MODEL_PATH, load_dataset, prepare_features


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVALUATION_REPORT_PATH = PROJECT_ROOT / "models" / "evaluation_report.json"

MODEL_ACCEPTANCE_THRESHOLD = 0.50


def main() -> None:
    """Load the saved model, evaluate it, and save an evaluation report."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model not found. Run: python src/train.py first.")

    # load_dataset() also applies the preprocessing/cleaning stage.
    df = load_dataset()
    X, y = prepare_features(df)

    # Use the same split settings as training so evaluation is consistent.
    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = joblib.load(MODEL_PATH)
    y_pred = model.predict(X_test)

    accuracy = float(accuracy_score(y_test, y_pred))
    precision = float(precision_score(y_test, y_pred, zero_division=0))
    recall = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))

    model_accepted = f1 >= MODEL_ACCEPTANCE_THRESHOLD

    evaluation_report = {
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_path": str(MODEL_PATH),
        "test_rows": int(len(X_test)),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "acceptance_metric": "f1_score",
        "acceptance_threshold": MODEL_ACCEPTANCE_THRESHOLD,
        "model_accepted": model_accepted,
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(
            y_test,
            y_pred,
            zero_division=0,
            output_dict=True,
        ),
    }

    EVALUATION_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(EVALUATION_REPORT_PATH, "w", encoding="utf-8") as file:
        json.dump(evaluation_report, file, indent=2)

    print("Evaluation completed.")
    print(f"Evaluation report saved to: {EVALUATION_REPORT_PATH}")
    print(
        json.dumps(
            {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "acceptance_threshold": MODEL_ACCEPTANCE_THRESHOLD,
                "model_accepted": model_accepted,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()