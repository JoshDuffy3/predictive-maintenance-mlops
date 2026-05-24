"""Train and save the predictive maintenance model."""

import json
from datetime import datetime, timezone

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

from preprocess import (
    METRICS_PATH,
    MODEL_PATH,
    build_model_pipeline,
    load_dataset,
    prepare_features,
)


# Minimum F1 score required before the model is considered acceptable.
# This is used later by the monitoring/deployment workflow.
MODEL_ACCEPTANCE_THRESHOLD = 0.30


def main() -> None:
    """Train the model, evaluate it, and save the model and metrics."""

    # load_dataset() also runs the cleaning/preprocessing stage.
    df = load_dataset()

    # Split the cleaned data into input features and target label.
    X, y = prepare_features(df)

    # Stratify keeps the failure/non-failure balance similar in train and test data.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Build and train the full preprocessing + Random Forest pipeline.
    model = build_model_pipeline(random_state=42)
    model.fit(X_train, y_train)

    # Test the model on unseen holdout data.
    y_pred = model.predict(X_test)

    accuracy = float(accuracy_score(y_test, y_pred))
    precision = float(precision_score(y_test, y_pred, zero_division=0))
    recall = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))

    model_accepted = f1 >= MODEL_ACCEPTANCE_THRESHOLD

    metrics = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_type": "RandomForestClassifier",
        "rows": int(len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "positive_failure_rate": float(y.mean()),
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

    # Save model and metrics for the Flask API, monitoring, and GitHub Actions.
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_PATH)

    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print("Training completed.")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")
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