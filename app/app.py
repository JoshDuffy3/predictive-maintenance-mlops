"""Flask API for machine failure prediction."""

import csv
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any

import joblib
import pandas as pd
from flask import Flask, jsonify, request

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from preprocess import FEATURE_COLUMNS, MODEL_PATH  # noqa: E402

app = Flask(__name__)

MODEL_FILE = Path(os.getenv("MODEL_PATH", MODEL_PATH))
LOG_PATH = PROJECT_ROOT / "logs" / "prediction_logs.csv"

FRIENDLY_INPUT_MAPPING = {
    "type": "Type",
    "air_temperature": "Air temperature [K]",
    "process_temperature": "Process temperature [K]",
    "rotational_speed": "Rotational speed [rpm]",
    "torque": "Torque [Nm]",
    "tool_wear": "Tool wear [min]",
}


def load_model():
    """Load the model if it exists. Return None if it has not been trained yet."""
    if MODEL_FILE.exists():
        return joblib.load(MODEL_FILE)
    return None


model = load_model()


def build_model_input(payload: Dict[str, Any]) -> pd.DataFrame:
    """Convert API JSON into the exact model feature columns."""
    row = {}

    for api_key, model_column in FRIENDLY_INPUT_MAPPING.items():
        if model_column in payload:
            row[model_column] = payload[model_column]
        elif api_key in payload:
            row[model_column] = payload[api_key]
        else:
            raise ValueError(f"Missing required input: {api_key}")

    row["Type"] = str(row["Type"]).upper()
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


def log_prediction(payload: Dict[str, Any], prediction: int, probability: float) -> None:
    """Append prediction details to a CSV log for simple monitoring."""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    file_exists = LOG_PATH.exists()

    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "timestamp_utc",
                "type",
                "air_temperature",
                "process_temperature",
                "rotational_speed",
                "torque",
                "tool_wear",
                "failure_prediction",
                "failure_probability",
            ],
        )
        if not file_exists:
            writer.writeheader()

        writer.writerow(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "type": payload.get("type", payload.get("Type")),
                "air_temperature": payload.get("air_temperature", payload.get("Air temperature [K]")),
                "process_temperature": payload.get("process_temperature", payload.get("Process temperature [K]")),
                "rotational_speed": payload.get("rotational_speed", payload.get("Rotational speed [rpm]")),
                "torque": payload.get("torque", payload.get("Torque [Nm]")),
                "tool_wear": payload.get("tool_wear", payload.get("Tool wear [min]")),
                "failure_prediction": prediction,
                "failure_probability": probability,
            }
        )


@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "ok",
            "model_loaded": model is not None,
            "model_path": str(MODEL_FILE),
        }
    )


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not found. Run: python src/train.py"}), 503

    payload = request.get_json(silent=True)
    if not payload:
        return jsonify({"error": "Request body must be valid JSON"}), 400

    try:
        X = build_model_input(payload)
        prediction = int(model.predict(X)[0])

        if hasattr(model, "predict_proba"):
            class_labels = list(model.classes_)
            if 1 in class_labels:
                failure_index = class_labels.index(1)
                probability = float(model.predict_proba(X)[0][failure_index])
            else:
                probability = float(prediction)
        else:
            probability = float(prediction)

        log_prediction(payload, prediction, probability)

        return jsonify(
            {
                "failure_prediction": prediction,
                "failure_label": "Failure risk" if prediction == 1 else "No failure predicted",
                "failure_probability": round(probability, 4),
                "model": "RandomForestClassifier",
            }
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:  # defensive for demo stability
        return jsonify({"error": f"Prediction failed: {exc}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
