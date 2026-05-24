"""Monitor model performance and prediction logs.

This script checks whether the trained model meets the minimum F1-score
threshold. It also reads prediction logs, if they exist, to give a simple
view of recent prediction activity.

"""

import json
from pathlib import Path

import pandas as pd

from preprocess import METRICS_PATH

#comment
PROJECT_ROOT = Path(__file__).resolve().parents[1]

LOG_PATH = PROJECT_ROOT / "logs" / "prediction_logs.csv"
MONITOR_STATUS_PATH = PROJECT_ROOT / "models" / "monitor_status.json"

DEFAULT_F1_THRESHOLD = 0.50
HIGH_RISK_PROBABILITY_THRESHOLD = 0.50


def summarise_prediction_logs() -> dict:
    """Return a simple summary of prediction logs if they exist."""
    if not LOG_PATH.exists():
        return {
            "prediction_log_found": False,
            "total_predictions": 0,
            "average_failure_probability": None,
            "high_risk_predictions": 0,
        }

    logs = pd.read_csv(LOG_PATH)

    if logs.empty:
        return {
            "prediction_log_found": True,
            "total_predictions": 0,
            "average_failure_probability": None,
            "high_risk_predictions": 0,
        }

    average_failure_probability = float(logs["failure_probability"].mean())

    high_risk_predictions = int(
        (logs["failure_probability"] >= HIGH_RISK_PROBABILITY_THRESHOLD).sum()
    )

    return {
        "prediction_log_found": True,
        "total_predictions": int(len(logs)),
        "average_failure_probability": round(average_failure_probability, 4),
        "high_risk_probability_threshold": HIGH_RISK_PROBABILITY_THRESHOLD,
        "high_risk_predictions": high_risk_predictions,
    }


def check_model_metrics() -> dict:
    """Check whether the saved model meets the acceptance threshold."""
    if not METRICS_PATH.exists():
        return {
            "metrics_found": False,
            "model_accepted": False,
            "retraining_required": True,
            "reason": "metrics.json not found. Run python src/train.py first.",
        }

    with open(METRICS_PATH, "r", encoding="utf-8") as file:
        metrics = json.load(file)

    f1_score = float(metrics.get("f1_score", 0))
    threshold = float(metrics.get("acceptance_threshold", DEFAULT_F1_THRESHOLD))

    model_accepted = f1_score >= threshold

    return {
        "metrics_found": True,
        "metric_checked": "f1_score",
        "f1_score": round(f1_score, 4),
        "acceptance_threshold": threshold,
        "model_accepted": model_accepted,
        "retraining_required": not model_accepted,
        "accuracy": round(float(metrics.get("accuracy", 0)), 4),
        "precision": round(float(metrics.get("precision", 0)), 4),
        "recall": round(float(metrics.get("recall", 0)), 4),
    }


def main() -> None:
    """Run model and prediction monitoring."""
    model_status = check_model_metrics()
    prediction_log_summary = summarise_prediction_logs()

    monitor_status = {
        "monitoring_status": "completed",
        "model_status": model_status,
        "prediction_log_summary": prediction_log_summary,
    }

    MONITOR_STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(MONITOR_STATUS_PATH, "w", encoding="utf-8") as file:
        json.dump(monitor_status, file, indent=2)

    print(json.dumps(monitor_status, indent=2))

    if model_status["retraining_required"]:
        print("Decision: retraining is required before deployment.")
    else:
        print("Decision: model passed threshold and is acceptable for deployment.")


if __name__ == "__main__":
    main()