import numpy as np

from app.app import app


class DummyModel:
    classes_ = np.array([0, 1])

    def predict(self, X):
        return np.array([1])

    def predict_proba(self, X):
        return np.array([[0.2, 0.8]])


def test_predict_endpoint_returns_prediction(monkeypatch):
    import app.app as flask_app_module

    monkeypatch.setattr(flask_app_module, "model", DummyModel())

    client = app.test_client()
    response = client.post(
        "/predict",
        json={
            "type": "M",
            "air_temperature": 298.1,
            "process_temperature": 308.6,
            "rotational_speed": 1551,
            "torque": 42.8,
            "tool_wear": 0,
        },
    )

    assert response.status_code == 200
    result = response.get_json()
    assert result["failure_prediction"] == 1
    assert result["failure_probability"] == 0.8


def test_predict_endpoint_rejects_missing_input(monkeypatch):
    import app.app as flask_app_module

    monkeypatch.setattr(flask_app_module, "model", DummyModel())

    client = app.test_client()
    response = client.post("/predict", json={"type": "M"})

    assert response.status_code == 400
    assert "Missing required input" in response.get_json()["error"]
