# Predictive Maintenance MLOps Pipeline

This project predicts machine failure risk from sensor readings using a Random Forest classifier.
The model is served through a Flask API.
GitHub Actions automates testing, training, Docker image building, and deployment.


## Local setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Download data

```bash
python src/download_data.py
```

## Train model

```bash
python src/train.py
```

This creates:

```text
models/model.joblib
models/metrics.json
```

## Run tests

```bash
pytest
```

## Run Flask API

```bash
python app/app.py
```

Open a second terminal and test:

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"type":"M","air_temperature":298.1,"process_temperature":308.6,"rotational_speed":1551,"torque":42.8,"tool_wear":0}'
```

## Run monitoring check

```bash
python src/monitor.py
```

## Docker

Build image:

```bash
docker build -t predictive-maintenance-api .
```

Run container:

```bash
docker run -p 5000:5000 predictive-maintenance-api
```
