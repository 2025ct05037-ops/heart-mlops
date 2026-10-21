# Heart Disease Prediction MLOps Project

This project builds and serves a machine-learning model that predicts whether a patient record indicates the presence of heart disease. It uses the Cleveland subset of the UCI Heart Disease dataset and demonstrates an end-to-end MLOps workflow: data preparation, model training and evaluation, experiment tracking, API serving, containerization, and Kubernetes deployment.

This is an educational project, not a medical diagnostic tool. Predictions must not be used as a substitute for professional medical advice.

## What It Does

- Downloads and cleans the dataset, removes rows with missing values, and converts the target to a binary label (`0`: no disease; `1`: disease present).
- Preprocesses numeric and categorical features, then trains logistic regression and random forest classifiers.
- Evaluates models with stratified 5-fold cross-validation using accuracy, precision, recall, and ROC AUC; logs parameters, metrics, and model artifacts with MLflow.
- Serves predictions through a FastAPI endpoint and exposes request metrics at `/metrics`.
- Includes Docker Compose and Kubernetes manifests, plus a GitHub Actions workflow for linting, tests, and model training.

## Project Structure

```text
data/
  raw/                 Downloaded source data
  processed/           Cleaned data used for training
src/
  data.py              Download and clean data
  features.py          Feature preprocessing
  models.py            Classifier definitions
  train.py             Training, cross-validation, and MLflow logging
  api.py               FastAPI prediction and metrics endpoints
tests/                 Data, feature, and model tests
k8s/                   Kubernetes Deployment and Service manifests
notebooks/             Exploratory data analysis
```

## Requirements

- Python 3.10 or newer
- Internet access to download the dataset when preparing it from scratch
- Docker Desktop for the container workflow
- A Kubernetes cluster and `kubectl` for the Kubernetes workflow

## Setup and Training

From the project root, create and activate a virtual environment, then install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The cleaned dataset is included at `data/processed/heart_clean.csv`. To download the source data and recreate the cleaned file, run:

```powershell
python -m src.data
```

Train both supported classifiers and record their runs and artifacts in MLflow:

```powershell
python -m src.train
```

The training script uses `log_reg` (logistic regression) and `rf` (random forest). It writes a local model file to `mlruns/models/final_model.pkl`; MLflow run metrics and model artifacts are stored under `mlruns/`.

## Run the API

Start the API from the project root:

```powershell
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

The interactive API documentation is available at <http://localhost:8000/docs>.

### Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/predict` | Returns a binary prediction and the estimated probability for class `1` |
| `GET` | `/metrics` | Returns Prometheus-format request count and latency metrics |

Example request:

```powershell
Invoke-RestMethod -Method Post -Uri http://localhost:8000/predict -ContentType 'application/json' -Body '{"age":63,"gender":1,"cp":1,"trestbps":145,"chol":233,"fbs":1,"restecg":2,"thalach":150,"exang":0,"oldpeak":2.3,"slope":3,"ca":0,"thal":6}'
```

The API loads its model from an MLflow artifact at startup. The current artifact path is configured in `src/api.py`; if you want to serve a different training run, update that path to the desired MLflow model artifact.

## Docker Compose

Build and start the API and MLflow services:

```powershell
docker compose up --build
```

- API: <http://localhost:8000/docs>
- MLflow UI: <http://localhost:5000>

Stop the services with `Ctrl+C`, or run `docker compose down` in another terminal.

## Kubernetes

Build the image using the name expected by the Deployment, make it available to your cluster, then apply the manifests:

```powershell
docker build -t heart-api:latest .
kubectl apply -f k8s/
```

The Service is configured as a NodePort on `30080`. With a local cluster, the API is typically reachable at <http://localhost:30080/docs>; the exact address depends on your Kubernetes environment. Check status with:

```powershell
kubectl get deployments,services,pods
```

## Tests and CI

Run the unit tests and lint checks locally:

```powershell
pytest -q
flake8 src tests
```

The GitHub Actions workflow runs linting, tests, and model training on pushes and pull requests.