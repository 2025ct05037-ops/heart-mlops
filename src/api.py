import mlflow
from fastapi import FastAPI, Response
from pydantic import BaseModel
import pandas as pd
import time
import logging
from prometheus_client import Counter, Histogram, generate_latest

app = FastAPI(title="Heart Disease Prediction API")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("heart-api")

REQUEST_COUNT = Counter("api_requests_total", "Total API requests")
REQUEST_LATENCY = Histogram("api_request_latency_seconds", "Latency of API requests")

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain")

class HeartInput(BaseModel):
    age: float
    gender: int
    cp: int
    trestbps: float
    chol: float
    fbs: int
    restecg: int
    thalach: float
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int

model = mlflow.sklearn.load_model(
    "mlruns/0/9109e4fad9f647adba7c94d16581b110/artifacts/model"
)

@app.post("/predict")
def predict(data: HeartInput):
    start_time = time.time()
    REQUEST_COUNT.inc()

    logger.info(f"Received request: {data.dict()}")

    df = pd.DataFrame([data.dict()])
    pred = model.predict(df)[0]
    prob = model.predict_proba(df)[0][1]

    logger.info(f"Prediction: {pred}, Probability: {prob}")

    REQUEST_LATENCY.observe(time.time() - start_time)

    return {
        "prediction": int(pred),
        "probability": float(prob)
    }
