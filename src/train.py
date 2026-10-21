import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate

from src.features import build_preprocessor
from src.models import get_log_reg, get_random_forest


def train_model(model_name="log_reg"):
    df = pd.read_csv("data/processed/heart_clean.csv")
    X = df.drop(columns=["num"])
    y = df["num"]

    if model_name == "log_reg":
        model = get_log_reg()
    elif model_name == "rf":
        model = get_random_forest()
    else:
        raise ValueError("Invalid model name.")

    preprocessor = build_preprocessor()
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "roc_auc": "roc_auc"
    }

    with mlflow.start_run(run_name=model_name):
        mlflow.log_param("model_name", model_name)
        mlflow.log_params(model.get_params())

        scores = cross_validate(pipeline, X, y, cv=cv, scoring=scoring)

        for metric, values in scores.items():
            if metric.startswith("test_"):
                mlflow.log_metric(metric, values.mean())

        pipeline.fit(X, y)
        mlflow.sklearn.log_model(pipeline, "model")

        # Save local model artifact
        joblib.dump(pipeline, "mlruns/models/final_model.pkl")
        print(f"Training complete for {model_name}")
        return pipeline


if __name__ == "__main__":
    train_model("log_reg")
    train_model("rf")
