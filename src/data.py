import pandas as pd
from pathlib import Path

DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)

COLS = [
    "age", "gender", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"
]


def download_data(save_path="data/raw/heart.csv"):
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_URL, header=None, names=COLS)
    df.to_csv(save_path, index=False)
    print(f"Downloaded raw dataset to {save_path}")
    return df


def clean_data(load_path="data/raw/heart.csv",
               save_path="data/processed/heart_clean.csv"):
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(load_path)
    df = df.replace("?", pd.NA)
    df = df.dropna()
    df["num"] = (df["num"] > 0).astype(int)
    df.to_csv(save_path, index=False)
    print(f"Cleaned dataset saved to {save_path}")
    return df


if __name__ == "__main__":
    download_data()
    clean_data()
