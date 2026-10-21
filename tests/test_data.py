from src.data import clean_data
import pandas as pd

def test_clean_data_no_missing():
    df = pd.read_csv("data/processed/heart_clean.csv")
    assert not df.isna().any().any()

def test_target_binary():
    df = pd.read_csv("data/processed/heart_clean.csv")
    assert set(df["num"].unique()) <= {0, 1}
