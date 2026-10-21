from src.features import build_preprocessor
import pandas as pd

def test_preprocessor_output_shape():
    df = pd.read_csv("data/processed/heart_clean.csv")
    X = df.drop(columns=["num"])
    pre = build_preprocessor()
    Xt = pre.fit_transform(X)
    assert Xt.shape[0] == X.shape[0]
