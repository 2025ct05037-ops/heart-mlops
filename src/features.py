from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def build_preprocessor():
    numeric_features = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    categorical_features = ["gender", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ]
    )

    return preprocessor
