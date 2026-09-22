from pathlib import Path

import joblib

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features.build_model_dataset import (
    load_analytics_dataset,
    prepare_model_dataset,
)


RANDOM_STATE = 123
PREDICTION_THRESHOLD = 0.10

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "churn_logistic_regression.joblib"


df = load_analytics_dataset()
X, y = prepare_model_dataset(df)

categorical_columns = X.select_dtypes(
    include=["object", "string"]
).columns.tolist()

numeric_columns = X.select_dtypes(
    include=["number"]
).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns,
        ),
        (
            "numeric",
            StandardScaler(),
            numeric_columns,
        ),
    ]
)

model = LogisticRegression(
    max_iter=2000,
    random_state=RANDOM_STATE,
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model),
    ]
)

pipeline.fit(X, y)

artifact = {
    "pipeline": pipeline,
    "threshold": PREDICTION_THRESHOLD,
    "model_name": "Logistic Regression",
    "random_state": RANDOM_STATE,
    "feature_columns": X.columns.tolist(),
}

joblib.dump(
    artifact,
    MODEL_PATH,
)

print("=== MODEL SAVED ===")
print(f"Model:      {artifact['model_name']}")
print(f"Threshold:  {artifact['threshold']:.2f}")
print(f"Features:   {len(artifact['feature_columns'])}")
print(f"Train rows: {len(X)}")
print(f"Path:       {MODEL_PATH}")