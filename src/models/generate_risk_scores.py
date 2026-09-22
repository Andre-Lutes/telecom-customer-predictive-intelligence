from pathlib import Path

import joblib
import pandas as pd
from sqlalchemy import text

from src.data.db_connection import get_engine
from src.features.build_model_dataset import (
    load_analytics_dataset,
    prepare_model_dataset,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "churn_logistic_regression.joblib"


# Load historical customer data
df = load_analytics_dataset()

customer_ids = df["customer_id"].copy()
actual_churn = df["churn_value"].copy()

X, _ = prepare_model_dataset(df)


# Load trained model artifact
artifact = joblib.load(MODEL_PATH)

pipeline = artifact["pipeline"]
prediction_threshold = artifact["threshold"]
feature_columns = artifact["feature_columns"]


# Guarantee the same feature structure used during training
X = X[feature_columns]


# Generate churn probabilities
churn_probability = pipeline.predict_proba(X)[:, 1]


scores = pd.DataFrame(
    {
        "customer_id": customer_ids,
        "churn_probability": churn_probability,
        "actual_churn_value": actual_churn,
    }
)


# Relative position of each customer in the risk portfolio
scores["risk_percentile"] = (
    scores["churn_probability"]
    .rank(method="average", pct=True)
    .mul(100)
)


# Risk deciles
scores["risk_decile"] = (
    pd.qcut(
        scores["churn_probability"].rank(method="first"),
        q=10,
        labels=False,
    )
    + 1
).astype(int)


def classify_risk(decile):
    if decile <= 3:
        return "Low"

    if decile <= 6:
        return "Medium"

    if decile <= 8:
        return "High"

    return "Critical"


scores["risk_band"] = scores["risk_decile"].apply(
    classify_risk
)


scores["prediction_threshold"] = prediction_threshold

scores["predicted_churn"] = (
    scores["churn_probability"] >= prediction_threshold
).astype(int)


scores = scores[
    [
        "customer_id",
        "churn_probability",
        "risk_percentile",
        "risk_decile",
        "risk_band",
        "prediction_threshold",
        "predicted_churn",
        "actual_churn_value",
    ]
]


# Persist scores in PostgreSQL
engine = get_engine()

with engine.begin() as connection:
    connection.execute(
        text("TRUNCATE TABLE ml.customer_churn_scores")
    )

    scores.to_sql(
        name="customer_churn_scores",
        schema="ml",
        con=connection,
        if_exists="append",
        index=False,
    )


print("=== RISK SCORING COMPLETED ===")
print(f"Model:            {artifact['model_name']}")
print(f"Customers scored: {len(scores)}")
print(f"Threshold:        {prediction_threshold:.2f}")
print(f"Features used:    {len(feature_columns)}")
print(f"Model artifact:   {MODEL_PATH}")

print("\n=== RISK BAND DISTRIBUTION ===")
print(scores["risk_band"].value_counts())

print("\n=== PREDICTED CHURN ===")
print(
    scores["predicted_churn"]
    .value_counts()
    .sort_index()
)

print("\n=== TOP 10 HIGHEST RISK CUSTOMERS ===")

print(
    scores.sort_values(
        "churn_probability",
        ascending=False,
    )
    .head(10)
    .to_string(index=False)
)