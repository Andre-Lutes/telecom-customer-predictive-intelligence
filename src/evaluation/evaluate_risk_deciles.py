import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features.build_model_dataset import (
    load_analytics_dataset,
    prepare_model_dataset,
)


RANDOM_STATE = 123


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

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE,
)

oof_probabilities = cross_val_predict(
    pipeline,
    X,
    y,
    cv=cv,
    method="predict_proba",
    n_jobs=1,
)[:, 1]

results = pd.DataFrame(
    {
        "customer_id": df["customer_id"],
        "churn_value": y,
        "churn_probability": oof_probabilities,
    }
)

# Decile 10 = highest risk
results["risk_decile"] = pd.qcut(
    results["churn_probability"].rank(method="first"),
    q=10,
    labels=False,
) + 1

overall_churn_rate = results["churn_value"].mean()
total_churners = results["churn_value"].sum()

decile_summary = (
    results.groupby("risk_decile")
    .agg(
        customers=("customer_id", "count"),
        churners=("churn_value", "sum"),
        avg_probability=("churn_probability", "mean"),
        churn_rate=("churn_value", "mean"),
    )
    .reset_index()
    .sort_values("risk_decile", ascending=False)
)

decile_summary["churn_rate_pct"] = (
    decile_summary["churn_rate"] * 100
)

decile_summary["capture_rate_pct"] = (
    decile_summary["churners"] / total_churners * 100
)

decile_summary["lift"] = (
    decile_summary["churn_rate"] / overall_churn_rate
)

decile_summary["cumulative_churners"] = (
    decile_summary["churners"].cumsum()
)

decile_summary["cumulative_capture_pct"] = (
    decile_summary["cumulative_churners"]
    / total_churners
    * 100
)

print("=== OVERALL ===")
print(f"Customers:          {len(results)}")
print(f"Total churners:     {total_churners}")
print(f"Overall churn rate: {overall_churn_rate * 100:.2f}%")

print("\n=== OOF RISK DECILES ===")

print(
    decile_summary[
        [
            "risk_decile",
            "customers",
            "churners",
            "avg_probability",
            "churn_rate_pct",
            "capture_rate_pct",
            "lift",
            "cumulative_capture_pct",
        ]
    ].to_string(
        index=False,
        formatters={
            "avg_probability": "{:.4f}".format,
            "churn_rate_pct": "{:.2f}%".format,
            "capture_rate_pct": "{:.2f}%".format,
            "lift": "{:.2f}x".format,
            "cumulative_capture_pct": "{:.2f}%".format,
        },
    )
)