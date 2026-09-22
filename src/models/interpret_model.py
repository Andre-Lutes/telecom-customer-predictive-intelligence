import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features.build_model_dataset import (
    load_analytics_dataset,
    prepare_model_dataset,
)


RANDOM_STATE = 123


df = load_analytics_dataset()
X, y = prepare_model_dataset(df)

# Same 15% test split used in final model evaluation
X_dev, X_test, y_dev, y_test = train_test_split(
    X,
    y,
    test_size=0.15,
    random_state=RANDOM_STATE,
    stratify=y,
)

categorical_columns = X_dev.select_dtypes(
    include=["object", "string"]
).columns.tolist()

numeric_columns = X_dev.select_dtypes(
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

pipeline.fit(X_dev, y_dev)

importance = permutation_importance(
    pipeline,
    X_test,
    y_test,
    scoring="roc_auc",
    n_repeats=20,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

results = pd.DataFrame(
    {
        "feature": X_test.columns,
        "importance_mean": importance.importances_mean,
        "importance_std": importance.importances_std,
    }
)

results = results.sort_values(
    "importance_mean",
    ascending=False,
).reset_index(drop=True)

print("=== TOP 15 FEATURES — PERMUTATION IMPORTANCE ===")

print(
    results.head(15).to_string(
        index=False,
        formatters={
            "importance_mean": "{:.4f}".format,
            "importance_std": "{:.4f}".format,
        },
    )
)