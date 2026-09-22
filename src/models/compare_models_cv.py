from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features.build_model_dataset import (
    load_analytics_dataset,
    prepare_model_dataset,
)


RANDOM_STATE = 42


df = load_analytics_dataset()
X, y = prepare_model_dataset(df)

# Keep 20% separated from the model-comparison process
X_dev, X_holdout, y_dev, y_holdout = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y,
)

categorical_columns = X_dev.select_dtypes(
    include=["object", "string"]
).columns.tolist()

numeric_columns = X_dev.select_dtypes(
    include=["number"]
).columns.tolist()


def build_preprocessor():
    return ColumnTransformer(
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


models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        random_state=RANDOM_STATE,
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=400,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),
}

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE,
)

scoring = {
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
}


print("=== DEVELOPMENT DATA ===")
print(f"Rows used in CV: {len(X_dev)}")
print(f"Reserved holdout: {len(X_holdout)}")


for model_name, model in models.items():
    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", model),
        ]
    )

    results = cross_validate(
        pipeline,
        X_dev,
        y_dev,
        cv=cv,
        scoring=scoring,
        n_jobs=1,
    )

    print(f"\n=== {model_name.upper()} ===")

    for metric in scoring:
        values = results[f"test_{metric}"]

        print(
            f"{metric.upper():10s}: "
            f"{values.mean():.4f} "
            f"(+/- {values.std():.4f})"
        )