import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features.build_model_dataset import (
    load_analytics_dataset,
    prepare_model_dataset,
)


RANDOM_STATE = 123


def build_preprocessor(X):
    categorical_columns = X.select_dtypes(
        include=["object", "string"]
    ).columns.tolist()

    numeric_columns = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

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


def find_best_threshold(y_true, probabilities):
    thresholds = np.arange(0.05, 0.81, 0.01)

    results = []

    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)

        f2 = fbeta_score(
            y_true,
            predictions,
            beta=2,
            zero_division=0,
        )

        results.append((threshold, f2))

    return max(results, key=lambda row: row[1])[0]


def evaluate_model(name, model, X_train, y_train, X_val, y_val, X_test, y_test):
    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X_train)),
            ("model", model),
        ]
    )

    pipeline.fit(X_train, y_train)

    val_probabilities = pipeline.predict_proba(X_val)[:, 1]

    best_threshold = find_best_threshold(
        y_val,
        val_probabilities,
    )

    test_probabilities = pipeline.predict_proba(X_test)[:, 1]

    test_predictions = (
        test_probabilities >= best_threshold
    ).astype(int)

    print(f"\n=== {name.upper()} ===")
    print(f"Validation threshold: {best_threshold:.2f}")
    print(f"Accuracy:  {accuracy_score(y_test, test_predictions):.4f}")
    print(f"Precision: {precision_score(y_test, test_predictions):.4f}")
    print(f"Recall:    {recall_score(y_test, test_predictions):.4f}")
    print(f"F1:        {f1_score(y_test, test_predictions):.4f}")
    print(
        f"F2:        "
        f"{fbeta_score(y_test, test_predictions, beta=2):.4f}"
    )
    print(f"ROC AUC:   {roc_auc_score(y_test, test_probabilities):.4f}")
    print(
        f"PR AUC:    "
        f"{average_precision_score(y_test, test_probabilities):.4f}"
    )


df = load_analytics_dataset()
X, y = prepare_model_dataset(df)

# 15% final test
X_dev, X_test, y_dev, y_test = train_test_split(
    X,
    y,
    test_size=0.15,
    random_state=RANDOM_STATE,
    stratify=y,
)

# Remaining 85% becomes approximately 70% train / 15% validation
validation_fraction = 0.15 / 0.85

X_train, X_val, y_train, y_val = train_test_split(
    X_dev,
    y_dev,
    test_size=validation_fraction,
    random_state=RANDOM_STATE,
    stratify=y_dev,
)

print("=== FINAL DATA SPLIT ===")
print(f"Train:      {len(X_train)}")
print(f"Validation: {len(X_val)}")
print(f"Test:       {len(X_test)}")

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

for model_name, model in models.items():
    evaluate_model(
        model_name,
        model,
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
    )