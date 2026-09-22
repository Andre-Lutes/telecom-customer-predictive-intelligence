import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
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


RANDOM_STATE = 42


def build_pipeline(X):
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

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


df = load_analytics_dataset()
X, y = prepare_model_dataset(df)

# 80% development / 20% final test
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y,
)

# Development set becomes 60% train / 20% validation
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval,
    y_trainval,
    test_size=0.25,
    random_state=RANDOM_STATE,
    stratify=y_trainval,
)

pipeline = build_pipeline(X_train)
pipeline.fit(X_train, y_train)

val_proba = pipeline.predict_proba(X_val)[:, 1]

thresholds = np.arange(0.05, 0.81, 0.01)

results = []

for threshold in thresholds:
    val_pred = (val_proba >= threshold).astype(int)

    precision = precision_score(
        y_val,
        val_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_val,
        val_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_val,
        val_pred,
        zero_division=0,
    )

    f2 = fbeta_score(
        y_val,
        val_pred,
        beta=2,
        zero_division=0,
    )

    results.append(
        (threshold, precision, recall, f1, f2)
    )

best_result = max(
    results,
    key=lambda row: row[4],
)

best_threshold = best_result[0]

print("=== DATA SPLIT ===")
print(f"Train:      {len(X_train)}")
print(f"Validation: {len(X_val)}")
print(f"Test:       {len(X_test)}")

print("\n=== BEST VALIDATION THRESHOLD ===")
print(f"Threshold: {best_result[0]:.2f}")
print(f"Precision: {best_result[1]:.4f}")
print(f"Recall:    {best_result[2]:.4f}")
print(f"F1:        {best_result[3]:.4f}")
print(f"F2:        {best_result[4]:.4f}")

# Refit using the complete 80% development sample
final_pipeline = build_pipeline(X_trainval)
final_pipeline.fit(X_trainval, y_trainval)

test_proba = final_pipeline.predict_proba(X_test)[:, 1]

default_pred = (test_proba >= 0.50).astype(int)
tuned_pred = (test_proba >= best_threshold).astype(int)

print("\n=== FINAL TEST — DEFAULT THRESHOLD 0.50 ===")
print(f"Accuracy:  {accuracy_score(y_test, default_pred):.4f}")
print(f"Precision: {precision_score(y_test, default_pred):.4f}")
print(f"Recall:    {recall_score(y_test, default_pred):.4f}")
print(f"F1:        {f1_score(y_test, default_pred):.4f}")
print(f"F2:        {fbeta_score(y_test, default_pred, beta=2):.4f}")

print("\n=== FINAL TEST — TUNED THRESHOLD ===")
print(f"Threshold: {best_threshold:.2f}")
print(f"Accuracy:  {accuracy_score(y_test, tuned_pred):.4f}")
print(f"Precision: {precision_score(y_test, tuned_pred):.4f}")
print(f"Recall:    {recall_score(y_test, tuned_pred):.4f}")
print(f"F1:        {f1_score(y_test, tuned_pred):.4f}")
print(f"F2:        {fbeta_score(y_test, tuned_pred, beta=2):.4f}")
print(f"ROC AUC:   {roc_auc_score(y_test, test_proba):.4f}")