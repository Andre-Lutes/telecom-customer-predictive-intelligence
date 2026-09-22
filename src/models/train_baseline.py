import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
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
TEST_SIZE = 0.20


df = load_analytics_dataset()
X, y = prepare_model_dataset(df)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)

categorical_columns = X_train.select_dtypes(
    include=["object", "string"]
).columns.tolist()

numeric_columns = X_train.select_dtypes(
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

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]

print("=== DATA SPLIT ===")
print(f"Train rows: {len(X_train)}")
print(f"Test rows:  {len(X_test)}")

print("\n=== FEATURE TYPES ===")
print(f"Numeric features:     {len(numeric_columns)}")
print(f"Categorical features: {len(categorical_columns)}")

print("\n=== BASELINE METRICS ===")
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")
print(f"ROC AUC:   {roc_auc_score(y_test, y_proba):.4f}")
print(f"PR AUC:    {average_precision_score(y_test, y_proba):.4f}")

print("\n=== CONFUSION MATRIX ===")
print(confusion_matrix(y_test, y_pred))

print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_test, y_pred, digits=4))