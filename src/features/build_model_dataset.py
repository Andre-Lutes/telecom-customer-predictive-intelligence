import pandas as pd

from src.data.db_connection import get_engine


TARGET_COLUMN = "churn_value"

COLUMNS_TO_DROP = [
    "customer_id",
    "country",
    "state",
    "city",
    "zip_code",
]


def load_analytics_dataset() -> pd.DataFrame:
    engine = get_engine()

    query = """
        SELECT *
        FROM analytics.customer_churn_dataset
    """

    return pd.read_sql(query, engine)


def prepare_model_dataset(df: pd.DataFrame):
    data = df.copy()

    # Structural missing values
    data["offer"] = data["offer"].fillna("No Offer")
    data["internet_type"] = data["internet_type"].fillna("No Internet")

    # Columns excluded from the baseline predictive model
    data = data.drop(columns=COLUMNS_TO_DROP)

    X = data.drop(columns=[TARGET_COLUMN])
    y = data[TARGET_COLUMN]

    return X, y


if __name__ == "__main__":
    df = load_analytics_dataset()
    X, y = prepare_model_dataset(df)

    print("=== ORIGINAL DATASET ===")
    print(df.shape)

    print("\n=== MODEL FEATURES ===")
    print(X.shape)

    print("\n=== TARGET ===")
    print(y.shape)

    print("\n=== NULL VALUES AFTER PREPARATION ===")
    print(X.isna().sum()[X.isna().sum() > 0])

    print("\n=== TARGET DISTRIBUTION ===")
    print(y.value_counts().sort_index())

    print("\n=== FEATURE COLUMNS ===")
    print(X.columns.tolist())