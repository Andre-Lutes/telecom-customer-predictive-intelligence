from pathlib import Path

import pandas as pd
from sqlalchemy import text

from src.data.db_connection import get_engine


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_FILE = PROJECT_ROOT / "data" / "raw" / "Telco_customer_churn_status.xlsx"

COLUMN_MAPPING = {
    "Status ID": "status_id",
    "Customer ID": "customer_id",
    "Count": "count",
    "Quarter": "quarter",
    "Satisfaction Score": "satisfaction_score",
    "Customer Status": "customer_status",
    "Churn Label": "churn_label",
    "Churn Value": "churn_value",
    "Churn Score": "churn_score",
    "CLTV": "cltv",
    "Churn Category": "churn_category",
    "Churn Reason": "churn_reason",
}


df = pd.read_excel(SOURCE_FILE)
df = df.rename(columns=COLUMN_MAPPING)

if len(df) != 7043:
    raise ValueError(f"Expected 7043 rows, found {len(df)}.")

if df["status_id"].isna().any():
    raise ValueError("Null status_id found.")

if df["status_id"].duplicated().any():
    raise ValueError("Duplicate status_id found.")

if df["customer_id"].isna().any():
    raise ValueError("Null customer_id found.")

if df["customer_id"].duplicated().any():
    raise ValueError("Duplicate customer_id found.")

if df["churn_value"].isna().any():
    raise ValueError("Null churn_value found.")

if not df["churn_value"].isin([0, 1]).all():
    raise ValueError("Unexpected churn_value found. Expected only 0 or 1.")

engine = get_engine()

with engine.begin() as connection:
    current_rows = connection.execute(
        text("SELECT COUNT(*) FROM raw.status")
    ).scalar_one()

    if current_rows != 0:
        raise RuntimeError(
            f"raw.status already contains {current_rows} rows."
        )

    demographics_rows = connection.execute(
        text("SELECT COUNT(*) FROM raw.demographics")
    ).scalar_one()

    if demographics_rows != 7043:
        raise RuntimeError(
            f"Expected 7043 rows in raw.demographics, found {demographics_rows}."
        )

    df.to_sql(
        name="status",
        schema="raw",
        con=connection,
        if_exists="append",
        index=False,
    )

print(f"Loaded {len(df)} rows into raw.status.")