from pathlib import Path

import pandas as pd
from sqlalchemy import text

from src.data.db_connection import get_engine


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_FILE = PROJECT_ROOT / "data" / "raw" / "Telco_customer_churn_demographics.xlsx"

COLUMN_MAPPING = {
    "Customer ID": "customer_id",
    "Count": "count",
    "Gender": "gender",
    "Age": "age",
    "Under 30": "under_30",
    "Senior Citizen": "senior_citizen",
    "Married": "married",
    "Dependents": "dependents",
    "Number of Dependents": "number_of_dependents",
}


df = pd.read_excel(SOURCE_FILE)
df = df.rename(columns=COLUMN_MAPPING)

if len(df) != 7043:
    raise ValueError(f"Expected 7043 rows, found {len(df)}.")

if df["customer_id"].isna().any():
    raise ValueError("Null customer_id found.")

if df["customer_id"].duplicated().any():
    raise ValueError("Duplicate customer_id found.")

engine = get_engine()

with engine.begin() as connection:
    current_rows = connection.execute(
        text("SELECT COUNT(*) FROM raw.demographics")
    ).scalar_one()

    if current_rows != 0:
        raise RuntimeError(
            f"raw.demographics already contains {current_rows} rows."
        )

    df.to_sql(
        name="demographics",
        schema="raw",
        con=connection,
        if_exists="append",
        index=False,
    )

print(f"Loaded {len(df)} rows into raw.demographics.")