from pathlib import Path

import pandas as pd
from sqlalchemy import text

from src.data.db_connection import get_engine


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_FILE = PROJECT_ROOT / "data" / "raw" / "Telco_customer_churn_population.xlsx"

COLUMN_MAPPING = {
    "ID": "id",
    "Zip Code": "zip_code",
    "Population": "population",
}


df = pd.read_excel(SOURCE_FILE)
df = df.rename(columns=COLUMN_MAPPING)

if len(df) != 1671:
    raise ValueError(f"Expected 1671 rows, found {len(df)}.")

if df["id"].isna().any():
    raise ValueError("Null id found.")

if df["id"].duplicated().any():
    raise ValueError("Duplicate id found.")

if df["zip_code"].isna().any():
    raise ValueError("Null zip_code found.")

# ZIP Code is an identifier, not a numeric measure.
df["zip_code"] = df["zip_code"].astype(str).str.zfill(5)

if df["zip_code"].duplicated().any():
    raise ValueError("Duplicate zip_code found.")

engine = get_engine()

with engine.begin() as connection:
    current_rows = connection.execute(
        text("SELECT COUNT(*) FROM raw.population")
    ).scalar_one()

    if current_rows != 0:
        raise RuntimeError(
            f"raw.population already contains {current_rows} rows."
        )

    df.to_sql(
        name="population",
        schema="raw",
        con=connection,
        if_exists="append",
        index=False,
    )

print(f"Loaded {len(df)} rows into raw.population.")