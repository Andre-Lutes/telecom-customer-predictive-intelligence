from pathlib import Path

import pandas as pd
from sqlalchemy import text

from src.data.db_connection import get_engine


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_FILE = PROJECT_ROOT / "data" / "raw" / "Telco_customer_churn_location.xlsx"

COLUMN_MAPPING = {
    "Location ID": "location_id",
    "Customer ID": "customer_id",
    "Count": "count",
    "Country": "country",
    "State": "state",
    "City": "city",
    "Zip Code": "zip_code",
    "Lat Long": "lat_long",
    "Latitude": "latitude",
    "Longitude": "longitude",
}


df = pd.read_excel(SOURCE_FILE)
df = df.rename(columns=COLUMN_MAPPING)

if len(df) != 7043:
    raise ValueError(f"Expected 7043 rows, found {len(df)}.")

if df["location_id"].isna().any():
    raise ValueError("Null location_id found.")

if df["location_id"].duplicated().any():
    raise ValueError("Duplicate location_id found.")

if df["customer_id"].isna().any():
    raise ValueError("Null customer_id found.")

if df["customer_id"].duplicated().any():
    raise ValueError("Duplicate customer_id found.")

if df["zip_code"].isna().any():
    raise ValueError("Null zip_code found.")

# ZIP Code is an identifier, not a numeric measure.
df["zip_code"] = df["zip_code"].astype(str).str.zfill(5)

engine = get_engine()

with engine.begin() as connection:
    current_rows = connection.execute(
        text("SELECT COUNT(*) FROM raw.location")
    ).scalar_one()

    if current_rows != 0:
        raise RuntimeError(
            f"raw.location already contains {current_rows} rows."
        )

    demographics_rows = connection.execute(
        text("SELECT COUNT(*) FROM raw.demographics")
    ).scalar_one()

    population_rows = connection.execute(
        text("SELECT COUNT(*) FROM raw.population")
    ).scalar_one()

    if demographics_rows != 7043:
        raise RuntimeError(
            f"Expected 7043 rows in raw.demographics, found {demographics_rows}."
        )

    if population_rows != 1671:
        raise RuntimeError(
            f"Expected 1671 rows in raw.population, found {population_rows}."
        )

    df.to_sql(
        name="location",
        schema="raw",
        con=connection,
        if_exists="append",
        index=False,
    )

print(f"Loaded {len(df)} rows into raw.location.")