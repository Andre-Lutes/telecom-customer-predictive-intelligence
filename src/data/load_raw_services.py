from pathlib import Path

import pandas as pd
from sqlalchemy import text

from src.data.db_connection import get_engine


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_FILE = PROJECT_ROOT / "data" / "raw" / "Telco_customer_churn_services.xlsx"

COLUMN_MAPPING = {
    "Service ID": "service_id",
    "Customer ID": "customer_id",
    "Count": "count",
    "Quarter": "quarter",
    "Referred a Friend": "referred_a_friend",
    "Number of Referrals": "number_of_referrals",
    "Tenure in Months": "tenure_in_months",
    "Offer": "offer",
    "Phone Service": "phone_service",
    "Avg Monthly Long Distance Charges": "avg_monthly_long_distance_charges",
    "Multiple Lines": "multiple_lines",
    "Internet Service": "internet_service",
    "Internet Type": "internet_type",
    "Avg Monthly GB Download": "avg_monthly_gb_download",
    "Online Security": "online_security",
    "Online Backup": "online_backup",
    "Device Protection Plan": "device_protection_plan",
    "Premium Tech Support": "premium_tech_support",
    "Streaming TV": "streaming_tv",
    "Streaming Movies": "streaming_movies",
    "Streaming Music": "streaming_music",
    "Unlimited Data": "unlimited_data",
    "Contract": "contract",
    "Paperless Billing": "paperless_billing",
    "Payment Method": "payment_method",
    "Monthly Charge": "monthly_charge",
    "Total Charges": "total_charges",
    "Total Refunds": "total_refunds",
    "Total Extra Data Charges": "total_extra_data_charges",
    "Total Long Distance Charges": "total_long_distance_charges",
    "Total Revenue": "total_revenue",
}


df = pd.read_excel(SOURCE_FILE)
df = df.rename(columns=COLUMN_MAPPING)

if len(df) != 7043:
    raise ValueError(f"Expected 7043 rows, found {len(df)}.")

if df["service_id"].isna().any():
    raise ValueError("Null service_id found.")

if df["service_id"].duplicated().any():
    raise ValueError("Duplicate service_id found.")

if df["customer_id"].isna().any():
    raise ValueError("Null customer_id found.")

if df["customer_id"].duplicated().any():
    raise ValueError("Duplicate customer_id found.")

engine = get_engine()

with engine.begin() as connection:
    current_rows = connection.execute(
        text("SELECT COUNT(*) FROM raw.services")
    ).scalar_one()

    if current_rows != 0:
        raise RuntimeError(
            f"raw.services already contains {current_rows} rows."
        )

    demographics_rows = connection.execute(
        text("SELECT COUNT(*) FROM raw.demographics")
    ).scalar_one()

    if demographics_rows != 7043:
        raise RuntimeError(
            f"Expected 7043 rows in raw.demographics, found {demographics_rows}."
        )

    df.to_sql(
        name="services",
        schema="raw",
        con=connection,
        if_exists="append",
        index=False,
    )

print(f"Loaded {len(df)} rows into raw.services.")