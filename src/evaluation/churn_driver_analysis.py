import pandas as pd

from src.data.db_connection import get_engine


engine = get_engine()

df = pd.read_sql(
    "SELECT * FROM analytics.customer_churn_dataset",
    engine,
)


def show_churn_rate(column):
    result = (
        df.groupby(column, dropna=False)["churn_value"]
        .agg(
            customers="count",
            churners="sum",
            churn_rate="mean",
        )
        .reset_index()
    )

    result["churn_rate"] = (
        result["churn_rate"] * 100
    ).round(2)

    print(f"\n=== CHURN BY {column.upper()} ===")
    print(result.to_string(index=False))


print("=== OVERALL CHURN ===")
print(f"Customers:  {len(df)}")
print(f"Churners:   {df['churn_value'].sum()}")
print(f"Churn rate: {df['churn_value'].mean() * 100:.2f}%")


# Contract
show_churn_rate("contract")


# Tenure
df["tenure_band"] = pd.cut(
    df["tenure_in_months"],
    bins=[0, 6, 12, 24, 48, 72],
    labels=[
        "01-06 months",
        "07-12 months",
        "13-24 months",
        "25-48 months",
        "49-72 months",
    ],
    include_lowest=True,
)

show_churn_rate("tenure_band")


# Monthly charge quartiles
df["monthly_charge_quartile"] = pd.qcut(
    df["monthly_charge"],
    q=4,
    labels=[
        "Q1 - Lowest",
        "Q2",
        "Q3",
        "Q4 - Highest",
    ],
)

show_churn_rate("monthly_charge_quartile")


# Referrals
df["referral_band"] = pd.cut(
    df["number_of_referrals"],
    bins=[-1, 0, 2, 5, float("inf")],
    labels=[
        "0 referrals",
        "1-2 referrals",
        "3-5 referrals",
        "6+ referrals",
    ],
)

show_churn_rate("referral_band")


# Other important categorical drivers
show_churn_rate("dependents")
show_churn_rate("referred_a_friend")
show_churn_rate("internet_type")
show_churn_rate("phone_service")