import numpy as np
import pandas as pd


def build_features_dataframe(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    # ---------------------------------
    # Tenure group
    # ---------------------------------

    df["tenure_group"] = pd.cut(
        df["tenure"],
        bins=[
            -1,
            6,
            12,
            24,
            48,
            72
        ],
        labels=[
            "0-6",
            "7-12",
            "13-24",
            "25-48",
            "49-72"
        ]
    )


    # ---------------------------------
    # Total services
    # ---------------------------------

    df["total_services"] = 0

    df["total_services"] += (
        df["phoneservice"] == "Yes"
    ).astype(int)

    df["total_services"] += (
        df["multiplelines"] == "Yes"
    ).astype(int)

    df["total_services"] += (
        df["internetservice"] != "No"
    ).astype(int)


    service_columns = [
        "onlinesecurity",
        "onlinebackup",
        "deviceprotection",
        "techsupport",
        "streamingtv",
        "streamingmovies"
    ]

    for column in service_columns:

        df["total_services"] += (
            df[column] == "Yes"
        ).astype(int)


    # ---------------------------------
    # Average monthly spend
    # ---------------------------------

    df["avg_monthly_spend"] = np.where(
        df["tenure"] > 0,
        df["totalcharges"] / df["tenure"],
        0
    )


    # ---------------------------------
    # Automatic payment
    # ---------------------------------

    automatic_methods = [
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]

    df["automatic_payment"] = (
        df["paymentmethod"]
        .isin(automatic_methods)
        .astype(int)
    )


    # ---------------------------------
    # Month-to-month contract
    # ---------------------------------

    df["is_month_to_month"] = (
        df["contract"] == "Month-to-month"
    ).astype(int)


    # ---------------------------------
    # Technical support
    # ---------------------------------

    df["has_tech_support"] = (
        df["techsupport"] == "Yes"
    ).astype(int)


    return df


def build_features(customer_data: dict) -> pd.DataFrame:

    df = pd.DataFrame(
        [customer_data]
    )

    return build_features_dataframe(df)