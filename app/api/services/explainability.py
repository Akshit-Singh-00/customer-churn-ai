from pathlib import Path

import numpy as np
import pandas as pd
import shap

from app.api.services.feature_engineering import (
    build_features_dataframe
)

def create_explainer(model, background_df):

    preprocessor = model.named_steps[
        "preprocessor"
    ]

    classifier = model.named_steps[
        "classifier"
    ]

    background_transformed = (
        preprocessor.transform(
            background_df
        )
    )

    if hasattr(
        background_transformed,
        "toarray"
    ):
        background_transformed = (
            background_transformed.toarray()
        )

    feature_names = (
        preprocessor
        .get_feature_names_out()
    )

    clean_feature_names = [
        name
        .replace("num__", "")
        .replace("cat__", "")
        for name in feature_names
    ]

    explainer = shap.Explainer(
        classifier,
        background_transformed,
        feature_names=clean_feature_names
    )

    return (
        explainer,
        preprocessor,
        clean_feature_names
    )

def load_background_data(
    sample_size: int = 100
):

    project_root = Path(
        __file__
    ).resolve().parents[3]

    data_path = (
        project_root
        / "data"
        / "processed"
        / "telco_customer_churn_clean.csv"
    )

    df = pd.read_csv(
        data_path
    )

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
    )

    # Create exactly the same features
    # used during model training

    df = build_features_dataframe(
        df
    )

    # Remove columns the model never used
    df = df.drop(
        columns=[
            "customerid",
            "churn",
            "churn_flag"
        ]
    )

    # Small sample is enough for SHAP background

    if len(df) > sample_size:

        df = df.sample(
            n=sample_size,
            random_state=42
        )

    return df

def explain_customer(
    customer_df,
    explainer,
    preprocessor,
    feature_names,
    top_n=5
):

    # Transform customer using
    # the fitted preprocessing pipeline

    customer_transformed = (
        preprocessor.transform(
            customer_df
        )
    )

    if hasattr(
        customer_transformed,
        "toarray"
    ):
        customer_transformed = (
            customer_transformed.toarray()
        )


    # Calculate SHAP values

    shap_values = explainer(
        customer_transformed
    )


    values = shap_values.values


    # Some binary classifiers return:
    #
    # (rows, features, classes)
    #
    # We want class 1 = Churn

    if values.ndim == 3:

        values = values[
            :,
            :,
            1
        ]


    # Get first customer's values

    customer_shap = values[0]


    # Create readable table

    explanation_df = pd.DataFrame(
        {
            "feature": feature_names,
            "shap_value": customer_shap
        }
    )


    # ---------------------------------
    # Risk increasing features
    # ---------------------------------

    risk_factors = (
        explanation_df[
            explanation_df[
                "shap_value"
            ] > 0
        ]
        .sort_values(
            "shap_value",
            ascending=False
        )
        .head(top_n)
    )


    # ---------------------------------
    # Protective features
    # ---------------------------------

    protective_factors = (
        explanation_df[
            explanation_df[
                "shap_value"
            ] < 0
        ]
        .sort_values(
            "shap_value",
            ascending=True
        )
        .head(top_n)
    )


    risk_list = []

    for _, row in risk_factors.iterrows():

        risk_list.append(
            {
                "feature": row["feature"],
                "impact": round(
                    float(
                        row["shap_value"]
                    ),
                    4
                )
            }
        )


    protective_list = []

    for _, row in protective_factors.iterrows():

        protective_list.append(
            {
                "feature": row["feature"],
                "impact": round(
                    float(
                        row["shap_value"]
                    ),
                    4
                )
            }
        )


    return {
        "risk_factors": risk_list,
        "protective_factors": protective_list
    }