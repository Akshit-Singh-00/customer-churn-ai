from pathlib import Path

import joblib

from app.api.services.feature_engineering import (
    build_features
)

from app.api.services.explainability import (
    create_explainer,
    load_background_data,
    explain_customer
)

# -----------------------------------
# Locate project root
# -----------------------------------

PROJECT_ROOT = Path(
    __file__
).resolve().parents[3]


# -----------------------------------
# Model file paths
# -----------------------------------

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "churn_model.joblib"
)

THRESHOLD_PATH = (
    PROJECT_ROOT
    / "models"
    / "churn_threshold.joblib"
)


# -----------------------------------
# Load model only once
# -----------------------------------

model = joblib.load(
    MODEL_PATH
)

threshold = joblib.load(
    THRESHOLD_PATH
)

# -----------------------------------
# Prepare SHAP explainer
# -----------------------------------

background_df = load_background_data(
    sample_size=100
)

(
    shap_explainer,
    shap_preprocessor,
    shap_feature_names
) = create_explainer(
    model,
    background_df
)


# -----------------------------------
# Risk category
# -----------------------------------

def get_risk_category(
    probability: float
) -> str:

    if probability >= 0.80:
        return "Critical"

    elif probability >= 0.60:
        return "High"

    elif probability >= 0.40:
        return "Medium"

    return "Low"


# -----------------------------------
# Prediction function
# -----------------------------------

def predict_customer(
    customer_data: dict
) -> dict:

    customer_df = build_features(
        customer_data
    )

    probability = model.predict_proba(
        customer_df
    )[0, 1]

    prediction_value = int(
        probability >= threshold
    )

    prediction_label = (
        "Churn"
        if prediction_value == 1
        else "Stay"
    )

    risk_category = get_risk_category(
        probability
    )

    explanation = explain_customer(
    customer_df=customer_df,
    explainer=shap_explainer,
    preprocessor=shap_preprocessor,
    feature_names=shap_feature_names,
    top_n=5
)

    return {

    "churn_probability": round(
        float(probability),
        4
    ),

    "churn_probability_percent": round(
        float(probability) * 100,
        2
    ),

    "prediction": prediction_label,

    "risk_category": risk_category,

    "threshold": round(
        float(threshold),
        4
    ),

    "top_risk_factors": explanation[
        "risk_factors"
    ],

    "top_protective_factors": explanation[
        "protective_factors"
    ]
}