from pathlib import Path
import os

import requests
import streamlit as st
from dotenv import load_dotenv


PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]


load_dotenv(
    PROJECT_ROOT / ".env"
)


def get_api_url():

    try:
        return st.secrets[
            "FASTAPI_URL"
        ]

    except Exception:

        return os.getenv(
            "FASTAPI_URL",
            "http://127.0.0.1:8000"
        )


API_URL = get_api_url()


# --------------------------------------------------
# Predict churn
# --------------------------------------------------

def predict_churn(
    customer_data: dict
):

    endpoint = (
        f"{API_URL}/predict"
    )

    try:

        response = requests.post(
            endpoint,
            json=customer_data,
            timeout=120
        )

        response.raise_for_status()

        return {
            "success": True,
            "data": response.json()
        }

    except requests.exceptions.RequestException as e:

        return {
            "success": False,
            "error": str(e)
        }


# --------------------------------------------------
# Get prediction history
# --------------------------------------------------

def get_prediction_history(
    limit=20
):

    endpoint = (
        f"{API_URL}/history"
    )

    try:

        response = requests.get(
            endpoint,
            params={
                "limit": limit
            },
            timeout=30
        )

        response.raise_for_status()

        return {
            "success": True,
            "data": response.json()
        }

    except requests.exceptions.RequestException as e:

        return {
            "success": False,
            "error": str(e)
        }


# --------------------------------------------------
# Get high-risk history
# --------------------------------------------------

def get_high_risk_history(
    limit=20
):

    endpoint = (
        f"{API_URL}/history/high-risk"
    )

    try:

        response = requests.get(
            endpoint,
            params={
                "limit": limit
            },
            timeout=30
        )

        response.raise_for_status()

        return {
            "success": True,
            "data": response.json()
        }

    except requests.exceptions.RequestException as e:

        return {
            "success": False,
            "error": str(e)
        }