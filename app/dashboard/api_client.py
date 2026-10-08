from pathlib import Path
import os

import requests
from dotenv import load_dotenv


# ----------------------------------
# Load environment variables
# ----------------------------------

PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]

load_dotenv(
    PROJECT_ROOT / ".env"
)


API_URL = os.getenv(
    "FASTAPI_URL",
    "http://127.0.0.1:8000"
)


# ----------------------------------
# Prediction API call
# ----------------------------------

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
            timeout=60
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