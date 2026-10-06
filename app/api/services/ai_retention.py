from pathlib import Path
import os

from dotenv import load_dotenv
from google import genai


# -----------------------------------------
# Load environment variables
# -----------------------------------------

PROJECT_ROOT = Path(
    __file__
).resolve().parents[3]

load_dotenv(
    PROJECT_ROOT / ".env"
)


GEMINI_API_KEY = os.getenv(
    "GEMINI_API_KEY"
)

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.8-flash"
)


# -----------------------------------------
# Create Gemini client
# -----------------------------------------

client = None

if GEMINI_API_KEY:

    client = genai.Client(
        api_key=GEMINI_API_KEY
    )


# -----------------------------------------
# Convert SHAP factors to text
# -----------------------------------------

def factors_to_text(
    factors: list
) -> str:

    if not factors:
        return "None identified."

    lines = []

    for factor in factors:

        lines.append(
            f"- {factor['feature']} "
            f"(model impact: "
            f"{factor['impact']})"
        )

    return "\n".join(lines)


# -----------------------------------------
# Generate retention advice
# -----------------------------------------

def generate_retention_advice(
    customer_data: dict,
    probability: float,
    prediction: str,
    risk_category: str,
    risk_factors: list,
    protective_factors: list
) -> dict:

    # -------------------------------------
    # No API key
    # -------------------------------------

    if client is None:

        return {
            "status": "unavailable",
            "recommendation": (
                "AI recommendation is unavailable "
                "because the Gemini API key "
                "is not configured."
            )
        }


    # -------------------------------------
    # Prepare SHAP factors
    # -------------------------------------

    risk_text = factors_to_text(
        risk_factors
    )

    protective_text = factors_to_text(
        protective_factors
    )


    # -------------------------------------
    # Customer summary
    # -------------------------------------

    customer_summary = f"""
Tenure: {customer_data['tenure']} months
Contract: {customer_data['contract']}
Internet Service: {customer_data['internetservice']}
Monthly Charges: {customer_data['monthlycharges']}
Total Charges: {customer_data['totalcharges']}
Technical Support: {customer_data['techsupport']}
Online Security: {customer_data['onlinesecurity']}
Online Backup: {customer_data['onlinebackup']}
Payment Method: {customer_data['paymentmethod']}
Paperless Billing: {customer_data['paperlessbilling']}
"""


    # -------------------------------------
    # Prompt
    # -------------------------------------

    prompt = f"""
You are an AI customer-retention assistant
for a telecom company.

A machine-learning model has already analyzed
this customer.

IMPORTANT RULES:

1. Never change or recalculate the churn probability.
2. Do not invent customer information.
3. SHAP factors explain model behavior;
   do not claim they prove causation.
4. Recommend only practical retention actions.
5. Do not base retention actions on gender,
   age, or demographic characteristics.
6. Keep the response concise.
7. Give exactly three retention actions.

MODEL RESULT

Churn Probability:
{probability * 100:.2f}%

Prediction:
{prediction}

Risk Category:
{risk_category}


CUSTOMER INFORMATION

{customer_summary}


FACTORS INCREASING THE MODEL'S
CHURN PREDICTION

{risk_text}


FACTORS REDUCING THE MODEL'S
CHURN PREDICTION

{protective_text}


Return:

Customer Risk Summary:
Explain the risk in 2-3 sentences.

Main Risk Factors:
Explain the important model factors simply.

Recommended Retention Actions:
Give exactly 3 practical actions.

Intervention Priority:
Low, Medium, High, or Critical.
"""


    # -------------------------------------
    # Call Gemini safely
    # -------------------------------------

    try:

        response = (
            client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt
            )
        )

        return {
            "status": "available",
            "recommendation": response.text
        }


    except Exception:

        return {
            "status": "unavailable",
            "recommendation": (
                "AI recommendation is temporarily "
                "unavailable. The machine-learning "
                "prediction and SHAP explanation "
                "are still available."
            )
        }