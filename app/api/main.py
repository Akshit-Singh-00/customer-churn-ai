from fastapi import FastAPI

from app.api.schemas import (
    CustomerInput
)

from app.api.services.prediction import (
    predict_customer
)


app = FastAPI(
    title="AI Customer Churn API",

    description=(
        "Machine learning API for predicting "
        "customer churn risk."
    ),

    version="1.0.0"
)


# -----------------------------------
# Home
# -----------------------------------

@app.get("/")
def home():

    return {
        "message": (
            "AI Customer Churn API is running"
        )
    }


# -----------------------------------
# Health check
# -----------------------------------

@app.get("/health")
def health_check():

    return {
        "status": "healthy"
    }


# -----------------------------------
# Prediction endpoint
# -----------------------------------

@app.post("/predict")
def predict(
    customer: CustomerInput
):

    customer_data = (
        customer.model_dump()
    )

    result = predict_customer(
        customer_data
    )

    return result