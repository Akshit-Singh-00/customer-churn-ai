from fastapi import (
    Depends,
    FastAPI
)

from app.api.database import (
    Base,
    engine,
    get_db
)

from app.api.services.history import (
    get_high_risk_predictions,
    get_recent_predictions,
    save_prediction
)

from app.api import db_models

from sqlalchemy.orm import Session

from app.api.schemas import (
    CustomerInput
)

from app.api.services.prediction import (
    predict_customer
)

Base.metadata.create_all(
    bind=engine
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
    customer: CustomerInput,
    db: Session = Depends(
        get_db
    )
):

    customer_data = (
        customer.model_dump()
    )

    result = predict_customer(
        customer_data
    )

    record = save_prediction(
        db=db,
        customer_data=customer_data,
        result=result
    )

    result[
        "prediction_id"
    ] = record.id

    return result

@app.get("/history")
def prediction_history(
    limit: int = 20,
    db: Session = Depends(
        get_db
    )
):

    records = get_recent_predictions(
        db,
        limit
    )

    return [
        {
            "id": record.id,

            "created_at":
                record.created_at,

            "tenure":
                record.tenure,

            "contract":
                record.contract,

            "internetservice":
                record.internetservice,

            "techsupport":
                record.techsupport,

            "paymentmethod":
                record.paymentmethod,

            "monthlycharges":
                record.monthlycharges,

            "totalcharges":
                record.totalcharges,

            "churn_probability":
                record.churn_probability,

            "prediction":
                record.prediction,

            "risk_category":
                record.risk_category,

            "ai_status":
                record.ai_status
        }

        for record in records
    ]

@app.get(
    "/history/high-risk"
)
def high_risk_history(
    limit: int = 20,
    db: Session = Depends(
        get_db
    )
):

    records = (
        get_high_risk_predictions(
            db,
            limit
        )
    )

    return [
        {
            "id": record.id,

            "created_at":
                record.created_at,

            "tenure":
                record.tenure,

            "contract":
                record.contract,

            "churn_probability":
                record.churn_probability,

            "prediction":
                record.prediction,

            "risk_category":
                record.risk_category
        }

        for record in records
    ]