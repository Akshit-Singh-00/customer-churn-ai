import json

from sqlalchemy.orm import Session

from app.api.db_models import PredictionHistory


def save_prediction(
    db: Session,
    customer_data: dict,
    result: dict
):

    record = PredictionHistory(

        tenure=customer_data["tenure"],

        contract=customer_data["contract"],

        internetservice=customer_data[
            "internetservice"
        ],

        techsupport=customer_data[
            "techsupport"
        ],

        paymentmethod=customer_data[
            "paymentmethod"
        ],

        monthlycharges=customer_data[
            "monthlycharges"
        ],

        totalcharges=customer_data[
            "totalcharges"
        ],

        churn_probability=result[
            "churn_probability"
        ],

        prediction=result[
            "prediction"
        ],

        risk_category=result[
            "risk_category"
        ],

        threshold=result[
            "threshold"
        ],

        risk_factors=json.dumps(
            result.get(
                "top_risk_factors",
                []
            )
        ),

        protective_factors=json.dumps(
            result.get(
                "top_protective_factors",
                []
            )
        ),

        ai_status=result.get(
            "ai_status"
        ),

        ai_recommendation=result.get(
            "ai_recommendation"
        )
    )

    db.add(record)
    db.commit()
    db.refresh(record)

    return record


def get_recent_predictions(
    db: Session,
    limit: int = 20
):

    return (
        db.query(PredictionHistory)
        .order_by(
            PredictionHistory.created_at.desc()
        )
        .limit(limit)
        .all()
    )


def get_high_risk_predictions(
    db: Session,
    limit: int = 20
):

    return (
        db.query(PredictionHistory)
        .filter(
            PredictionHistory.risk_category.in_(
                ["High", "Critical"]
            )
        )
        .order_by(
            PredictionHistory.churn_probability.desc()
        )
        .limit(limit)
        .all()
    )