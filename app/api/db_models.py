from datetime import datetime

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text
)

from app.api.database import Base


class PredictionHistory(Base):

    __tablename__ = "prediction_history"

    id = Column(
        Integer,
        primary_key=True,
        index=True
    )

    created_at = Column(
        DateTime,
        default=datetime.utcnow,
        nullable=False
    )

    tenure = Column(
        Integer,
        nullable=False
    )

    contract = Column(
        String,
        nullable=False
    )

    internetservice = Column(
        String,
        nullable=False
    )

    techsupport = Column(
        String,
        nullable=False
    )

    paymentmethod = Column(
        String,
        nullable=False
    )

    monthlycharges = Column(
        Float,
        nullable=False
    )

    totalcharges = Column(
        Float,
        nullable=False
    )

    churn_probability = Column(
        Float,
        nullable=False
    )

    prediction = Column(
        String,
        nullable=False
    )

    risk_category = Column(
        String,
        nullable=False
    )

    threshold = Column(
        Float,
        nullable=False
    )

    risk_factors = Column(
        Text,
        nullable=True
    )

    protective_factors = Column(
        Text,
        nullable=True
    )

    ai_status = Column(
        String,
        nullable=True
    )

    ai_recommendation = Column(
        Text,
        nullable=True
    )