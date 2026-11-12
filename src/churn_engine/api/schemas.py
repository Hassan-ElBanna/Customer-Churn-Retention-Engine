"""Pydantic request and response schemas for the FastAPI serving layer."""

from typing import Any, Dict, List
from pydantic import BaseModel, Field


class CustomerPayload(BaseModel):
    """Single customer record for real-time churn & retention scoring."""

    customer_id: str = Field(default="CUST-001", description="Unique customer identifier")
    features: Dict[str, Any] = Field(
        ...,
        description="Key-value map of raw customer attributes (e.g., tenure, MonthlyCharges, Contract)",
        examples=[
            {
                "tenure": 3,
                "MonthlyCharges": 89.5,
                "TotalCharges": 268.5,
                "Contract": "Month-to-month",
                "TechSupport": "No",
                "PaymentMethod": "Electronic check",
            }
        ],
    )


class BatchCustomerPayload(BaseModel):
    """Batch of customer records for bulk scoring."""

    customers: List[CustomerPayload]


class RetentionResponse(BaseModel):
    """Prediction & personalized retention playbook response."""

    customer_id: str
    churn_probability: float
    will_churn: bool
    risk_tier: str
    estimated_annual_value: float
    revenue_at_risk: float
    recommended_max_incentive: float
    primary_risk_drivers: List[str]
    recommended_actions: List[str]
    channel_priority: str
    model_name: str

