"""Phase 4: Customer Retention & Next-Best-Action Recommendation Engine."""

from dataclasses import dataclass, asdict
from typing import Any, Dict, List


@dataclass
class RetentionPlan:
    """Actionable retention plan tailored to a single customer's churn profile."""

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

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class RetentionEngine:
    """
    Translates raw churn probabilities and customer attributes into
    ROI-positive retention interventions and prioritized actions.
    """

    def __init__(
        self,
        low_threshold: float = 0.30,
        medium_threshold: float = 0.60,
        high_threshold: float = 0.80,
        default_annual_revenue: float = 1200.0,
        max_offer_budget_ratio: float = 0.25,
    ):
        self.low_threshold = low_threshold
        self.medium_threshold = medium_threshold
        self.high_threshold = high_threshold
        self.default_annual_revenue = default_annual_revenue
        self.max_offer_budget_ratio = max_offer_budget_ratio

    def classify_risk_tier(self, churn_prob: float) -> str:
        """Assign customer to Low, Medium, High, or Critical risk tier."""
        if churn_prob >= self.high_threshold:
            return "Critical"
        if churn_prob >= self.medium_threshold:
            return "High"
        if churn_prob >= self.low_threshold:
            return "Medium"
        return "Low"

    def estimate_annual_value(self, customer_features: Dict[str, Any]) -> float:
        """Estimate Customer Annual Recurring Revenue (ARR) from billing attributes."""
        lower_map = {k.lower(): v for k, v in customer_features.items() if v is not None}

        if "monthlycharges" in lower_map:
            try:
                return round(float(lower_map["monthlycharges"]) * 12.0, 2)
            except (TypeError, ValueError):
                pass

        if "estimatedsalary" in lower_map or "balance" in lower_map:
            try:
                balance = float(lower_map.get("balance", 0.0) or 0.0)
                return round(max(balance * 0.03, self.default_annual_revenue), 2)
            except (TypeError, ValueError):
                pass

        return self.default_annual_revenue

    @staticmethod
    def identify_risk_drivers(customer_features: Dict[str, Any]) -> List[str]:
        """Inspect customer attributes for known churn risk triggers."""
        drivers: List[str] = []
        lower_map = {k.lower(): v for k, v in customer_features.items() if v is not None}

        contract = str(lower_map.get("contract", "")).lower()
        if "month-to-month" in contract or contract == "monthly":
            drivers.append("Month-to-month contract (lack of long-term lock-in)")

        tenure = lower_map.get("tenure")
        if tenure is not None:
            try:
                if float(tenure) <= 6:
                    drivers.append("Early onboarding phase (tenure <= 6 months)")
            except (TypeError, ValueError):
                pass

        monthly_charges = lower_map.get("monthlycharges")
        if monthly_charges is not None:
            try:
                if float(monthly_charges) >= 80.0:
                    drivers.append("High monthly billing tier (>= $80/mo)")
            except (TypeError, ValueError):
                pass

        tech_support = str(lower_map.get("techsupport", "")).lower()
        if tech_support == "no":
            drivers.append("No active Tech Support subscription")

        payment = str(lower_map.get("paymentmethod", "")).lower()
        if "electronic check" in payment or "manual" in payment:
            drivers.append("Non-automated payment method (higher friction)")

        num_products = lower_map.get("numofproducts")
        if num_products is not None:
            try:
                if int(num_products) == 1 or int(num_products) >= 3:
                    drivers.append(f"Sub-optimal product bundle count ({num_products})")
            except (TypeError, ValueError):
                pass

        is_active = lower_map.get("isactivemember")
        if is_active is not None and str(is_active).lower() in ("0", "no", "false"):
            drivers.append("Low platform engagement / inactive member status")

        if not drivers:
            drivers.append("Composite behavioral risk pattern across usage metrics")

        return drivers

    def generate_plan(
        self,
        customer_id: str,
        churn_prob: float,
        customer_features: Dict[str, Any],
        decision_threshold: float = 0.50,
    ) -> RetentionPlan:
        """Generate a complete retention prescription for a customer."""
        prob = round(float(churn_prob), 4)
        will_churn = prob >= decision_threshold
        risk_tier = self.classify_risk_tier(prob)
        annual_val = self.estimate_annual_value(customer_features)
        revenue_at_risk = round(annual_val * prob, 2)
        max_incentive = round(revenue_at_risk * self.max_offer_budget_ratio, 2)
        drivers = self.identify_risk_drivers(customer_features)

        actions: List[str] = []
        if risk_tier == "Critical":
            channel = "Dedicated Customer Success Manager Call (Within 24h)"
            actions.append(
                f"Authorize executive retention offer up to ${max_incentive:.2f} value (e.g., 20% discount for 12-month contract upgrade)."
            )
            actions.append("Schedule priority account health review & resolve open support tickets.")
        elif risk_tier == "High":
            channel = "Proactive Outbound Phone / SMS + In-App Offer"
            actions.append(
                f"Offer targeted loyalty credit up to ${max_incentive:.2f} or complimentary add-on (Tech Support / Security)."
            )
            actions.append("Incentivize switch to Auto-Pay or Annual Contract plan.")
        elif risk_tier == "Medium":
            channel = "Automated Personalized Email & In-App Nudge"
            actions.append("Send feature adoption walkthrough & highlight unused plan benefits.")
            actions.append("Trigger satisfaction check-in survey (CSAT/NPS).")
        else:
            channel = "Standard Lifecycle Newsletter"
            actions.append("No discount needed — maintain standard engagement & referral program.")

        return RetentionPlan(
            customer_id=str(customer_id),
            churn_probability=prob,
            will_churn=will_churn,
            risk_tier=risk_tier,
            estimated_annual_value=annual_val,
            revenue_at_risk=revenue_at_risk,
            recommended_max_incentive=max_incentive,
            primary_risk_drivers=drivers,
            recommended_actions=actions,
            channel_priority=channel,
        )

