import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class ChurnFeatureEngineer(BaseEstimator, TransformerMixin):
    """Create contemporaneous churn features without using the target."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        z = X.copy()
        monthly = pd.to_numeric(z["monthly_charges"], errors="coerce")
        total = pd.to_numeric(z["total_charges"], errors="coerce")
        tenure = pd.to_numeric(z["tenure_months"], errors="coerce")
        complaints = pd.to_numeric(z["complaints_last_12m"], errors="coerce")
        late = pd.to_numeric(z["late_payments_last_12m"], errors="coerce")
        satisfaction = pd.to_numeric(z["satisfaction_score"], errors="coerce")
        usage = pd.to_numeric(z["monthly_usage_gb"], errors="coerce")
        z["average_charge_per_tenure_month"] = total / tenure.clip(lower=1)
        z["payment_risk_score"] = late + (z["payment_method"].eq("Electronic check")).astype(int)
        z["complaint_intensity"] = complaints / (tenure / 12).clip(lower=1)
        z["service_friction"] = complaints * (late + 1)
        z["low_satisfaction_flag"] = (satisfaction <= 5).astype(float).where(satisfaction.notna())
        z["high_cost_low_usage"] = monthly / (usage + 1)
        z["tenure_monthly_charge_interaction"] = tenure * monthly
        z["contract_month_to_month"] = z["contract_type"].eq("Month-to-month").astype(int)
        return z.replace([np.inf, -np.inf], np.nan)
