import numpy as np
import pandas as pd

np.random.seed(42)
n_samples = 3000

customer_ids = [f"CUST-{1000 + i}" for i in range(n_samples)]
tenure_months = np.random.randint(1, 72, size=n_samples)
contract = np.random.choice(["Month-to-month", "One year", "Two year"], size=n_samples, p=[0.55, 0.25, 0.20])
paperless_billing = np.random.choice(["Yes", "No"], size=n_samples, p=[0.6, 0.4])
payment_method = np.random.choice(
    ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
    size=n_samples,
    p=[0.35, 0.20, 0.25, 0.20]
)
internet_service = np.random.choice(["Fiber optic", "DSL", "No"], size=n_samples, p=[0.45, 0.35, 0.20])
tech_support = np.random.choice(["Yes", "No"], size=n_samples, p=[0.3, 0.7])
online_security = np.random.choice(["Yes", "No"], size=n_samples, p=[0.3, 0.7])

monthly_charges = np.round(
    np.random.uniform(20.0, 115.0, size=n_samples) + (internet_service == "Fiber optic") * 15.0,
    2
)
total_charges = np.round(monthly_charges * tenure_months + np.random.normal(0, 20, size=n_samples), 2)
total_charges = np.maximum(total_charges, monthly_charges)

# Realistic churn logic: short tenure, month-to-month contract, fiber optic, and no tech support increase churn risk
churn_log_odds = (
    -1.5
    - 0.05 * tenure_months
    + 1.2 * (contract == "Month-to-month")
    - 0.8 * (contract == "Two year")
    + 0.5 * (internet_service == "Fiber optic")
    - 0.6 * (tech_support == "Yes")
    + 0.015 * monthly_charges
    + (payment_method == "Electronic check") * 0.4
)
churn_prob = 1 / (1 + np.exp(-churn_log_odds))
churn = (np.random.rand(n_samples) < churn_prob).astype(int)

df = pd.DataFrame({
    "customer_id": customer_ids,
    "tenure_months": tenure_months,
    "contract": contract,
    "paperless_billing": paperless_billing,
    "payment_method": payment_method,
    "internet_service": internet_service,
    "tech_support": tech_support,
    "online_security": online_security,
    "monthly_charges": monthly_charges,
    "total_charges": total_charges,
    "churn": churn
})

# Add a few realistic missing values
missing_idx = np.random.choice(n_samples, size=15, replace=False)
df.loc[missing_idx, "total_charges"] = np.nan

df.to_csv("customer_churn_data.csv", index=False)
print(f"Dataset successfully created: customer_churn_data.csv ({len(df)} rows)")
print(f"Churn rate: {df['churn'].mean():.2%}")