import joblib
import pandas as pd

# Load the saved model pipeline
pipeline = joblib.load("churn_pipeline.joblib")

# Sample profiles representing two realistic customer cases
incoming_customers = pd.DataFrame([
    {
        "customer_id": "CUST-9001",
        "tenure_months": 3,
        "contract": "Month-to-month",
        "paperless_billing": "Yes",
        "payment_method": "Electronic check",
        "internet_service": "Fiber optic",
        "tech_support": "No",
        "online_security": "No",
        "monthly_charges": 95.50,
        "total_charges": 286.50
    },
    {
        "customer_id": "CUST-9002",
        "tenure_months": 48,
        "contract": "Two year",
        "paperless_billing": "No",
        "payment_method": "Credit card",
        "internet_service": "DSL",
        "tech_support": "Yes",
        "online_security": "Yes",
        "monthly_charges": 55.00,
        "total_charges": 2640.00
    }
])

# Generate predictions
features = incoming_customers.drop(columns=["customer_id"])
predictions = pipeline.predict(features)
probabilities = pipeline.predict_proba(features)[:, 1]

print("\n--- Real-Time Churn Inference ---")
for idx, row in incoming_customers.iterrows():
    status = "CHURN RISK" if predictions[idx] == 1 else "RETAINED"
    print(f"Customer: {row['customer_id']} | Contract: {row['contract']:<14} | Risk Score: {probabilities[idx]:.2%} -> Status: {status}")