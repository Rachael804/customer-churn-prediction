import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

# 1. Load Data
df = pd.read_csv("customer_churn_data.csv")
X = df.drop(columns=["customer_id", "churn"])
y = df["churn"]

# 2. Train / Test Split (Stratified to maintain class balance)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Define Preprocessing Pipeline
numeric_features = ["tenure_months", "monthly_charges", "total_charges"]
categorical_features = [
    "contract", "paperless_billing", "payment_method",
    "internet_service", "tech_support", "online_security"
]

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# 4. Handle Class Imbalance with scale_pos_weight
scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)

# 5. Define Full Pipeline with XGBoost
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", XGBClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42
    ))
])

# 6. Train Model Pipeline
print("Training Customer Churn XGBoost Model...")
pipeline.fit(X_train, y_train)

# 7. Evaluate Performance
y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]

print("\n--- Model Evaluation ---")
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Retained", "Churned"]))

# 8. Save Serialized Model Pipeline
joblib.dump(pipeline, "churn_pipeline.joblib")
print("\nModel pipeline successfully saved to 'churn_pipeline.joblib'")