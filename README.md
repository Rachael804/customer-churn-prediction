# Enterprise Customer Churn Prediction Engine

An end-to-end machine learning classification system trained on the **IBM Telco Customer Churn** benchmark dataset (7,043 customer accounts) to identify retention risks and forecast subscriber churn.

## 📊 Model Evaluation Benchmark

| Metric | Score | Industry Standard Interpretation |
| :--- | :--- | :--- |
| **ROC-AUC** | **0.8298** | Strong discriminative ability between churned and retained cohorts |
| **Churn Recall** | **0.73** | Successfully captures 73% of at-risk subscribers for retention campaigns |
| **Overall Accuracy** | **0.75** | Balanced performance across imbalanced target classes |
| **Model Type** | **XGBoost** | Gradient-boosted decision trees with class weighting |

## 🛠️ Architecture & Features
- **Data Preprocessing**: `ColumnTransformer` pipeline handling standard scaling on numeric fields (`tenure`, `MonthlyCharges`, `TotalCharges`) and one-hot encoding on categorical customer attributes.
- **Class Balancing**: Balanced target weights (`scale_pos_weight`) to penalize false negatives on the minority churn class.
- **Production Artifact**: Full pipeline serialized with `joblib` into `churn_pipeline.joblib` for inference integration.

## 🚀 Quickstart

```powershell
# Clone repo
git clone [https://github.com/rachael804/Customer-churn.git](https://github.com/rachael804/Customer-churn.git)
cd Customer-churn

# Activate environment and install dependencies
pip install -r requirements.txt

# Execute training pipeline
python src/train.py