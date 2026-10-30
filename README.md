# Telecom Customer Churn Prediction Engine

An end-to-end classification system built in Python to predict customer churn in the telecommunications sector. The pipeline automates feature preprocessing, handles target class imbalance, trains an XGBoost classifier, and serves real-time probability scores on incoming customer accounts.

---

## Key Results & Metrics

- **ROC-AUC Score**: **0.83**
- **Churn Recall**: **73%** (catches nearly 3 out of every 4 customers at risk of leaving)
- **Overall Accuracy**: **75%**

---

## Technical Workflow

1. **Synthetic Data Synthesis (`generate_data.py`)**: Simulates 3,000 realistic customer records with features like tenure, contract type, payment channels, monthly charges, and service add-ons.
2. **Production Preprocessing Pipeline (`train_pipeline.py`)**:
   - Numeric imputation via median values and feature scaling using `StandardScaler`.
   - Categorical handling using `SimpleImputer` and `OneHotEncoder`.
   - Packaged inside Scikit-Learn `ColumnTransformer` to eliminate data leakage.
3. **Imbalance Mitigation**: Applied positive class weighting (`scale_pos_weight`) inside XGBoost to ensure the model doesn't overlook churners in favor of the majority class.
4. **Model Serialization**: Saved the fitted pipeline artifact using `joblib` for zero-friction production inference.
5. **Inference Engine (`predict.py`)**: Accepts raw dictionaries/dataframes and generates instant churn risk categories and probabilities.

---

## Project Structure

```text
├── generate_data.py       # Dataset generation script
├── train_pipeline.py      # Pipeline definition, model training, and evaluation
├── predict.py             # Real-time inference script
├── churn_pipeline.joblib  # Serialized trained model
├── requirements.txt       # Frozen environment dependencies
└── README.md              # Project documentation