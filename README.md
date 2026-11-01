# Customer Churn Prediction & Retention Analytics

End-to-end churn modeling on the public IBM Telco Customer Churn dataset (7,043 customers, 26.5% churn).

**Pipeline:** cleaning -> feature engineering (`num_services`, `avg_monthly_spend`, `tenure_group`, `autopay`) -> one-hot/scaling -> **SMOTE** (inside CV, no leakage) -> Logistic Regression vs Random Forest vs **XGBoost** -> threshold tuning -> **SHAP** explainability -> Streamlit app.

## Results (20% stratified hold-out, seed 42)
| Model | CV ROC-AUC | Test ROC-AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Logistic Regression | 0.844 | 0.841 | 0.508 | 0.791 | 0.619 |
| Random Forest | 0.836 | 0.836 | 0.583 | 0.631 | 0.606 |
| **XGBoost** | **0.845** | **0.844** | 0.587 | 0.631 | 0.608 |

Threshold tuned to 0.34 for F1 (F1 = 0.631). Top SHAP drivers: month-to-month contract, short tenure, no online security, fiber-optic internet, no tech support, electronic-check payment.

Honest note: the three models are close; gains over logistic regression are small, which is typical for this dataset.

## Run
```bash
pip install -r requirements.txt
python train.py            # trains, saves metrics, plots, models/churn_model.joblib
streamlit run app.py       # interactive risk scorer
```
Outputs: `outputs/model_comparison.csv`, `roc_curves.png`, `shap_summary.png`, `top_churn_drivers.csv`.
