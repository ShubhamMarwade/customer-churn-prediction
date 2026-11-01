"""Streamlit churn risk scorer. Run: streamlit run app.py (after python train.py)"""
import joblib, pandas as pd, streamlit as st
art = joblib.load("models/churn_model.joblib"); pipe, thr = art["pipeline"], art["threshold"]
st.title("Customer Churn Risk Scorer")
c1, c2 = st.columns(2)
with c1:
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    monthly = st.slider("Monthly charges", 18.0, 120.0, 70.0)
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    internet = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
    pay = st.selectbox("Payment method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
with c2:
    sec = st.selectbox("Online security", ["No", "Yes", "No internet service"])
    tech = st.selectbox("Tech support", ["No", "Yes", "No internet service"])
    paperless = st.selectbox("Paperless billing", ["Yes", "No"])
    senior = st.selectbox("Senior citizen", [0, 1])
    partner = st.selectbox("Partner", ["No", "Yes"])
row = dict(gender="Male", SeniorCitizen=senior, Partner=partner, Dependents="No", tenure=tenure,
           PhoneService="Yes", MultipleLines="No", InternetService=internet, OnlineSecurity=sec,
           OnlineBackup="No", DeviceProtection="No", TechSupport=tech, StreamingTV="No", StreamingMovies="No",
           Contract=contract, PaperlessBilling=paperless, PaymentMethod=pay, MonthlyCharges=monthly,
           TotalCharges=monthly * max(tenure, 1))
row["num_services"] = sum(row[c] == "Yes" for c in ["OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies"])
row["avg_monthly_spend"] = row["TotalCharges"] / max(tenure, 1)
row["tenure_group"] = "0-12" if tenure <= 12 else "13-24" if tenure <= 24 else "25-48" if tenure <= 48 else "49-72"
row["autopay"] = int("automatic" in pay)
p = pipe.predict_proba(pd.DataFrame([row])[art["columns"]])[0, 1]
st.metric("Churn probability", f"{p:.1%}")
st.error("High churn risk - consider a retention offer") if p >= thr else st.success("Low churn risk")
