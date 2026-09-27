import os

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")
INTERNET_OPTS = ["No", "Yes", "No internet service"]

st.set_page_config(page_title="Churn Predictor", page_icon="📉")
st.title("📉 Customer Churn Predictor")
st.caption("Enter customer details to predict how likely they are to leave.")

with st.sidebar:
    st.header("Model info")
    try:
        info = requests.get(f"{API_URL}/model-info", timeout=5).json()
        st.write(f"**Best model:** {info['best_model']}")
        st.dataframe(pd.DataFrame(info["results"]).set_index("model"))
        st.subheader("Top features")
        feats = pd.DataFrame(info["top_features"]).set_index("feature")
        st.bar_chart(feats["importance"])
    except (requests.RequestException, KeyError, ValueError):
        st.error("Backend not running or model not trained.\nRun: python train.py, then uvicorn api:app")

with st.form("customer"):
    c1, c2, c3 = st.columns(3)
    with c1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior = st.selectbox("Senior citizen", ["No", "Yes"])
        partner = st.selectbox("Partner", ["No", "Yes"])
        dependents = st.selectbox("Dependents", ["No", "Yes"])
        tenure = st.number_input("Tenure (months)", 0, 100, 12)
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    with c2:
        phone = st.selectbox("Phone service", ["Yes", "No"])
        lines = st.selectbox("Multiple lines", ["No", "Yes", "No phone service"])
        internet = st.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
        security = st.selectbox("Online security", INTERNET_OPTS)
        backup = st.selectbox("Online backup", INTERNET_OPTS)
        protection = st.selectbox("Device protection", INTERNET_OPTS)
    with c3:
        support = st.selectbox("Tech support", INTERNET_OPTS)
        tv = st.selectbox("Streaming TV", INTERNET_OPTS)
        movies = st.selectbox("Streaming movies", INTERNET_OPTS)
        paperless = st.selectbox("Paperless billing", ["Yes", "No"])
        payment = st.selectbox("Payment method", [
            "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        monthly = st.number_input("Monthly charges", 0.0, 500.0, 70.0)

    total = st.number_input("Total charges", 0.0, 50000.0, float(monthly * tenure))
    submitted = st.form_submit_button("Predict", type="primary")

if submitted:
    customer = {
        "gender": gender, "SeniorCitizen": 1 if senior == "Yes" else 0, "Partner": partner,
        "Dependents": dependents, "tenure": int(tenure), "PhoneService": phone, "MultipleLines": lines,
        "InternetService": internet, "OnlineSecurity": security, "OnlineBackup": backup,
        "DeviceProtection": protection, "TechSupport": support, "StreamingTV": tv,
        "StreamingMovies": movies, "Contract": contract, "PaperlessBilling": paperless,
        "PaymentMethod": payment, "MonthlyCharges": monthly, "TotalCharges": total,
    }
    r = requests.post(f"{API_URL}/predict", json=customer)
    if not r.ok:
        st.error(r.json().get("detail", "Prediction failed"))
    else:
        d = r.json()
        st.metric("Churn probability", f"{d['churn_probability']:.0%}", f"{d['risk']} risk", delta_color="off")
        st.progress(d["churn_probability"])
        if d["will_churn"]:
            st.warning("This customer is likely to churn. Consider a retention offer.")
        else:
            st.success("This customer is likely to stay.")
