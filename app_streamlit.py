# app_streamlit.py

import streamlit as st
import pandas as pd
import joblib

# ---- Load model and encoders ----
model = joblib.load("extra_trees_churn.pkl")
payment_encoder = joblib.load("payment_encoder.pkl")
gender_encoder = joblib.load("gender_encoder.pkl")

# ---- Title ----
st.title("📊 Telecom Customer Churn Prediction (Extra Trees Classifier)")
st.write("Enter customer details to check if they are likely to churn:")

# ---- User Inputs ----
tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
city_tier = st.selectbox("City Tier", [1, 2, 3])
payment = st.selectbox("Payment Method", ["Credit Card", "Debit Card", "UPI", "Net Banking", "Wallet"])
gender = st.radio("Gender", ["Male", "Female"])
service_score = st.slider("Service Score", 0, 5, 3)
rev_per_month = st.number_input("Revenue per Month", min_value=0, max_value=10000, value=500)

# ---- Prepare Input ----
# Encode categorical features
payment_encoded = payment_encoder.transform([payment])[0]
gender_encoded = gender_encoder.transform([gender])[0]

input_data = pd.DataFrame({
    "Tenure": [tenure],
    "City_Tier": [city_tier],
    "Payment": [payment_encoded],
    "Gender": [gender_encoded],
    "Service_Score": [service_score],
    "rev_per_month": [rev_per_month]
})

# ---- Prediction ----
if st.button("Predict Churn"):
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.error("⚠️ Customer is likely to Churn")
    else:
        st.success("✅ Customer will Stay")
