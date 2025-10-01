pip install streamlit joblib pandas scikit-learn
# requirements.txt
streamlit
joblib
pandas
scikit-learn
import streamlit as st
import pandas as pd
import joblib

# Load the trained Extra Trees model
model = joblib.load("extra_trees_churn.pkl")

# Title
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
input_data = pd.DataFrame({
    "Tenure": [tenure],
    "City_Tier": [city_tier],
    "Payment": [payment],
    "Gender": [gender],
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
