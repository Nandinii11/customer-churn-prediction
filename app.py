import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Load trained model
# -----------------------------
model = joblib.load("churn_model.pkl")

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="🔮",
    layout="wide"
)

# -----------------------------
# Title
# -----------------------------
st.title("Customer Churn Prediction")
st.write(
    "Enter customer details below to predict whether the customer is likely to churn."
)

st.divider()

# -----------------------------
# Customer Information
# -----------------------------
st.subheader("Customer Information")

col1, col2, col3 = st.columns(3)

with col1:
    account_length = st.number_input(
        "Account Length",
        min_value=0,
        value=100
    )

    area_code = st.number_input(
        "Area Code",
        min_value=0,
        value=415
    )

    international_plan = st.selectbox(
        "International Plan",
        ["No", "Yes"]
    )

    voice_mail_plan = st.selectbox(
        "Voice Mail Plan",
        ["No", "Yes"]
    )

    number_vmail_messages = st.number_input(
        "Number of Voicemail Messages",
        min_value=0,
        value=10
    )

with col2:
    total_day_minutes = st.number_input(
        "Total Day Minutes",
        min_value=0.0,
        value=180.0
    )

    total_day_calls = st.number_input(
        "Total Day Calls",
        min_value=0,
        value=100
    )

    total_day_charge = st.number_input(
        "Total Day Charge",
        min_value=0.0,
        value=30.0
    )

    total_eve_minutes = st.number_input(
        "Total Evening Minutes",
        min_value=0.0,
        value=200.0
    )

    total_eve_calls = st.number_input(
        "Total Evening Calls",
        min_value=0,
        value=100
    )

    total_eve_charge = st.number_input(
        "Total Evening Charge",
        min_value=0.0,
        value=15.0
    )

with col3:
    total_night_minutes = st.number_input(
        "Total Night Minutes",
        min_value=0.0,
        value=200.0
    )

    total_night_calls = st.number_input(
        "Total Night Calls",
        min_value=0,
        value=100
    )

    total_night_charge = st.number_input(
        "Total Night Charge",
        min_value=0.0,
        value=10.0
    )

    total_intl_minutes = st.number_input(
        "Total International Minutes",
        min_value=0.0,
        value=10.0
    )

    total_intl_calls = st.number_input(
        "Total International Calls",
        min_value=0,
        value=4
    )

    total_intl_charge = st.number_input(
        "Total International Charge",
        min_value=0.0,
        value=3.0
    )

    customer_service_calls = st.number_input(
        "Customer Service Calls",
        min_value=0,
        value=1
    )

# -----------------------------
# Prediction
# -----------------------------
st.divider()

if st.button(" Predict Churn", use_container_width=True):

    # Create input dataframe
    input_data = pd.DataFrame({
        "Account length": [account_length],
        "Area code": [area_code],
        "International plan": [international_plan],
        "Voice mail plan": [voice_mail_plan],
        "Number vmail messages": [number_vmail_messages],
        "Total day minutes": [total_day_minutes],
        "Total day calls": [total_day_calls],
        "Total day charge": [total_day_charge],
        "Total eve minutes": [total_eve_minutes],
        "Total eve calls": [total_eve_calls],
        "Total eve charge": [total_eve_charge],
        "Total night minutes": [total_night_minutes],
        "Total night calls": [total_night_calls],
        "Total night charge": [total_night_charge],
        "Total intl minutes": [total_intl_minutes],
        "Total intl calls": [total_intl_calls],
        "Total intl charge": [total_intl_charge],
        "Customer service calls": [customer_service_calls]
    })

    # Prediction
    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0][1]

    # -----------------------------
    # Display result
    # -----------------------------
    if prediction == 1:
        st.error(" Customer is likely to churn.")
    else:
        st.success(" Customer is likely to stay.")

    st.metric(
        "Churn Probability",
        f"{probability * 100:.2f}%"
    )