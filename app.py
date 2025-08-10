# app.py
import streamlit as st
import pandas as pd
from tensorflow.keras.models import load_model

# ------------------------------
# Load model
# ------------------------------
model = load_model("tfmodel.keras")

# ------------------------------
# Get expected columns from training CSV
# ------------------------------
df_train = pd.read_csv("Churn.csv")
X_train = pd.get_dummies(df_train.drop(['Churn', 'Customer ID'], axis=1))
expected_columns = X_train.columns

st.set_page_config(page_title="Churn Prediction", layout="centered")
st.title("📊 Customer Churn Prediction")

st.write("Fill out the customer details to predict churn.")

# ------------------------------
# Collect user input (NO Customer ID)
# ------------------------------
gender = st.selectbox("Gender", ["Male", "Female"])
senior_citizen = st.selectbox("Senior Citizen", [0, 1])
partner = st.selectbox("Partner", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["Yes", "No"])
tenure = st.number_input("Tenure (months)", min_value=0)
phone_service = st.selectbox("Phone Service", ["Yes", "No"])
multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
payment_method = st.selectbox(
    "Payment Method",
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
)
monthly_charges = st.number_input("Monthly Charges", min_value=0.0)
total_charges = st.number_input("Total Charges", min_value=0.0)

# ------------------------------
# Create dataframe from input
# ------------------------------
user_data = pd.DataFrame({
    "gender": [gender],
    "Senior Citizen": [senior_citizen],
    "Partner": [partner],
    "Dependents": [dependents],
    "tenure": [tenure],
    "Phone Service": [phone_service],
    "Multiple Lines": [multiple_lines],
    "Internet Service": [internet_service],
    "Online Security": [online_security],
    "Online Backup": [online_backup],
    "Device Protection": [device_protection],
    "Tech Support": [tech_support],
    "Streaming TV": [streaming_tv],
    "Streaming Movies": [streaming_movies],
    "Contract": [contract],
    "Paperless Billing": [paperless_billing],
    "Payment Method": [payment_method],
    "Monthly Charges": [monthly_charges],
    "Total Charges": [total_charges]
})

# ------------------------------
# Preprocess user input
# ------------------------------
user_encoded = pd.get_dummies(user_data)
user_encoded = user_encoded.reindex(columns=expected_columns, fill_value=0)

# ------------------------------
# Prediction
# ------------------------------
if st.button("Predict Churn"):
    prediction = model.predict(user_encoded)
    prediction_class = (prediction[0][0] >= 0.5)

    if prediction_class:
        st.error(f"❌ This customer is likely to CHURN ({prediction[0][0]*100:.2f}%)")
    else:
        st.success(f"✅ This customer is likely to STAY ({(1-prediction[0][0])*100:.2f}%)")
