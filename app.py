import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ---------------- LOAD MODEL ----------------
with open("loan_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model_columns.pkl", "rb") as f:
    columns = pickle.load(f)

st.title("💰 Loan Approval Prediction System")

# ---------------- INPUTS ----------------
gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
credit_history = st.selectbox("Credit History", [1, 0])
property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

coapp_income = st.number_input("Coapplicant Income", value=0.0)
app_income = st.number_input("Applicant Income", value=5000.0)
loan_amount = st.number_input("Loan Amount (in thousands)", value=120.0)
loan_term = st.number_input("Loan Term", value=360.0)

# ---------------- ENCODING ----------------
gender = 1 if gender == "Male" else 0
married = 1 if married == "Yes" else 0
dependents = 3 if dependents == "3+" else int(dependents)
education = 1 if education == "Graduate" else 0
self_employed = 1 if self_employed == "Yes" else 0

if property_area == "Urban":
    property_area = 2
elif property_area == "Semiurban":
    property_area = 1
else:
    property_area = 0

# ---------------- PREDICTION ----------------
if st.button("Predict Loan Status"):

    input_dict = {
        "Gender": gender,
        "Married": married,
        "Dependents": dependents,
        "Education": education,
        "Self_Employed": self_employed,
        "Credit_History": credit_history,
        "Property_Area": property_area,
        "Loan_Amount_Term": np.log(loan_term + 1),
        "CoapplicantIncome": np.log(coapp_income + 1),
        "ApplicantIncome": np.log(app_income + 1),
        "LoanAmount": np.log(loan_amount + 1),
    }

    # Convert to DataFrame
    input_df = pd.DataFrame([input_dict])

    # Match model columns
    input_df = input_df.reindex(columns=columns, fill_value=0)

    # Ensure numeric
    input_df = input_df.astype(float)

    # Prediction
    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1] * 100

    # Output
    if prediction == 1:
        st.success(f"✅ Loan Approved ({prob:.2f}% chance)")
    else:
        st.error(f"❌ Loan Rejected ({prob:.2f}% chance)")
