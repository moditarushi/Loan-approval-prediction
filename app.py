import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and columns
model = joblib.load("loan_model.pkl")
columns = joblib.load("model_columns.pkl")

st.title("💰 Loan Approval Prediction System")

st.write("Enter applicant details below:")

# ---------------- INPUT ----------------
gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
credit_history = st.selectbox("Credit History", [1, 0])
property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

coapp_income = st.number_input("Coapplicant Income", 0.0)
app_income = st.number_input("Applicant Income", 0.0)
loan_amount = st.number_input("Loan Amount", 0.0)
loan_term = st.number_input("Loan Term", 0.0)

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

# ---------------- TRANSFORM ----------------
total_income = app_income + coapp_income

app_income_log = np.log(app_income + 1)
coapp_income_log = np.log(coapp_income + 1)
loan_amount_log = np.log(loan_amount + 1)
loan_term_log = np.log(loan_term + 1)
total_income_log = np.log(total_income + 1)

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
        "Loan_Amount_Term": loan_term_log,
        "CoapplicantIncome": coapp_income_log,
        "ApplicantIncome": app_income_log,
        "LoanAmount": loan_amount_log,
        "Total_Income": total_income_log
    }

    # Convert to DataFrame
    input_df = pd.DataFrame([input_dict])

    # Match columns with training
    input_df = input_df.reindex(columns=columns, fill_value=0)

    # Prediction
    prediction = model.predict(input_df)
    prob = model.predict_proba(input_df)

    approval_prob = prob[0][1] * 100

    if prediction[0] == 1:
        st.success(f"✅ Loan Approved ({approval_prob:.2f}% chance)")
    else:
        st.error(f"❌ Loan Rejected ({approval_prob:.2f}% chance)")
