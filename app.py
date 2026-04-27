import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load model & columns
model = pickle.load(open("loan_model.pkl", "rb"))
columns = pickle.load(open("model_columns.pkl", "rb"))

st.title("💰 Loan Approval Prediction System")

# ---------------- INPUT ----------------

gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])

credit_history = st.selectbox("Credit History", [1.0, 0.0])
property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

loan_term = st.number_input("Loan Term", value=360)
coapp_income = st.number_input("Coapplicant Income", value=0.0)
app_income = st.number_input("Applicant Income", value=5000.0)
loan_amount = st.number_input("Loan Amount", value=120.0)

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

        # Try both possibilities (VERY IMPORTANT)
        "Loan_Term_Log": loan_term_log,
        "LoanAmount": loan_amount_log,
        "LoanAmount_Log": loan_amount_log,

        "ApplicantIncome": app_income_log,
        "ApplicantIncome_Log": app_income_log,

        "CoapplicantIncome": coapp_income_log,
        "CoapplicantIncome_Log": coapp_income_log,
    }

    # Convert to dataframe
    input_df = pd.DataFrame([input_dict])

    # 🔥 Force match columns EXACTLY
    for col in columns:
        if col not in input_df:
            input_df[col] = 0

    input_df = input_df[columns]

    # Convert to numeric
    input_df = input_df.astype(float)

    # DEBUG (optional)
    # st.write(input_df)

    # Prediction
    prediction = model.predict(input_df)
    prob = model.predict_proba(input_df)

    approval_prob = prob[0][1] * 100

    if prediction[0] == 1:
        st.success(f"✅ Loan Approved ({approval_prob:.2f}%)")
    else:
        st.error(f"❌ Loan Rejected ({approval_prob:.2f}%)")
