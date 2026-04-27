import streamlit as st
import pickle
import numpy as np
import pandas as pd

# MUST be first Streamlit command
st.set_page_config(page_title="Loan Prediction", page_icon="💰")

# Load model
model = pickle.load(open("loan_model.pkl", "rb"))
columns = pickle.load(open("model_columns.pkl", "rb"))

# Sidebar
st.sidebar.title("About")
st.sidebar.info("This app predicts loan approval using Machine Learning.")

# Title
st.title("💰 Loan Approval Prediction System")
st.write("Enter applicant details below:")

# -------------------- INPUT --------------------

st.subheader("🧍 Personal Details")

gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])

st.subheader("💰 Financial Details")

credit_history = st.selectbox("Credit History", [1.0, 0.0])
property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

loan_term = st.number_input("Loan Amount Term (months)", value=360)
coapp_income = st.number_input("Coapplicant Income", value=0.0)
app_income = st.number_input("Applicant Income", value=5000.0)
loan_amount = st.number_input("Loan Amount (in thousands)", value=120.0)

# -------------------- ENCODING --------------------

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

# -------------------- TRANSFORM --------------------

total_income = app_income + coapp_income

app_income_log = np.log(app_income + 1)
loan_amount_log = np.log(loan_amount + 1)
loan_term_log = np.log(loan_term + 1)
total_income_log = np.log(total_income + 1)

# -------------------- PREDICTION --------------------

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

    input_df = pd.DataFrame([input_dict])

    input_df = input_df.reindex(columns=columns,fill_value=0)

    prediction = model.predict(input_df)
    prob = model.predict_proba(input_df)

    approval_prob = prob[0][1] * 100

    st.markdown("---")

    # -------------------- OUTPUT --------------------

    if prediction[0] == 1:
        st.success("✅ Loan Approved")

        if approval_prob > 80:
            st.write("💚 Strong approval chances")
        elif approval_prob > 60:
            st.write("💛 Moderate approval chances")
        else:
            st.warning("⚠️ Risky approval")

    else:
        st.error("❌ Loan Rejected")

        if approval_prob < 40:
            st.write("🚫 Very low approval chances")

    # Progress bar
    st.progress(int(approval_prob))

    # Probability display
    st.info(f"📊 Approval Probability: {approval_prob:.2f}%")

    # Insights
    st.markdown("### 📌 Model Insights")
    st.markdown("""
    - Higher income increases approval chances  
    - Good credit history is most important  
    - Lower loan amount improves approval  
    """)

    
