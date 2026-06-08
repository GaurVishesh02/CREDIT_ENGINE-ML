import joblib
import numpy as np
import pandas as pd
import streamlit as st

# 1. Clear text format cache to avoid old pkl errors
@st.cache_resource
def load_assets():
    model = joblib.load("loan_model.pkl")
    scaler = joblib.load("scaler.pkl")
    ohe = joblib.load("ohe.pkl")
    return model, scaler, ohe

model, scaler, ohe = load_assets()

st.set_page_config(page_title="Credit Engine Portal", layout="wide")
st.title("🏦 CREDIT_Engine: Automated Loan Approval System")
st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 👤 Applicant Metrics")
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    marital_status = st.selectbox("Marital Status", ["Married", "Single"])
    dependents = st.number_input("Dependents Count", min_value=0, max_value=10, value=1)
    # Check text mapping values
    education_level = st.selectbox("Education Level", ["Graduate", "Undergraduate", "High School"])

with col2:
    st.markdown("### 💰 Financial Data")
    applicant_income = st.number_input("Applicant Income", min_value=0, value=12000)
    coapplicant_income = st.number_input("Coapplicant Income", min_value=0, value=4000)
    savings = st.number_input("Savings Account Balance", min_value=0, value=25000)
    collateral_value = st.number_input("Collateral Value", min_value=0, value=30000)
    dti_ratio = st.slider("DTI Ratio (Debt to Income)", min_value=0.0, max_value=1.0, value=0.35, step=0.01)

with col3:
    st.markdown("### 📊 Risk Parameters")
    credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=680)
    loan_amount = st.number_input("Loan Amount Requested", min_value=0, value=20000)
    loan_term = st.number_input("Loan Term (Months)", min_value=12, max_value=360, value=60)
    existing_loans = st.number_input("Existing Loans Count", min_value=0, max_value=10, value=0)
    loan_purpose = st.selectbox("Loan Purpose", ["Home", "Personal", "Education", "Car", "Business"])
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
    employer_category = st.selectbox("Employer Category", ["Private", "Government", "MNC", "Unemployed"])
    employment_status = st.selectbox("Employment Status", ["Salaried", "Self-employed", "Unemployed"])

st.divider()

if st.button("Run CREDIT_Engine Assessment", type="primary"):
    
    # Map 'Graduate' to 1 directly to prevent LabelEncoder from triggering unseen labels
    edu_map = {"Graduate": 1, "Undergraduate": 0, "High School": 2}
    mapped_edu = edu_map.get(education_level, 0)

    # Reconstruct dictionary dataframe payload
    raw_input_data = pd.DataFrame([{
        'Applicant_Income': float(applicant_income),
        'Coapplicant_Income': float(coapplicant_income),
        'Employment_Status': employment_status,
        'Age': float(age),
        'Marital_Status': marital_status,
        'Dependents': float(dependents),
        'Credit_Score': float(credit_score),
        'Existing_Loans': float(existing_loans),
        'DTI_Ratio': float(dti_ratio),
        'Savings': float(savings),
        'Collateral_Value': float(collateral_value),
        'Loan_Amount': float(loan_amount),
        'Loan_Term': float(loan_term),
        'Loan_Purpose': loan_purpose,
        'Property_Area': property_area,
        'Education_Level': int(mapped_edu), # Directly setting integer, no 'le.transform' used anywhere!
        'Gender': gender,
        'Employer_Category': employer_category
    }])

    try:
        # OneHotEncoder mapping block (Cell 23)
        ohe_cols = ["Employment_Status", "Marital_Status", "Loan_Purpose", "Property_Area", "Gender", "Employer_Category"]
        encoded_arr = ohe.transform(raw_input_data[ohe_cols])
        encoded_features_df = pd.DataFrame(encoded_arr, columns=ohe.get_feature_names_out(ohe_cols), index=raw_input_data.index)
        
        master_df = pd.concat([raw_input_data.drop(columns=ohe_cols), encoded_features_df], axis=1)

        # Feature Engineering (Cell 41)
        master_df["DTI_Ratio_sq"] = master_df["DTI_Ratio"] ** 2
        master_df["Credit_Score_sq"] = master_df["Credit_Score"] ** 2

        # Drop tracks to isolate final model matrix format
        final_x_matrix = master_df.drop(columns=["Credit_Score", "DTI_Ratio"])

        # Scaling and Model Output
        scaled_x_matrix = scaler.transform(final_x_matrix)
        verdict = model.predict(scaled_x_matrix)

        st.subheader("🎯 Engine Execution Verdict")
        if verdict[0] == 1:
            st.success("🎉 **Loan Approved!** Applicant satisfies core financial risk thresholds.")
        else:
            st.error("❌ **Loan Rejected!** Risk matrix exceeded safe parameters.")

    except Exception as error_log:
        st.error(f"Pipeline processing failure: {error_log}")