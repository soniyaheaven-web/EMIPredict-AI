import streamlit as st
import joblib
import pandas as pd

# ── Styling ───────────────────────────────────────────
st.markdown("""
<style>
.stApp { background: linear-gradient(to bottom right, #12071f, #2b0a4d, #000000); color: white; }
h2, h3, label { color: white !important; font-weight: bold; }
.stNumberInput input, .stSelectbox div[data-baseweb="select"] {
    background-color: #1e1e2f !important; color: white !important;
    border-radius: 12px !important; border: 1px solid #7c3aed !important;
}
.stButton>button {
    background: linear-gradient(to right, #7c3aed, #a855f7);
    color: white; border: none; border-radius: 15px;
    height: 50px; width: 280px; font-size: 20px; font-weight: bold;
}
.stButton>button:hover { background: linear-gradient(to right, #9333ea, #c084fc); transform: scale(1.03); }
.block-container { padding-top: 2rem; padding-left: 6rem; padding-right: 6rem; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<h1 style='text-align:center; color:white; font-size:38px;
           white-space:nowrap; font-weight:bold;'>
    💸 EMI Eligibility Prediction
</h1>
""", unsafe_allow_html=True)

# ── Load model & encoders ─────────────────────────────
model = joblib.load("emi_prediction_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")

st.subheader("🧑‍💼 Enter Customer Details")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=25)

    gender_input = st.selectbox("Gender", ["Male", "Female"])
    # Training-ல 'Male'=5, 'Female'=2
    gender = label_encoders['gender'].transform([gender_input])[0]

    marital_input = st.selectbox("Marital Status", ["Married", "Single"])
    # Training-ல 'Married'=0, 'Single'=1
    marital_status = label_encoders['marital_status'].transform([marital_input])[0]

    education_input = st.selectbox("Education",
        ["Graduate", "High School", "Post Graduate", "Professional"])
    education = label_encoders['education'].transform([education_input])[0]

    monthly_salary = st.number_input("Monthly Income ₹", min_value=0.0, value=50000.0)

    employment_input = st.selectbox("Employment Type",
        ["Government", "Private", "Self-employed"])
    employment_type = label_encoders['employment_type'].transform([employment_input])[0]

    years_of_employment = st.number_input("Years of Employment", min_value=0.0, value=3.0)

    company_input = st.selectbox("Company Type",
        ["Large Indian", "MNC", "Mid-size", "Small", "Startup"])
    company_type = label_encoders['company_type'].transform([company_input])[0]

    house_input = st.selectbox("House Type", ["Family", "Own", "Rented"])
    house_type = label_encoders['house_type'].transform([house_input])[0]

    monthly_rent = st.number_input("Monthly Rent ₹", min_value=0.0)

with col2:
    family_size = st.number_input("Family Size", min_value=1.0, value=3.0)
    dependents = st.number_input("Dependents", min_value=0.0)
    school_fees = st.number_input("School Fees ₹", min_value=0.0)
    college_fees = st.number_input("College Fees ₹", min_value=0.0)
    travel_expenses = st.number_input("Travel Expenses ₹", min_value=0.0)
    groceries_utilities = st.number_input("Groceries & Utilities ₹", min_value=0.0)
    other_monthly_expenses = st.number_input("Other Monthly Expenses ₹", min_value=0.0)
    existing_loans = st.selectbox("Existing Loans", ["No", "Yes"])
    existing_loans = 1 if existing_loans == "Yes" else 0
    current_emi_amount = st.number_input("Current EMI Amount ₹", min_value=0.0)
    credit_score = st.number_input("Credit Score", min_value=300.0, max_value=850.0, value=650.0)

with col3:
    bank_balance = st.number_input("Bank Balance ₹", min_value=0.0)
    emergency_fund = st.number_input("Emergency Fund ₹", min_value=0.0)

    scenario_input = st.selectbox("EMI Scenario", [
        "E-commerce Shopping EMI", "Education EMI",
        "Home Appliances EMI", "Personal Loan EMI", "Vehicle EMI"
    ])
    emi_scenario = label_encoders['emi_scenario'].transform([scenario_input])[0]

    requested_amount = st.number_input("Requested Loan Amount ₹", min_value=0.0)
    requested_tenure = st.number_input("Requested Tenure (Months)", min_value=1.0, value=24.0)
    max_monthly_emi = st.number_input("Max Monthly EMI ₹", min_value=0.0)

    # Auto-calculated
    total_expenses = (monthly_rent + school_fees + college_fees +
                      travel_expenses + groceries_utilities + other_monthly_expenses)
    st.info(f"💰 Total Expenses: ₹{total_expenses:,.0f}")

    savings = monthly_salary - total_expenses
    st.info(f"💵 Savings: ₹{savings:,.0f}")

    emi_ratio = current_emi_amount / monthly_salary if monthly_salary > 0 else 0
    st.info(f"📊 EMI Ratio: {emi_ratio:.2f}")

# ── Predict Button ────────────────────────────────────
# ✅ New — perfectly centered
st.markdown("<br>", unsafe_allow_html=True)
col_left, col_center, col_right = st.columns([1.5, 2, 1.5])
with col_center:
    predict_btn = st.button("✨ Predict EMI Eligibility", use_container_width=True)

if predict_btn:
    input_data = pd.DataFrame({
        'age': [age],
        'gender': [gender],
        'marital_status': [marital_status],
        'education': [education],
        'monthly_salary': [monthly_salary],
        'employment_type': [employment_type],
        'years_of_employment': [years_of_employment],
        'company_type': [company_type],
        'house_type': [house_type],
        'monthly_rent': [monthly_rent],
        'family_size': [family_size],
        'dependents': [dependents],
        'school_fees': [school_fees],
        'college_fees': [college_fees],
        'travel_expenses': [travel_expenses],
        'groceries_utilities': [groceries_utilities],
        'other_monthly_expenses': [other_monthly_expenses],
        'existing_loans': [existing_loans],
        'current_emi_amount': [current_emi_amount],
        'credit_score': [credit_score],
        'bank_balance': [bank_balance],
        'emergency_fund': [emergency_fund],
        'emi_scenario': [emi_scenario],
        'requested_amount': [requested_amount],
        'requested_tenure': [requested_tenure],
        'max_monthly_emi': [max_monthly_emi]
    })

    prediction = model.predict(input_data)[0]

    # ✅ Correct mapping: Eligible=0, High_Risk=1, Not_Eligible=2
    if prediction == 0:
        st.success("✅ Prediction: Eligible")
        st.success("🎉 Customer is Eligible for EMI!")
        st.balloons()
    elif prediction == 1:
        st.warning("⚠️ Prediction: High Risk")
        st.warning("Customer may get EMI with higher interest rate")
    else:
        st.error("❌ Prediction: Not Eligible")
        st.error("Customer is Not Eligible for EMI")
   
    if savings < 0:
        st.warning("⚠️ Note: Expenses exceed income — financially risky!")
    
