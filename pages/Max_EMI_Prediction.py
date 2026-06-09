import streamlit as st
import joblib
import pandas as pd

st.markdown("""
<style>
.stApp {
    background: linear-gradient(to bottom right, #12071f, #2b0a4d, #000000);
    color: white;
}
h2, h3, label { color: white !important; font-weight: bold; }
.stNumberInput input, .stSelectbox div[data-baseweb="select"] {
    background-color: #1e1e2f !important;
    color: white !important;
    border-radius: 12px !important;
    border: 1px solid #7c3aed !important;
}
.stButton>button {
    background: linear-gradient(to right, #7c3aed, #a855f7);
    color: white; border: none; border-radius: 15px;
    height: 50px; font-size: 20px; font-weight: bold;
}
.stButton>button:hover {
    background: linear-gradient(to right, #9333ea, #c084fc);
    transform: scale(1.03);
}
.block-container { padding-top: 2rem; padding-left: 4rem; padding-right: 4rem; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<h1 style='text-align:center; color:white; font-size:38px;
           white-space:nowrap; font-weight:bold;'>
    💰 Maximum EMI Prediction
</h1>
""", unsafe_allow_html=True)

# Load model & encoders
reg_model = joblib.load("best_regression_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")

st.subheader("🧑‍💼 Enter Customer Details")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=25)

    gender_input = st.selectbox("Gender", ["Male", "Female"])
    gender = int(label_encoders['gender'].transform([gender_input])[0])

    marital_input = st.selectbox("Marital Status", ["Married", "Single"])
    marital_status = int(label_encoders['marital_status'].transform([marital_input])[0])

    education_input = st.selectbox("Education",
        ["Graduate", "High School", "Post Graduate", "Professional"])
    education = int(label_encoders['education'].transform([education_input])[0])

    monthly_salary = st.number_input("Monthly Income ₹", min_value=0.0, value=50000.0)

    employment_input = st.selectbox("Employment Type",
        ["Government", "Private", "Self-employed"])
    employment_type = int(label_encoders['employment_type'].transform([employment_input])[0])

    years_of_employment = st.number_input("Years of Employment", min_value=0.0, value=3.0)

    company_input = st.selectbox("Company Type",
        ["Large Indian", "MNC", "Mid-size", "Small", "Startup"])
    company_type = int(label_encoders['company_type'].transform([company_input])[0])

    house_input = st.selectbox("House Type", ["Family", "Own", "Rented"])
    house_type = int(label_encoders['house_type'].transform([house_input])[0])

    monthly_rent = st.number_input("Monthly Rent ₹", min_value=0.0)

with col2:
    family_size = st.number_input("Family Size", min_value=1.0, value=3.0)
    dependents = st.number_input("Dependents", min_value=0.0)
    school_fees = st.number_input("School Fees ₹", min_value=0.0)
    college_fees = st.number_input("College Fees ₹", min_value=0.0)
    travel_expenses = st.number_input("Travel Expenses ₹", min_value=0.0)
    groceries_utilities = st.number_input("Groceries & Utilities ₹", min_value=0.0)
    other_monthly_expenses = st.number_input("Other Monthly Expenses ₹", min_value=0.0)

    existing_loans_input = st.selectbox("Existing Loans", ["No", "Yes"])
    existing_loans = 1 if existing_loans_input == "Yes" else 0

    current_emi_amount = st.number_input("Current EMI Amount ₹", min_value=0.0)
    credit_score = st.number_input("Credit Score",
        min_value=300.0, max_value=850.0, value=650.0)

with col3:
    bank_balance = st.number_input("Bank Balance ₹", min_value=0.0)
    emergency_fund = st.number_input("Emergency Fund ₹", min_value=0.0)

    scenario_input = st.selectbox("EMI Scenario", [
        "E-commerce Shopping EMI", "Education EMI",
        "Home Appliances EMI", "Personal Loan EMI", "Vehicle EMI"
    ])
    emi_scenario = int(label_encoders['emi_scenario'].transform([scenario_input])[0])

    requested_amount = st.number_input("Requested Loan Amount ₹", min_value=0.0)
    requested_tenure = st.number_input("Requested Tenure (Months)",
        min_value=1.0, value=24.0)

    # Auto-calculated
    total_expenses = (
        monthly_rent + school_fees + college_fees +
        travel_expenses + groceries_utilities + other_monthly_expenses
    )
    st.info(f"💰 Total Expenses: ₹{total_expenses:,.0f}")

    savings = monthly_salary - total_expenses
    st.info(f"💵 Savings: ₹{savings:,.0f}")

    emi_ratio = current_emi_amount / monthly_salary if monthly_salary > 0 else 0
    st.info(f"📊 EMI Ratio: {emi_ratio:.2f}")

# ── Predict Button ─────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
col_l, col_c, col_r = st.columns([1.5, 2, 1.5])
with col_c:
    predict_btn = st.button("💰 Predict Maximum EMI", use_container_width=True)

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
        'requested_tenure': [requested_tenure]
    })

    prediction = reg_model.predict(input_data)[0]

    st.markdown("---")

    # Result display
    st.markdown(f"""
    <div style='background: linear-gradient(to right, #1e1e2f, #2d1b4e);
                padding: 30px; border-radius: 20px;
                border: 2px solid #7c3aed; text-align: center;'>
        <h2 style='color:#a855f7;'>🎯 Prediction Result</h2>
        <h1 style='color:#22c55e; font-size:52px;'>
            ₹{prediction:,.0f}
        </h1>
        <h3 style='color:white;'>Maximum Safe Monthly EMI Amount</h3>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Financial Summary
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("Monthly Income", f"₹{monthly_salary:,.0f}")
    with col_b:
        st.metric("Total Expenses", f"₹{total_expenses:,.0f}")
    with col_c:
        st.metric("Available Savings", f"₹{savings:,.0f}")

    # Advice
    st.markdown("<br>", unsafe_allow_html=True)
    emi_percent = (prediction / monthly_salary * 100) if monthly_salary > 0 else 0

    if emi_percent <= 30:
        st.success(f"✅ EMI is {emi_percent:.1f}% of income — Financially Comfortable!")
    elif emi_percent <= 50:
        st.warning(f"⚠️ EMI is {emi_percent:.1f}% of income — Manageable but be careful!")
    else:
        st.error(f"❌ EMI is {emi_percent:.1f}% of income — High financial burden!")

    if savings < 0:
        st.warning("⚠️ Note: Expenses exceed income — financially risky!")

        prediction = reg_model.predict(input_data)[0]

    st.markdown("---")

    # ✅ Celebration effects based on EMI amount
    emi_percent = (prediction / monthly_salary * 100) if monthly_salary > 0 else 0

    if emi_percent <= 30:
        st.balloons()        # 🎈 Balloon effect
    elif emi_percent <= 50:
        st.balloons()        # 🎈 Balloon only
