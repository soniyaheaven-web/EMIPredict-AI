import streamlit as st

st.set_page_config(
    page_title="EMIPredict AI",
    page_icon="💸",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(to bottom right, #12071f, #2b0a4d, #000000);
    color: white;
}
.card {
    background: #1e1e2f;
    padding: 30px;streamlit
    border-radius: 15px;
    border: 1px solid #7c3aed;
    margin: 10px;
    height: 180px;
    display: flex;
    flex-direction: column;
    justify-content: center;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<h1 style='text-align:center; color:white; font-size:42px; font-weight:bold;'>
    💸 EMIPredict AI
</h1>
<h3 style='text-align:center; color:#a855f7;'>
    Intelligent Financial Risk Assessment Platform
</h3>
""", unsafe_allow_html=True)

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class='card'>
        <h3 style='color:#a855f7;'>🎯 Classification</h3>
        <p style='color:white;'>EMI Eligibility Prediction<br>
        Eligible / High Risk / Not Eligible</p>
        <p style='color:#22c55e;'>✅ XGBoost — 98% Accuracy</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='card'>
        <h3 style='color:#a855f7;'>📊 Regression</h3>
        <p style='color:white;'>Maximum EMI Amount Prediction<br>
        Continuous Value Prediction</p>
        <p style='color:#22c55e;'>✅ Random Forest — R² 96.43%</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class='card'>
        <h3 style='color:#a855f7;'>📈 EDA Dashboard</h3>
        <p style='color:white;'>Exploratory Data Analysis<br>
        Charts, Graphs & Insights</p>
        <p style='color:#a855f7;'>Visual insights from 400K records</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='card'>
        <h3 style='color:#a855f7;'>🔬 Model Comparison</h3>
        <p style='color:white;'>MLflow Experiment Tracking<br>
        All Models Performance</p>
        <p style='color:#a855f7;'>Classification & Regression results</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("""
<p style='text-align:center; color:#a855f7; font-size:18px;'>
    👈 Select a page from the left sidebar to get started!
</p>
""", unsafe_allow_html=True)