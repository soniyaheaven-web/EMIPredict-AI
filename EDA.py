import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


st.markdown("""
<style>
.stApp {
    background: linear-gradient(to bottom right, #12071f, #2b0a4d, #000000);
    color: white;
}
.block-container { padding-top: 2rem; padding-left: 4rem; padding-right: 4rem; }
h1, h2, h3 { color: white !important; }

/* ✅ Metric fix — bright white */
[data-testid="metric-container"] {
    background: #1e1e2f;
    border: 1px solid #7c3aed;
    border-radius: 10px;
    padding: 15px;
}
[data-testid="stMetricLabel"] { color: white !important; font-size: 40px !important; }
[data-testid="stMetricValue"] { color: #a855f7 !important; font-size: 32px !important; font-weight: bold !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<h1 style='text-align:center; color:white; font-size:38px; font-weight:bold;'>
    📊 Exploratory Data Analysis
</h1>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = pd.read_csv("emi_prediction_dataset.csv", low_memory=False)
    return df

df = load_data()

st.markdown("---")

# ── Dataset Overview ──────────────────────────────────
st.subheader("📋 Dataset Overview")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Records", f"{len(df):,}")
with col2:
    st.metric("Total Features", f"{len(df.columns)}")
with col3:
    st.metric("EMI Scenarios", "5")
with col4:
    st.metric("Target Variables", "2")

st.markdown("---")

# ── EMI Eligibility Distribution ─────────────────────
st.subheader("🎯 EMI Eligibility Distribution")
col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots(figsize=(5,3))
    fig.patch.set_facecolor('#1e1e2f')
    ax.set_facecolor('#1e1e2f')
    counts = df['emi_eligibility'].value_counts()
    colors = ['#22c55e', '#f59e0b', '#ef4444']
    ax.pie(counts.values, labels=counts.index,
           colors=colors, autopct='%1.1f%%',
           textprops={'color': 'white', 'fontsize': 12},
           wedgeprops={'edgecolor': 'white'})
    ax.set_title('EMI Eligibility Distribution',
                 color='white', fontsize=12, fontweight='bold')
    st.pyplot(fig)
    plt.close()

with col2:
    fig, ax = plt.subplots(figsize=(5,3.5))
    fig.patch.set_facecolor('#1e1e2f')
    ax.set_facecolor('#1e1e2f')
    bars = ax.bar(counts.index, counts.values,
                  color=colors, edgecolor='white', linewidth=1.2)
    ax.set_title('Eligibility Count', color='white',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Category', color='white', fontsize=12)
    ax.set_ylabel('Count', color='white', fontsize=12)
    ax.tick_params(colors='white', labelsize=10)
    for spine in ax.spines.values():
        spine.set_color('#7c3aed')
    # Value labels on bars
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1000,
                f'{val:,}', ha='center', color='white',
                fontsize=12, fontweight='bold')
    st.pyplot(fig)
    plt.close()

st.markdown("---")

# ── EMI Scenario Distribution ─────────────────────────
st.subheader("📂 EMI Scenario Distribution")
fig, ax = plt.subplots(figsize=(6, 5))
fig.patch.set_facecolor('#1e1e2f')
ax.set_facecolor('#1e1e2f')
scenario_counts = df['emi_scenario'].value_counts()
bars = ax.barh(scenario_counts.index, scenario_counts.values,
               color='#7c3aed', edgecolor='white', linewidth=1.2)
ax.set_title('Records per EMI Scenario', color='white',
             fontsize=14, fontweight='bold')
ax.set_xlabel('Count', color='white', fontsize=14)
ax.tick_params(colors='white', labelsize=10)
for spine in ax.spines.values():
    spine.set_color('#7c3aed')
for bar, val in zip(bars, scenario_counts.values):
    ax.text(val + 500, bar.get_y() + bar.get_height()/2,
            f'{val:,}', va='center', color='white',
            fontsize=10, fontweight='bold')
st.pyplot(fig)
plt.close()

st.markdown("---")

# ── Monthly Salary Distribution ───────────────────────
st.subheader("💰 Monthly Salary Distribution")
col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor('#1e1e2f')
    ax.set_facecolor('#1e1e2f')
    df['monthly_salary'] = pd.to_numeric(df['monthly_salary'], errors='coerce')
    ax.hist(df['monthly_salary'].dropna(), bins=50,
            color='#a855f7', edgecolor='white', alpha=1.0)
    ax.set_title('Monthly Salary Distribution', color='white',
                 fontsize=16, fontweight='bold')
    ax.set_xlabel('Salary (₹)', color='white', fontsize=16)
    ax.set_ylabel('Count', color='white', fontsize=16)
    ax.tick_params(colors='white', labelsize=10)
    for spine in ax.spines.values():
        spine.set_color('#7c3aed')
    st.pyplot(fig)
    plt.close()

with col2:
    fig, ax = plt.subplots(figsize=(6, 5.2))
    fig.patch.set_facecolor('#1e1e2f')
    ax.set_facecolor('#1e1e2f')
    df.boxplot(column='monthly_salary', by='emi_eligibility',
               ax=ax, patch_artist=True)
    ax.set_title('Salary by Eligibility', color='white',
                 fontsize=16, fontweight='bold')
    ax.set_xlabel('EMI Eligibility', color='white', fontsize=16)
    ax.set_ylabel('Salary (₹)', color='white', fontsize=16)
    ax.tick_params(colors='white', labelsize=10)
    fig.patch.set_facecolor('#1e1e2f')
    ax.set_facecolor('#1e1e2f')
    plt.suptitle('')
    for spine in ax.spines.values():
        spine.set_color('#7c3aed')
    st.pyplot(fig)
    plt.close()

st.markdown("---")

# ── Credit Score Analysis ─────────────────────────────
st.subheader("📈 Credit Score Analysis")
col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor('#1e1e2f')
    ax.set_facecolor('#1e1e2f')
    df['credit_score'] = pd.to_numeric(df['credit_score'], errors='coerce')
    ax.hist(df['credit_score'].dropna(), bins=50,
            color='#22c55e', edgecolor='white', alpha=0.9)
    ax.set_title('Credit Score Distribution', color='white',
                 fontsize=16, fontweight='bold')
    ax.set_xlabel('Credit Score', color='white', fontsize=16)
    ax.set_ylabel('Count', color='white', fontsize=16)
    ax.tick_params(colors='white', labelsize=10)
    for spine in ax.spines.values():
        spine.set_color('#7c3aed')
    st.pyplot(fig)
    plt.close()

with col2:
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor('#1e1e2f')
    ax.set_facecolor('#1e1e2f')
    eligible = df[df['emi_eligibility']=='Eligible']['credit_score'].dropna()
    not_eligible = df[df['emi_eligibility']=='Not_Eligible']['credit_score'].dropna()
    high_risk = df[df['emi_eligibility']=='High_Risk']['credit_score'].dropna()
    ax.hist(eligible, bins=30, alpha=0.7, color='#22c55e', label='Eligible')
    ax.hist(high_risk, bins=30, alpha=0.7, color='#f59e0b', label='High Risk')
    ax.hist(not_eligible, bins=30, alpha=0.7, color='#ef4444', label='Not Eligible')
    ax.set_title('Credit Score by Eligibility', color='white',
                 fontsize=16, fontweight='bold')
    ax.set_xlabel('Credit Score', color='white', fontsize=16)
    ax.set_ylabel('Count', color='white', fontsize=16)
    ax.tick_params(colors='white', labelsize=10)
    ax.legend(facecolor='#1e1e2f', labelcolor='white', fontsize=16)
    for spine in ax.spines.values():
        spine.set_color('#7c3aed')
    st.pyplot(fig)
    plt.close()

st.markdown("---")

# ── Employment Type Analysis ──────────────────────────
st.subheader("👔 Employment Type Analysis")
col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor('#1e1e2f')
    ax.set_facecolor('#1e1e2f')
    emp_counts = df['employment_type'].value_counts()
    ax.pie(emp_counts.values, labels=emp_counts.index,
           autopct='%1.1f%%',
           textprops={'color': 'white', 'fontsize': 16},
           wedgeprops={'edgecolor': 'white'})
    ax.set_title('Employment Type Distribution',
                 color='white', fontsize=16, fontweight='bold')
    st.pyplot(fig)
    plt.close()

with col2:
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor('#1e1e2f')
    ax.set_facecolor('#1e1e2f')
    # ✅ Clean gender values — Male/Female only
    gender_clean = df['gender'].str.strip().str.title()
    gender_clean = gender_clean[gender_clean.isin(['Male', 'Female'])]
    gender_counts = gender_clean.value_counts()
    bars = ax.bar(gender_counts.index, gender_counts.values,
                  color=['#a855f7', '#22c55e'], edgecolor='white',
                  linewidth=1.2)
    ax.set_title('Gender Distribution', color='white',
                 fontsize=16, fontweight='bold')
    ax.set_xlabel('Gender', color='white', fontsize=16)
    ax.set_ylabel('Count', color='white', fontsize=16)
    ax.tick_params(colors='white', labelsize=10)
    for spine in ax.spines.values():
        spine.set_color('#7c3aed')
    for bar, val in zip(bars, gender_counts.values):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 1000,
                f'{val:,}', ha='center', color='white',
                fontsize=16, fontweight='bold')
    st.pyplot(fig)
    plt.close()

st.markdown("---")

# ── Key Statistics ────────────────────────────────────
st.subheader("📊 Key Statistics")

def format_stats(series, name):
    s = series.describe().round(2)
    s.index = [i.capitalize() for i in s.index]
    s.name = name
    return s

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div style='background:#1e1e2f; padding:15px; border-radius:10px;
                border:1px solid #7c3aed; margin-bottom:10px;'>
        <h4 style='color:#a855f7; margin:0;'>💰 Salary Stats</h4>
    </div>
    """, unsafe_allow_html=True)
    st.dataframe(format_stats(df['monthly_salary'], 'Monthly Salary'),
                 use_container_width=True)

with col2:
    st.markdown("""
    <div style='background:#1e1e2f; padding:15px; border-radius:10px;
                border:1px solid #7c3aed; margin-bottom:10px;'>
        <h4 style='color:#a855f7; margin:0;'>📈 Credit Score Stats</h4>
    </div>
    """, unsafe_allow_html=True)
    st.dataframe(format_stats(df['credit_score'], 'Credit Score'),
                 use_container_width=True)

with col3:
    st.markdown("""
    <div style='background:#1e1e2f; padding:15px; border-radius:10px;
                border:1px solid #7c3aed; margin-bottom:10px;'>
        <h4 style='color:#a855f7; margin:0;'>💳 Max EMI Stats</h4>
    </div>
    """, unsafe_allow_html=True)
    st.dataframe(format_stats(df['max_monthly_emi'], 'Max Monthly EMI'),
                 use_container_width=True)