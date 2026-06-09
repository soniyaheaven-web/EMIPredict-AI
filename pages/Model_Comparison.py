import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

st.markdown("""
<style>
.stApp {
    background: linear-gradient(to bottom right, #12071f, #2b0a4d, #000000);
    color: white;
}
.block-container { padding-top: 2rem; padding-left: 4rem; padding-right: 4rem; }
h1, h2, h3 { color: white !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<h1 style='text-align:center; color:white; font-size:38px; font-weight:bold;'>
    🔬 Model Comparison
</h1>
""", unsafe_allow_html=True)

st.markdown("---")

# ── Classification Models ─────────────────────────────
st.subheader("🎯 Classification Models Comparison")

clf_data = {
    'Model': ['Logistic Regression', 'Random Forest', 'XGBoost'],
    'Accuracy': [0.8791, 0.9550, 0.9801],
    'Precision': [0.8439, 0.9504, 0.9793],
    'Recall': [0.8791, 0.9550, 0.9801],
    'F1 Score': [0.8590, 0.9398, 0.9783]
}
clf_df = pd.DataFrame(clf_data)

# ── Classification Table ──────────────────────────────
st.markdown("""
<div style='background:#1e1e2f; padding:15px; border-radius:10px;
            border:1px solid #7c3aed; margin-bottom:15px;'>
    <h4 style='color:#a855f7; margin:0;'>📊 Performance Metrics Table</h4>
</div>
""", unsafe_allow_html=True)

# Highlight best model
def highlight_best(s):
    is_max = s == s.max()
    return ['background-color: #14532d; color: white' if v
            else 'color: white' for v in is_max]

styled_clf = clf_df.set_index('Model').style\
    .apply(highlight_best)\
    .format("{:.4f}")
st.dataframe(styled_clf, use_container_width=True)

st.markdown("""
<p style='color:#22c55e; font-size:16px; font-weight:bold;'>
    🏆 Best Classification Model: XGBoost — 98.01% Accuracy
</p>
""", unsafe_allow_html=True)

st.markdown("---")

# ── Classification Bar Chart ──────────────────────────
col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots(figsize=(8,6))
    fig.patch.set_facecolor('#1e1e2f')
    ax.set_facecolor('#1e1e2f')
    x = np.arange(len(clf_df['Model']))
    width = 0.2
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    colors = ['#7c3aed', '#22c55e', '#f59e0b', '#ef4444']
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        bars = ax.bar(x + i*width, clf_df[metric],
                      width, label=metric, color=color,
                      edgecolor='white', linewidth=0.8)
    ax.set_title('Classification Models — All Metrics',
                 color='white', fontsize=24, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(['LR', 'RF', 'XGB'],
                        color='white', fontsize=20)
    ax.set_ylabel('Score', color='white', fontsize=22)
    ax.set_ylim(0.7, 1.05)
    ax.tick_params(colors='white', labelsize=20)
    ax.legend(facecolor='#1e1e2f', labelcolor='white', fontsize=16)
    for spine in ax.spines.values():
        spine.set_color('#7c3aed')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with col2:
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor('#1e1e2f')
    ax.set_facecolor('#1e1e2f')
    model_colors = ['#f59e0b', '#22c55e', '#7c3aed']
    bars = ax.bar(clf_df['Model'], clf_df['Accuracy'],
                  color=model_colors, edgecolor='white', linewidth=1.2)
    ax.set_title('Accuracy Comparison',
                 color='white', fontsize=24, fontweight='bold')
    ax.set_ylabel('Accuracy', color='white', fontsize=22)
    ax.set_ylim(0.7, 1.05)
    ax.tick_params(colors='white', labelsize=16)
    for spine in ax.spines.values():
        spine.set_color('#7c3aed')
    for bar, val in zip(bars, clf_df['Accuracy']):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.005,
                f'{val:.2%}', ha='center',
                color='white', fontsize=16, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

st.markdown("---")

# ── Regression Models ─────────────────────────────────
st.subheader("📊 Regression Models Comparison")

reg_data = {
    'Model': ['Linear Regression', 'Random Forest', 'XGBoost'],
    'RMSE': [3132.49, 1251.87, 1296.36],
    'MAE': [2313.78, 727.84, 878.70],
    'R² Score': [0.7766, 0.9643, 0.9617],
    'MAPE (%)': [162.84, 23.25, 47.81]
}
reg_df = pd.DataFrame(reg_data)

st.markdown("""
<div style='background:#1e1e2f; padding:15px; border-radius:10px;
            border:1px solid #7c3aed; margin-bottom:15px;'>
    <h4 style='color:#a855f7; margin:0;'>📊 Performance Metrics Table</h4>
</div>
""", unsafe_allow_html=True)

def highlight_reg(s):
    if s.name in ['RMSE', 'MAE', 'MAPE (%)']:
        is_best = s == s.min()  # Lower is better
    else:
        is_best = s == s.max()  # Higher is better
    return ['background-color: #14532d; color: white' if v
            else 'color: white' for v in is_best]

styled_reg = reg_df.set_index('Model').style\
    .apply(highlight_reg)\
    .format("{:.4f}")
st.dataframe(styled_reg, use_container_width=True)

st.markdown("""
<p style='color:#22c55e; font-size:16px; font-weight:bold;'>
    🏆 Best Regression Model: Random Forest — RMSE: 1251.87 | R²: 96.43%
</p>
""", unsafe_allow_html=True)

st.markdown("---")

# ── Regression Charts ─────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor('#1e1e2f')
    ax.set_facecolor('#1e1e2f')
    model_colors = ['#f59e0b', '#22c55e', '#7c3aed']
    bars = ax.bar(reg_df['Model'], reg_df['RMSE'],
                  color=model_colors, edgecolor='white', linewidth=1.2)
    ax.set_title('RMSE Comparison (Lower is Better)',
                 color='white', fontsize=23, fontweight='bold')
    ax.set_ylabel('RMSE (₹)', color='white', fontsize=22)
    ax.tick_params(colors='white', labelsize=18)
    for spine in ax.spines.values():
        spine.set_color('#7c3aed')
    for bar, val in zip(bars, reg_df['RMSE']):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 30,
                f'₹{val:,.0f}', ha='center',
                color='white', fontsize=24, fontweight='bold')
    ax.set_xticklabels(['LR', 'RF', 'XGB'], color='white', fontsize=24)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with col2:
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor('#1e1e2f')
    ax.set_facecolor('#1e1e2f')
    bars = ax.bar(reg_df['Model'], reg_df['R² Score'],
                  color=model_colors, edgecolor='white', linewidth=1.2)
    ax.set_title('R² Score Comparison (Higher is Better)',
                 color='white', fontsize=23, fontweight='bold')
    ax.set_ylabel('R² Score', color='white', fontsize=22)
    ax.set_ylim(0.5, 1.05)
    ax.tick_params(colors='white', labelsize=20)
    for spine in ax.spines.values():
        spine.set_color('#7c3aed')
    for bar, val in zip(bars, reg_df['R² Score']):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.005,
                f'{val:.2%}', ha='center',
                color='white', fontsize=24, fontweight='bold')
    ax.set_xticklabels(['LR', 'RF', 'XGB'], color='white', fontsize=24)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

st.markdown("---")

# ── Final Summary ─────────────────────────────────────
st.subheader("🏆 Final Model Selection Summary")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div style='background: linear-gradient(to right, #14532d, #166534);
                padding:33.10px; border-radius:15px;
                border:2px solid #22c55e; text-align:center;'>
        <h3 style='color:#22c55e;'>🎯 Classification</h3>
        <h2 style='color:white;'>XGBoost</h2>
        <p style='color:white; font-size:18px;'>Accuracy: <b>98.01%</b></p>
        <p style='color:white; font-size:18px;'>F1 Score: <b>97.83%</b></p>
        <p style='color:#22c55e;'>✅ Selected for Production</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style='background: linear-gradient(to right, #1e3a5f, #1e40af);
                padding:27px; border-radius:15px;
                border:2px solid #3b82f6; text-align:center;'>
        <h3 style='color:#60a5fa;'>📊 Regression</h3>
        <h2 style='color:white;'>Random Forest</h2>
        <p style='color:white; font-size:18px;'>RMSE: <b>₹1,251.87</b></p>
        <p style='color:white; font-size:18px;'>R² Score: <b>96.43%</b></p>
        <p style='color:#60a5fa;'>✅ Selected for Production</p>
    </div>
    """, unsafe_allow_html=True)
