 **EMIPredict AI — Intelligent Financial Risk Assessment Platform**

**Project Overview**
EMIPredict AI is a comprehensive financial risk assessment platform that integrates 
machine learning models with MLflow experiment tracking to create an interactive 
web application for EMI prediction. The platform helps financial institutions make 
data-driven loan decisions by predicting EMI eligibility and maximum EMI amount.

**Problem Statement**
Nowadays, people struggle to pay EMI due to poor financial planning and inadequate 
risk assessment. This project solves this critical issue by providing data-driven 
insights for better loan decisions using machine learning models trained on 400,000 
realistic financial records across 5 EMI scenarios.

## 🛠️ Tech Stack
- **Language:** Python 3.12
- **Web Framework:** Streamlit
- **Machine Learning:** Scikit-learn, XGBoost
- **Experiment Tracking:** MLflow
- **Data Processing:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Model Persistence:** Joblib
- **Deployment:** Streamlit Cloud

**Project Workflow**

**Step 1: Data Loading and Preprocessing**
- Loaded 404,800 financial records from CSV dataset
- Handled missing values: numerical columns filled with median, categorical with Unknown
- Converted mixed-type columns using pd.to_numeric with errors coerce
- Removed duplicate records
- Handled outliers using IQR method
- Removed irrelevant column: loan_burden

**Step 2: Feature Engineering**
- Fixed existing_loans column: Yes/No mapped to 1/0
- Applied LabelEncoder to all categorical columns separately
- Saved individual encoders in label_encoders.pkl

**Step 3: Exploratory Data Analysis**
- EMI eligibility distribution: Not_Eligible 77.3%, Eligible 18.4%, High_Risk 4.3%
- Monthly salary distribution and relationship with eligibility
- Credit score patterns across eligibility categories
- Employment type: Private 69.9%, Government 20.1%, Self-employed 10.0%
- Gender distribution: Male 241,107, Female 160,679

**Step 4: Classification Model Development**
| Model | Accuracy | F1 Score |
|---|---|---|
| Logistic Regression | 87.91% | 85.90% |
| Random Forest | 95.50% | 93.98% |
| **XGBoost** | **98.01%** | **97.83%** |

**Best Model: XGBoost Classifier — 98.01% Accuracy**

**Step 5: Regression Model Development**
| Model | RMSE | R² Score |
|---|---|---|
| Linear Regression | 3132.49 | 77.66% |
| **Random Forest** | **1251.87** | **96.43%** |
| XGBoost | 1296.36 | 96.17% |

**Best Model: Random Forest Regressor — R²: 96.43%**

**Step 6: MLflow Integration**
- Configured MLflow experiment tracking for classification and regression
- Logged parameters, hyperparameters, and metrics for all 6 models
- Created artifact storage for models
- Registered best models in MLflow model registry

**Step 7: Streamlit Application**
5-page multi-page web application:
1. **Home** — Project overview
2. **EDA** — Charts and insights from 400K records
3. **EMI Eligibility** — Real-time classification prediction
4. **Max EMI Prediction** — Real-time regression prediction
5. **Model Comparison** — MLflow results and charts

---

## 📈 Model Performance Summary

### Classification:
- **Best Model:** XGBoost Classifier
- **Accuracy:** 98.01%
- **F1 Score:** 97.83%

### Regression:
- **Best Model:** Random Forest Regressor
- **RMSE:** ₹1,251.87
- **R² Score:** 96.43%

---

**How to Run Locally**
```bash
git clone https://github.com/soniyaheaven-web/EMIPredict-AI.git
cd EMIPredict-AI
pip install -r requirements.txt
streamlit run Home.py
```
