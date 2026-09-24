# 🫁 PulmoVision AI – Lung Cancer Risk Prediction System

**PulmoVision AI** is a state-of-the-art machine learning web application and clinical analytics platform designed for pulmonary cancer risk stratification. Built with Python and Streamlit, it pairs an ensemble Random Forest classifier with MinMaxScaler normalization to predict multi-class cancer risk (*Low*, *Medium*, *High*) across 23 patient demographic, environmental, genetic, and symptom attributes.

---

## 🌟 Key Features

1. **🏠 Executive Dashboard**
   - Live KPI metrics computed dynamically from the patient cohort (Total Patients, Average Age, Risk distribution percentages).
   - Interactive Plotly Donut Chart of risk tiers.
   - Cohort Age distribution, Smoking exposure breakdown, and Air pollution risk analysis.

2. **🩺 Clinical Risk Predictor**
   - Input form covering 23 clinical features grouped by:
     - *Demographics & Vitals* (Age, Biological Sex)
     - *Lifestyle Factors* (Smoking, Alcohol, Balanced Diet, Obesity)
     - *Environmental / Occupational Hazards* (Air Pollution, Dust Allergy, Occupational Hazards, Passive Smoke)
     - *Medical & Genetic Predisposition* (Genetic Risk, Chronic Lung Disease)
     - *Clinical Symptoms Matrix* (Hemoptysis/Coughing of Blood, Dyspnea, Chest Pain, Fatigue, Wheezing, Weight Loss, etc.)
   - One-click Clinical Presets (*Low Risk*, *Medium Risk*, *High Risk*, *Random Dataset Patient*).
   - Instant multi-class prediction badge with probability breakdown (*High %*, *Medium %*, *Low %*).
   - Automated Primary Risk Driver attribution (comparing patient parameters against population baselines).
   - Tailored clinical directives & downloadable medical assessment reports.

3. **📊 Interactive Analytics (EDA)**
   - Age vs. Smoking Scatter Distribution with risk clustering.
   - Biological Sex vs. Cancer Risk analysis.
   - Random Forest Feature Importance ranking (Gini impurity weights).
   - Pearson Correlation ($r$) with cancer progression.
   - Single-feature distribution explorer with dynamic statistical summaries.

4. **📋 Dataset Explorer**
   - Full 1,000 patient records explorer with live multi-column filtering:
     - Search by Patient ID
     - Filter by Cancer Risk Level
     - Filter by Biological Sex
     - Filter by Age Range
   - Descriptive cohort statistics and 1-click CSV dataset export.
   - 1-click patient record loading directly into the Risk Predictor.

5. **🤖 Machine Learning Pipeline & Benchmark**
   - Complete architectural walkthrough (MinMaxScaler $\rightarrow$ Random Forest).
   - Model benchmark matrix comparing Random Forest, AdaBoost, KNN, Decision Tree, and Logistic Regression.
   - 200-patient test set Confusion Matrix heatmap and multi-class classification report.
   - Gini importance ranking for all 23 features.

6. **ℹ️ About & Documentation**
   - Full feature dictionary, data specifications, system workflow, and medical disclaimers.

---

## 🚀 Quick Start

### 1. Prerequisites
Ensure you have Python 3.9+ installed.

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the Application
```bash
streamlit run streamlit_app.py
```
or double-click `run.bat` on Windows.

---

## 📂 Project Structure

```text
ML/
│
├── streamlit_app.py                # Main Streamlit Application Entry Point
├── train_model.py                  # Model Training & Pipeline Serialization
├── cancer patient data sets.xlsx   # Source Clinical Dataset (1,000 Records)
├── requirements.txt                # Python Dependencies
├── run.bat                         # Quick Launch Script for Windows
├── README.md                       # Project Documentation
│
└── models/
    ├── model.joblib                # Trained Random Forest Classifier
    └── scaler.joblib               # Fitted MinMaxScaler Preprocessor
```

---

## 🧬 Machine Learning Pipeline

```text
Patient Inputs (23 Features)
          │
          ▼
MinMaxScaler Normalization [0, 1]
          │
          ▼
Random Forest Classifier (150 Estimators)
          │
          ▼
Prediction: [Low | Medium | High] + Probabilities
```

---

## ⚠️ Medical Disclaimer

*PulmoVision AI is an educational and research-oriented machine learning platform. Its predictions are algorithmic estimations and do not constitute a certified medical diagnosis or clinical treatment plan. Always consult qualified healthcare professionals for medical advice.*
