# 🏥 Hospital Readmission Penalty Predictor

### *Explainable AI–Powered Medicare Penalty Risk Analysis*

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B.svg)](https://streamlit.io/)
[![Machine Learning](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![Healthcare Analytics](https://img.shields.io/badge/Domain-Healthcare-green.svg)]()
[![License](https://img.shields.io/badge/License-MIT-brightgreen.svg)](LICENSE)

A **production-ready machine learning web application** that predicts **Medicare Hospital Readmission penalties** and, uniquely, **explains *why* a hospital is at risk** using CMS-aligned rule-based logic.

Built with **Streamlit** and powered by **6 advanced ML models**, this application enables **data-driven decision-making, transparency, and operational insight** for healthcare administrators.

---

## 🎯 Project Overview

The **Hospital Readmissions Reduction Program (HRRP)** is a CMS initiative that penalizes hospitals with excessive readmission rates by reducing Medicare reimbursements.

This application goes beyond prediction by adding **Explainable AI** to answer the critical question:

> **“Why is this hospital at risk of a penalty?”**

### What Makes This Project Different ✨

✔ Predicts penalty risk
✔ Shows probability & confidence
✔ **Explains top CMS-defined risk drivers**
✔ Uses **transparent, rule-based logic aligned with HRRP policy**
✔ Designed for **real-world healthcare decision-making**

---

## 📊 Dataset

* **Source:** CMS FY-2025 Hospital Readmissions Reduction Program
* **Coverage:**

  * 10,000+ hospital-condition records
  * All U.S. states, DC, and territories
* **Medical Conditions:**

  * Heart Attack
  * Heart Failure
  * Pneumonia
  * COPD
  * Hip/Knee Replacement
  * CABG

---

## 🚀 Core Capabilities

### ✅ Machine Learning

* 6 optimized classification models
* Accuracy range: **93% – 98%**
* Stratified train-test split
* 5-fold cross-validation

### ✅ Advanced Explainability

* **Top Risk Drivers Identification**
* **Direction of impact (↑ increases / ↓ reduces risk)**
* **Plain-language explanations**
* **CMS rule-based logic (not black-box SHAP only)**

Example explanation shown in app:

> *“Predicted readmission rate exceeds CMS benchmark and excess readmission ratio is above 1.0, directly triggering penalty risk.”*

---

## 🤖 Machine Learning Models Used

| Model                  | Type              | Accuracy |
| ---------------------- | ----------------- | -------- |
| Logistic Regression    | Linear Classifier | ~98%     |
| Decision Tree          | Rule-based        | ~93%     |
| Support Vector Machine | RBF Kernel        | ~95–96%  |
| Random Forest          | Bagging Ensemble  | ~86%     |
| Gradient Boosting      | Boosting Ensemble | ~97–98%  |
| Neural Network (MLP)   | Deep Learning     | ~95–97%  |

---

## 🧠 Explainable AI – Risk Driver Engine (Your Contribution)

Unlike standard ML dashboards, this app **explicitly explains penalty risk** using **CMS-aligned rules**.

### Risk Drivers Considered

* Predicted vs Expected Readmission Rate
* Excess Readmission Ratio (>1 is a direct CMS trigger)
* Readmissions per discharge
* Discharge volume (financial exposure)
* Benchmark compliance

### Output Provided

* **Top 5 drivers**
* **Impact direction:** ↑ Increases Risk / ↓ Reduces Risk
* **Clear, non-technical explanation**
* **Actionable interpretation**

This makes the model:
✔ Auditable
✔ Interpretable
✔ Clinically & administratively useful

---

## 🧩 Application Features

### 1️⃣ Data Overview

* Dataset statistics
* Sample preview
* Column metadata
* Statistical summaries

### 2️⃣ Data Preprocessing

* Missing value handling
* Feature scaling
* Encoding
* Correlation analysis
* Outlier detection

### 3️⃣ Exploratory Data Analysis

* Penalty distribution
* Readmission patterns
* Condition-wise risk
* State-wise trends

### 4️⃣ Model Training

* Single-model training
* Performance metrics
* Confusion matrix
* ROC curve
* Classification report

### 5️⃣ Model Comparison

* Side-by-side evaluation
* Best model identification
* Combined ROC curves

### 6️⃣ Advanced Live Prediction System

* **Single Model (Fast)**
* **Multi-Model Consensus**
* Risk probability & confidence
* Visual risk gauge
* **Explainable Risk Drivers (Your addition)**

---

## 📈 Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC
* Cross-Validation Mean & Std

---

## 🛠️ Installation & Setup

### Prerequisites

* Python 3.8+
* pip

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the App

```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
Hospital-Readmission-Penalty-Predictor/
│
├── app.py
├── requirements.txt
├── README.md
├── FY_2025_Hospital_Readmissions_Reduction_Program_Hospital.csv
```

---

## 💡 Use Cases

### Healthcare Administrators

* Identify penalty risk early
* Understand *why* risk exists
* Plan quality improvement initiatives

### Data & Business Analysts

* End-to-end ML pipeline
* Explainable healthcare analytics
* Production-ready Streamlit app

### Recruiters & Interviewers

* Real-world dataset
* Explainability (key differentiator)
* Clear business impact

---

## 🔮 Future Enhancements

* SHAP-based explainability comparison
* Time-series trend analysis
* Hospital clustering
* User authentication

---

## 👤 Author

**Likhitha S**

* LinkedIn: [https://www.linkedin.com/in/likhitha-s-12144323b](https://www.linkedin.com/in/likhitha-s-12144323b)
* GitHub: [https://github.com/Likhitha-S806](https://github.com/Likhitha-S806)

> *This project extends a baseline ML predictor by adding explainable, CMS-aligned risk interpretation — making it suitable for real healthcare decision-making.*

---

## ⭐ Final Note

If you find this project useful or insightful, please ⭐ the repository.

**Status:** ✅ Production Ready
**Domain:** Healthcare Analytics
**Focus:** Explainable AI + ML
**Last Updated:** 2025


