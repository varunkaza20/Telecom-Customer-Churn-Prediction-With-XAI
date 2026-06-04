# 📡 Telecom Customer Churn Prediction & Explainability System

A production-ready **Streamlit** web application that predicts telecom customer churn risk and provides actionable explanations using **SHAP** (SHapley Additive exPlanations).

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🔮 **Churn Prediction** | Enter customer details and get instant churn probability with risk categorisation |
| 🔍 **Local SHAP Explanation** | Waterfall plot showing which features drove each individual prediction |
| 💡 **Business Recommendations** | Dynamically generated retention strategies based on SHAP analysis |
| 🌍 **Global SHAP Interpretation** | Feature importance bar, beeswarm, and dependence plots |
| 📊 **Model Comparison** | Side-by-side metrics for Logistic Regression, Decision Tree, Random Forest, XGBoost, and Stacking |

---

## 🛠️ Technology Stack

- **Frontend**: Streamlit with custom CSS (light glassmorphism theme)
- **ML Model**: XGBoost Classifier
- **Explainability**: SHAP TreeExplainer
- **Visualisations**: Plotly · Matplotlib · Seaborn
- **Data Processing**: Pandas · NumPy · Scikit-Learn

---

## 📂 Project Structure

```
Telecom_Customer_Churn_Prediction/
├── app.py                      # Main Streamlit application
├── models/
│   ├── best_model.pkl          # XGBoost production model
│   └── scaler.pkl              # StandardScaler
├── utils/
│   ├── __init__.py
│   ├── preprocessing.py        # Feature encoding & engineering
│   ├── prediction.py           # Model loading & inference
│   └── shap_utils.py           # SHAP explainer & plots
├── assets/
│   └── style.css               # Custom light theme CSS
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone <repo-url>
cd Telecom_Customer_Churn_Prediction
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run app.py
```

The app will open at **http://localhost:8501**.

---

## ☁️ Deploy to Streamlit Community Cloud

1. Push this repository to **GitHub**.
2. Go to [share.streamlit.io](https://share.streamlit.io).
3. Select the repository, branch, and set the main file path to `app.py`.
4. Click **Deploy**.

> **Note**: Ensure `models/best_model.pkl` and `models/scaler.pkl` are committed to the repository.

---

## 📊 Model Performance

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Stacking Classifier | 85.6% | 85.4% | 86.4% | 85.9% | 93.3% |
| Random Forest | 85.3% | 84.8% | 86.5% | 85.7% | 93.0% |
| **XGBoost** *(deployed)* | **83.9%** | **82.7%** | **86.4%** | **84.5%** | **92.4%** |
| Logistic Regression | 80.9% | 78.9% | 85.2% | 81.9% | 89.4% |
| Decision Tree | 80.4% | 78.7% | 84.1% | 81.3% | 87.8% |

---

## 📝 License

This project is for educational and demonstration purposes.
