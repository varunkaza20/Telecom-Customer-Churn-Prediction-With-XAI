# 📡 Telecom Customer Churn Prediction & Explainability (XAI) System

Deploy Link: [telecom-customer-churn-prediction-with-xai.streamlit.app](https://telecom-customer-churn-prediction-with-xai.streamlit.app)

---

## 📖 About the Project

### Problem Statement
* In telecom, customers can easily switch companies.
* Acquisition costs are high, making retention critical.
* Churn rate measures how many users cancel subscriptions.
* Predicting churn helps businesses improve services and protect revenue.

### Aim
* Classify potential churn cases using customer demographics and usage metrics.
* Solve a binary classification task on an imbalanced dataset.

---

## 📊 Dataset Information

### Context
* Analyze customer data to build targeted retention programs. (IBM Sample Dataset)

### Content
Each row is one customer, with attributes covering:
* **Churn status:** Whether the customer left within the last month.
* **Services:** Phone, internet, online security, device protection, tech support, etc.
* **Account Info:** Tenure, contract type, payment method, paperless billing, monthly charges, and total charges.
* **Demographics:** Gender, senior citizen status, partners, and dependents.

### Dataset Attributes
* **customerID**: Unique ID
* **gender**: Male or Female
* **SeniorCitizen**: Senior citizen or not (1, 0)
* **Partner**: Has partner or not (Yes, No)
* **Dependents**: Has dependents or not (Yes, No)
* **tenure**: Months stayed with the company
* **PhoneService**: Has phone service or not (Yes, No)
* **MultipleLines**: Has multiple lines or not (Yes, No, No phone service)
* **InternetService**: Internet provider (DSL, Fiber optic, No)
* **OnlineSecurity**: Has online security or not (Yes, No, No internet service)
* **OnlineBackup**: Has online backup or not (Yes, No, No internet service)
* **DeviceProtection**: Has device protection or not (Yes, No, No internet service)
* **TechSupport**: Has tech support or not (Yes, No, No internet service)
* **StreamingTV**: Has streaming TV or not (Yes, No, No internet service)
* **StreamingMovies**: Has streaming movies or not (Yes, No, No internet service)
* **Contract**: Contract term (Month-to-month, One year, Two year)
* **PaperlessBilling**: Has paperless billing or not (Yes, No)
* **PaymentMethod**: Payment method (Electronic check, Mailed check, Bank transfer, Credit card)
* **MonthlyCharges**: Monthly charge amount
* **TotalCharges**: Total charge amount
* **Churn**: Churn status (Yes, No)

---

## ⚙️ Model Training & ML Techniques

1. **Preprocessing**:
   - Categorical columns mapped to numerical values.
   - Numerical metrics scaled using `StandardScaler`.
2. **SMOTE**:
   - Synthetic Minority Over-sampling Technique used to balance the dataset.
   - Prevents the model from favoring the majority class.
3. **Model Selection**:
   - Compared Logistic Regression, Decision Trees, Random Forest, and XGBoost.
   - Deployed **XGBoost Classifier** for its high accuracy (83.9%) and recall (86.4%).

---

## 🔍 SHAP Interpretability & Insights

SHAP (SHapley Additive exPlanations) is a game-theoretic approach to explain model predictions using Shapley values.

### Local Explanations
* **Waterfall Plot:** Shows how individual customer features shift the churn risk.
* **Recommendations:** Automatically suggests retention actions based on risk drivers.

### Global Insights
* **Feature Importance:** Identifies **Contract Type** and **Tenure** as the top drivers of churn.
* **Beeswarm Plot:** Shows that long tenure decreases churn risk, while month-to-month contracts and high monthly charges increase it.
* **Dependence Analysis:** Identifies a sharp increase in churn risk when monthly charges exceed $70–$80.

---

## 🛠️ Technology Stack

* **Language**: Python
* **Dashboard**: Streamlit (light glassmorphism theme)
* **ML Libraries**: XGBoost, Scikit-learn, Joblib
* **Explainability**: SHAP
* **Visuals**: Plotly, Matplotlib, Seaborn
* **Data**: Pandas, NumPy

---

## 📂 Project Structure

```
Telecom_Customer_Churn_Prediction/
├── app.py                      # Streamlit entry point
├── models/
│   ├── best_model.pkl          # Deployed XGBoost model
│   └── scaler.pkl              # Scaler object
├── utils/
│   ├── preprocessing.py        # Feature mapping and scaling
│   ├── prediction.py           # Inference logic
│   └── shap_utils.py           # SHAP plots and insights
├── assets/
│   └── style.css               # Glassmorphism CSS styling
├── pages/
│   ├── prediction.py           # Churn prediction tab
│   ├── shap_global.py          # Global SHAP insights tab
│   └── model_comparison.py     # Model evaluation tab
├── requirements.txt            # Package requirements
└── README.md                   # Documentation
```

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/varunkaza20/Telecom-Customer-Churn-Prediction-With-XAI.git
cd Telecom_Customer_Churn_Prediction
```

### 2. Set up virtual environment
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the app
```bash
streamlit run app.py
```

---

## 📊 Model Performance Comparison

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Stacking Classifier | 85.6% | 85.4% | 86.4% | 85.9% | 93.3% |
| Random Forest | 85.3% | 84.8% | 86.5% | 85.7% | 93.0% |
| **XGBoost** *(deployed)* | **83.9%** | **82.7%** | **86.4%** | **84.5%** | **92.4%** |
| Logistic Regression | 80.9% | 78.9% | 85.2% | 81.9% | 89.4% |
| Decision Tree | 80.4% | 78.7% | 84.1% | 81.3% | 87.8% |

### Why XGBoost was chosen for deployment:
* **SHAP Compatibility:** XGBoost works natively with SHAP's fast `TreeExplainer` (runs in milliseconds). Explaining Stacking Classifiers requires slow model-agnostic explainers (`KernelExplainer`), causing significant lag.
* **Inference Speed:** XGBoost has much lower latency, making it ideal for real-time web dashboard predictions.
* **Simplicity vs. Performance:** The Stacking Classifier adds heavy training and computational complexity for a marginal performance gain (~1.7% accuracy increase).

---

## 📝 License

This project is for educational and demonstration purposes.
