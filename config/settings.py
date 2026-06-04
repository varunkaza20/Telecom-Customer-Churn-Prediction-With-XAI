"""
config/settings.py
Application-wide constants: Plotly layout, model comparison metrics, page metadata.
"""

import pandas as pd


# ── Plotly light theme template ───────────────────────────────────
PLOTLY_LAYOUT = dict(
    template="plotly_white",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#0f172a"),
    margin=dict(l=40, r=40, t=50, b=40),
)

# ── Chart colour palette ────────────────────────────────────────
CHART_COLORS = ["#8b5cf6", "#3b82f6", "#06b6d4", "#f59e0b", "#f43f5e"]

# ── Model comparison metrics (from training notebook) ────────────
MODEL_METRICS = pd.DataFrame([
    {"Model": "Stacking Classifier", "Accuracy": 0.8560, "Precision": 0.8538,
     "Recall": 0.8643, "F1 Score": 0.8590, "ROC-AUC": 0.9326},
    {"Model": "Random Forest",       "Accuracy": 0.8528, "Precision": 0.8479,
     "Recall": 0.8654, "F1 Score": 0.8566, "ROC-AUC": 0.9303},
    {"Model": "XGBoost",             "Accuracy": 0.8392, "Precision": 0.8269,
     "Recall": 0.8643, "F1 Score": 0.8452, "ROC-AUC": 0.9240},
    {"Model": "Logistic Regression",  "Accuracy": 0.8090, "Precision": 0.7888,
     "Recall": 0.8520, "F1 Score": 0.8192, "ROC-AUC": 0.8936},
    {"Model": "Decision Tree",       "Accuracy": 0.8038, "Precision": 0.7873,
     "Recall": 0.8407, "F1 Score": 0.8131, "ROC-AUC": 0.8778},
])

# ── Page configuration ──────────────────────────────────────────
PAGE_CONFIG = dict(
    page_title="Telecom Churn Prediction",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Navigation options ───────────────────────────────────────────
NAV_OPTIONS = [
    "Customer Churn Prediction",
]
