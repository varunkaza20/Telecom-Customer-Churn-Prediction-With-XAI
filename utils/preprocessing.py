"""
preprocessing.py
Feature encoding, engineering, and scaling utilities.
Mirrors the exact pipeline from the training notebooks.
"""

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# 1.  Feature order — must match scaler.feature_names_in_
# ---------------------------------------------------------------------------
FEATURE_ORDER = [
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "TotalServices",
    "TenureGroup",
    "AvgMonthlyCharge",
]

# ---------------------------------------------------------------------------
# 2.  Encoding maps  (LabelEncoder — alphabetical order)
# ---------------------------------------------------------------------------
ENCODING_MAPS = {
    "SeniorCitizen": {"No": 0, "Yes": 1},
    "Partner": {"No": 0, "Yes": 1},
    "Dependents": {"No": 0, "Yes": 1},
    "PaperlessBilling": {"No": 0, "Yes": 1},
    "OnlineSecurity": {"No": 0, "No internet service": 1, "Yes": 2},
    "OnlineBackup": {"No": 0, "No internet service": 1, "Yes": 2},
    "DeviceProtection": {"No": 0, "No internet service": 1, "Yes": 2},
    "TechSupport": {"No": 0, "No internet service": 1, "Yes": 2},
    "Contract": {"Month-to-month": 0, "One year": 1, "Two year": 2},
    "PaymentMethod": {
        "Bank transfer (automatic)": 0,
        "Credit card (automatic)": 1,
        "Electronic check": 2,
        "Mailed check": 3,
    },
}

# Reverse maps for display purposes (encoded → label)
DECODING_MAPS = {
    feat: {v: k for k, v in mapping.items()}
    for feat, mapping in ENCODING_MAPS.items()
}

# Human-readable feature labels for SHAP / UI
FEATURE_LABELS = {
    "SeniorCitizen": "Senior Citizen",
    "Partner": "Partner",
    "Dependents": "Dependents",
    "tenure": "Tenure (months)",
    "OnlineSecurity": "Online Security",
    "OnlineBackup": "Online Backup",
    "DeviceProtection": "Device Protection",
    "TechSupport": "Tech Support",
    "Contract": "Contract Type",
    "PaperlessBilling": "Paperless Billing",
    "PaymentMethod": "Payment Method",
    "MonthlyCharges": "Monthly Charges ($)",
    "TotalCharges": "Total Charges ($)",
    "TotalServices": "Total Premium Services",
    "TenureGroup": "Tenure Group",
    "AvgMonthlyCharge": "Avg Monthly Charge ($)",
}


def encode_input(raw: dict) -> pd.DataFrame:
    """
    Convert raw user inputs (UI labels) to the encoded DataFrame
    expected by the model.

    Parameters
    ----------
    raw : dict
        Keys are feature names; values are human-readable labels
        (e.g. ``{"Contract": "Month-to-month", "tenure": 12, ...}``).

    Returns
    -------
    pd.DataFrame
        Single-row DataFrame with 16 columns in ``FEATURE_ORDER``.
    """
    encoded = {}

    # --- categorical features ---
    for feat, mapping in ENCODING_MAPS.items():
        value = raw.get(feat)
        if value is not None:
            encoded[feat] = mapping.get(value, value)
        else:
            encoded[feat] = 0  # safe default

    # --- numeric features (pass-through) ---
    encoded["tenure"] = float(raw.get("tenure", 0))
    encoded["MonthlyCharges"] = float(raw.get("MonthlyCharges", 0))
    encoded["TotalCharges"] = float(raw.get("TotalCharges", 0))

    # --- engineered features ---
    service_cols = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport"]
    encoded["TotalServices"] = sum(1 for s in service_cols if encoded.get(s) == 2)

    tenure_val = encoded["tenure"]
    if tenure_val <= 12:
        encoded["TenureGroup"] = 0
    elif tenure_val <= 24:
        encoded["TenureGroup"] = 1
    elif tenure_val <= 48:
        encoded["TenureGroup"] = 2
    else:
        encoded["TenureGroup"] = 3

    encoded["AvgMonthlyCharge"] = encoded["TotalCharges"] / (encoded["tenure"] + 1)

    # Build single-row DataFrame in the exact feature order
    df = pd.DataFrame([encoded])[FEATURE_ORDER]
    return df


def scale_features(df: pd.DataFrame, scaler) -> np.ndarray:
    """Apply the fitted StandardScaler and return a 2-D array."""
    return scaler.transform(df)
