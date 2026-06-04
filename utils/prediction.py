"""
prediction.py
Model loading, inference, and risk categorisation.
"""

import os
import joblib
import numpy as np
import streamlit as st

from utils.preprocessing import encode_input, scale_features

# Path to saved artefacts
_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


@st.cache_resource
def load_model():
    """Load the production XGBoost model (cached)."""
    path = os.path.join(_MODEL_DIR, "best_model.pkl")
    return joblib.load(path)


@st.cache_resource
def load_scaler():
    """Load the fitted StandardScaler (cached)."""
    path = os.path.join(_MODEL_DIR, "scaler.pkl")
    return joblib.load(path)


def predict_churn(model, scaler, raw_input: dict):
    """
    End-to-end prediction pipeline.

    Returns
    -------
    prediction : int
        0 (stay) or 1 (churn).
    probability : float
        Churn probability (class-1 probability), 0-1.
    scaled_features : np.ndarray
        Scaled feature vector (used downstream for SHAP).
    encoded_df : pd.DataFrame
        Encoded (but un-scaled) feature DataFrame.
    """
    encoded_df = encode_input(raw_input)
    scaled = scale_features(encoded_df, scaler)

    prediction = int(model.predict(scaled)[0])
    probability = float(model.predict_proba(scaled)[0][1])

    return prediction, probability, scaled, encoded_df


def get_risk_category(probability: float):
    """
    Map churn probability to a human-readable risk level.

    Returns
    -------
    label : str
    color : str   (CSS colour)
    emoji : str
    """
    if probability < 0.35:
        return "Low Risk", "#00C853", "🟢"
    elif probability < 0.65:
        return "Medium Risk", "#FF9100", "🟡"
    else:
        return "High Risk", "#FF1744", "🔴"
