"""
shap_utils.py
SHAP explainer creation, value computation, and plot generation.
"""

import os
import numpy as np
import pandas as pd
import shap
import matplotlib
import matplotlib.pyplot as plt
import streamlit as st

from utils.preprocessing import FEATURE_ORDER, FEATURE_LABELS

# Use non-interactive backend so matplotlib doesn't try to open windows
matplotlib.use("Agg")

# Path to the precomputed global SHAP values
_GLOBAL_SHAP_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "models", "global_shap.joblib"
)

# ------------------------------------------------------------------
# Caching
# ------------------------------------------------------------------
@st.cache_resource
def get_explainer(_model):
    """Create a SHAP TreeExplainer (cached)."""
    return shap.TreeExplainer(_model)


@st.cache_resource
def get_global_shap_data():
    """
    Load precomputed global SHAP values.
    Returns (shap_values, X_display)
    """
    import joblib
    data = joblib.load(_GLOBAL_SHAP_PATH)
    return data["shap_values"], data["X_display"]


# ------------------------------------------------------------------
# Local explanation helpers
# ------------------------------------------------------------------
def compute_local_shap(explainer, scaled_row: np.ndarray):
    """Compute SHAP values for a single instance."""
    return explainer.shap_values(scaled_row)


def get_top_contributors(shap_vals_row: np.ndarray, n: int = 5):
    """
    Return the top-n positive and top-n negative SHAP contributors.

    Returns
    -------
    positive : list[tuple[str, float]]
    negative : list[tuple[str, float]]
    """
    indices_pos = np.argsort(shap_vals_row)[::-1][:n]
    indices_neg = np.argsort(shap_vals_row)[:n]

    positive = [
        (FEATURE_LABELS.get(FEATURE_ORDER[i], FEATURE_ORDER[i]), float(shap_vals_row[i]))
        for i in indices_pos
        if shap_vals_row[i] > 0
    ]
    negative = [
        (FEATURE_LABELS.get(FEATURE_ORDER[i], FEATURE_ORDER[i]), float(shap_vals_row[i]))
        for i in indices_neg
        if shap_vals_row[i] < 0
    ]
    return positive, negative


# ------------------------------------------------------------------
# Plot helpers  (return matplotlib Figure objects)
# ------------------------------------------------------------------
def create_waterfall_plot(explainer, shap_vals_row, feature_values_row):
    """SHAP waterfall plot for one customer."""
    explanation = shap.Explanation(
        values=shap_vals_row,
        base_values=explainer.expected_value,
        data=feature_values_row,
        feature_names=[FEATURE_LABELS.get(f, f) for f in FEATURE_ORDER],
    )
    fig, ax = plt.subplots(figsize=(10, 7))
    plt.sca(ax)
    shap.plots.waterfall(explanation, show=False)
    plt.tight_layout()
    return fig


def create_summary_bar_plot(shap_values, X_display):
    """Global SHAP feature-importance bar plot."""
    fig, ax = plt.subplots(figsize=(10, 7))
    plt.sca(ax)
    shap.summary_plot(
        shap_values,
        X_display,
        plot_type="bar",
        feature_names=[FEATURE_LABELS.get(f, f) for f in FEATURE_ORDER],
        show=False,
    )
    plt.tight_layout()
    return fig


def create_beeswarm_plot(shap_values, X_display):
    """Global SHAP beeswarm (dot) plot."""
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.sca(ax)
    shap.summary_plot(
        shap_values,
        X_display,
        feature_names=[FEATURE_LABELS.get(f, f) for f in FEATURE_ORDER],
        show=False,
    )
    plt.tight_layout()
    return fig


def create_dependence_plot(shap_values, feature_name, X_display):
    """SHAP dependence plot for a selected feature."""
    idx = FEATURE_ORDER.index(feature_name)
    fig, ax = plt.subplots(figsize=(9, 6))
    shap.dependence_plot(
        idx,
        shap_values,
        X_display,
        feature_names=[FEATURE_LABELS.get(f, f) for f in FEATURE_ORDER],
        ax=ax,
        show=False,
    )
    plt.tight_layout()
    return fig


# ------------------------------------------------------------------
# Auto-generated insights
# ------------------------------------------------------------------
def generate_insights(shap_values, X_display):
    """
    Derive top-level textual insights from global SHAP values.
    Returns a list of (title, description, icon) tuples.
    """
    mean_abs = np.abs(shap_values).mean(axis=0)
    ranked = np.argsort(mean_abs)[::-1]

    insights = []
    templates = {
        "Contract": (
            "Contract Type Matters Most",
            "Contract type is the strongest predictor of churn. Month-to-month customers are significantly more likely to leave.",
            "📋",
        ),
        "tenure": (
            "Tenure Reduces Churn Risk",
            "Longer tenure significantly reduces churn risk. Customers beyond 2 years are much more likely to stay.",
            "⏳",
        ),
        "MonthlyCharges": (
            "High Monthly Charges Drive Churn",
            "Higher monthly charges are associated with increased churn probability. Cost sensitivity is a key factor.",
            "💰",
        ),
        "TotalCharges": (
            "Total Charges Reflect Loyalty",
            "Higher total charges indicate longer customer relationships and are associated with lower churn risk.",
            "💳",
        ),
        "OnlineSecurity": (
            "Online Security Retains Customers",
            "Customers without online security are more likely to churn. Bundling security services helps retention.",
            "🔒",
        ),
        "TechSupport": (
            "Tech Support is Protective",
            "Customers with tech support are significantly less likely to churn. Support services increase satisfaction.",
            "🛠️",
        ),
        "TotalServices": (
            "More Services, More Sticky",
            "Customers who subscribe to more premium services show lower churn rates due to deeper engagement.",
            "📦",
        ),
        "PaperlessBilling": (
            "Paperless Billing Signals Risk",
            "Paperless billing is associated with higher churn — these digitally-engaged customers switch providers more easily.",
            "📄",
        ),
        "PaymentMethod": (
            "Payment Method Influences Churn",
            "Electronic check users churn at much higher rates compared to automatic payment methods.",
            "💲",
        ),
    }

    for idx in ranked[:5]:
        feat = FEATURE_ORDER[idx]
        if feat in templates:
            insights.append(templates[feat])
        else:
            label = FEATURE_LABELS.get(feat, feat)
            insights.append((
                f"{label} Impacts Churn",
                f"{label} is among the top predictors of customer churn based on SHAP analysis.",
                "📊",
            ))

    return insights


def generate_recommendations(positive_contributors, negative_contributors):
    """
    Create dynamic business recommendations based on a customer's
    top SHAP contributors.

    Returns a list of (recommendation_text, icon) tuples.
    """
    recs = []
    pos_names = {name for name, _ in positive_contributors}

    # Match feature labels to actionable recommendations
    rec_map = {
        "Contract Type": (
            "Offer a discounted upgrade to a 1-year or 2-year contract with locked-in pricing to improve retention.",
            "📋",
        ),
        "Monthly Charges ($)": (
            "Review the customer's plan and offer a tailored discount or bundle to reduce their monthly bill.",
            "💰",
        ),
        "Tech Support": (
            "Provide a complimentary tech support package for 3-6 months to increase service satisfaction.",
            "🛠️",
        ),
        "Online Security": (
            "Bundle a free online security add-on to increase value perception and service stickiness.",
            "🔒",
        ),
        "Tenure (months)": (
            "Engage the customer with a loyalty rewards programme — they are still in the early relationship phase.",
            "⏳",
        ),
        "Paperless Billing": (
            "Reach out with personalised retention offers via email, since the customer prefers digital communication.",
            "📄",
        ),
        "Payment Method": (
            "Encourage switching to automatic payment (bank transfer or credit card) by offering a small monthly discount.",
            "💲",
        ),
        "Device Protection": (
            "Offer a device protection plan to add value and increase the customer's switching cost.",
            "📱",
        ),
        "Online Backup": (
            "Provide a trial of the online backup service to deepen engagement with the platform.",
            "☁️",
        ),
        "Total Premium Services": (
            "Cross-sell additional services to deepen the customer's engagement with the platform.",
            "📦",
        ),
        "Avg Monthly Charge ($)": (
            "Analyse the customer's usage pattern and suggest a more cost-effective plan to improve satisfaction.",
            "📊",
        ),
    }

    for name in pos_names:
        if name in rec_map:
            recs.append(rec_map[name])

    # Always add a general retention recommendation
    if not recs:
        recs.append((
            "Proactively reach out to the customer with a personalised retention offer and satisfaction survey.",
            "🎯",
        ))

    # Cap at 5 recommendations
    return recs[:5]
