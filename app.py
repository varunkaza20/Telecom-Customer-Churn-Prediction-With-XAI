"""
Telecom Customer Churn Prediction & Explainability System
=========================================================
Production-ready Streamlit dashboard with SHAP explainability.
Entry point for the application.
"""

import streamlit as st

from config.settings import PAGE_CONFIG
from config.theme import load_css
from components.sidebar import render_sidebar
from pages import prediction, shap_global, model_comparison

# ── Page configuration ───────────────────────────────────────────
st.set_page_config(**PAGE_CONFIG)

def main():
    # Load CSS theme
    load_css()

    # Render sidebar and get navigation choice
    render_sidebar()

    # Initialize session state for active tab
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = "Prediction"

    # ── Page header ──────────────────────────────────────────────
    st.markdown(
        """
        <div style="text-align:center; padding:0.5rem 0 0.2rem;">
            <h1 style="margin:0; font-size:1.8rem; font-weight:800;
                background:linear-gradient(135deg, #2563eb, #7c3aed, #0891b2);
                -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
                Telecom Customer Churn Prediction & Explainability System
            </h1>
            <p style="color:#475569; font-size:0.95rem; margin-top:0.3rem;">
                Predict customer churn risk and understand the driving factors
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Navigation Buttons ───────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Customer Churn Prediction", use_container_width=True):
            st.session_state.active_tab = "Prediction"
    with col2:
        if st.button("SHAP Global Interpretation", use_container_width=True):
            st.session_state.active_tab = "SHAP"
    with col3:
        if st.button("Model Comparison", use_container_width=True):
            st.session_state.active_tab = "Comparison"

    st.markdown("---")

    # ── Route to selected tab ────────────────────────────────────
    if st.session_state.active_tab == "Prediction":
        prediction.render()
    elif st.session_state.active_tab == "SHAP":
        shap_global.render()
    elif st.session_state.active_tab == "Comparison":
        model_comparison.render()

if __name__ == "__main__":
    main()
