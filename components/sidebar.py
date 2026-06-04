"""
components/sidebar.py
Sidebar component for navigation and information.
"""

import streamlit as st


def render_sidebar():
    with st.sidebar:
        # Logo / branding
        st.markdown(
            """
            <div style="text-align:center; padding: 1.2rem 0 0.6rem;">
                <span style="font-size:2.8rem;"></span>
                <h2 style="margin:0.3rem 0 0; background: linear-gradient(135deg, #2563eb, #7c3aed);
                    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                    font-weight:800; font-size:1.25rem; line-height:1.3;">
                    Telecom Churn<br>Prediction System
                </h2>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("---")

        # ── Project overview ──
        with st.expander("Project Overview", expanded=False):
            st.markdown(
                """
                An end-to-end ML system that predicts whether a telecom
                customer will **churn** (leave) and provides actionable
                explanations using **SHAP** values.
                """
            )

        # ── Dataset info ──
        with st.expander("Dataset Information", expanded=False):
            st.markdown(
                """
                | Item | Value |
                |------|-------|
                | **Source** | Telecom Customer Churn |
                | **Samples** | 7 043 customers |
                | **Features** | 16 (incl. engineered) |
                | **Target** | Churn (binary) |
                | **Class Balance** | 73.5 % Stay · 26.5 % Churn |
                """
            )

        # ── Model info ──
        with st.expander("Model Information", expanded=False):
            st.markdown(
                """
                | Item | Value |
                |------|-------|
                | **Production Model** | XGBoost Classifier |
                | **Accuracy** | 83.9 % |
                | **F1 Score** | 84.5 % |
                | **ROC-AUC** | 92.4 % |
                | **Scaler** | StandardScaler |
                | **Explainer** | SHAP TreeExplainer |
                """
            )

        st.markdown("---")
        st.markdown(
            '<p style="text-align:center; color:#475569; font-size:0.72rem;">'
            "Built with Streamlit · SHAP · XGBoost</p>",
            unsafe_allow_html=True,
        )
