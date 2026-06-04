"""
pages/prediction.py
Tab 1: Customer Churn Prediction
"""

import streamlit as st
import matplotlib.pyplot as plt

from utils.preprocessing import ENCODING_MAPS
from utils.prediction import load_model, load_scaler, predict_churn, get_risk_category
from utils.shap_utils import (
    get_explainer,
    compute_local_shap,
    get_top_contributors,
    create_waterfall_plot,
    generate_recommendations,
)
from components.charts import create_gauge


def render():
    # Header
    st.markdown('<div class="header-gradient"></div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="section-title">Customer Churn Prediction</div>
        <div class="section-subtitle">
            Enter customer details to predict churn risk and understand contributing factors
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Input form ───────────────────────────────────────────────
    with st.form("prediction_form"):
        col1, col2, col3, col4 = st.columns(4)

        # — Customer Information —
        with col1:
            st.markdown(
                '<p class="form-section-title">👤 Customer Info</p>',
                unsafe_allow_html=True,
            )
            gender = st.selectbox("Gender", ["Male", "Female"], key="gender")
            senior = st.selectbox("Senior Citizen", ["No", "Yes"], key="senior")
            partner = st.selectbox("Partner", ["No", "Yes"], key="partner")
            dependents = st.selectbox("Dependents", ["No", "Yes"], key="deps")

        # — Service Information —
        with col2:
            st.markdown(
                '<p class="form-section-title">📶 Services</p>',
                unsafe_allow_html=True,
            )
            phone = st.selectbox("Phone Service", ["Yes", "No"], key="phone")
            multi_lines = st.selectbox(
                "Multiple Lines",
                ["No", "Yes", "No phone service"],
                key="multilines",
            )
            internet = st.selectbox(
                "Internet Service", ["DSL", "Fiber optic", "No"], key="internet"
            )
            online_sec = st.selectbox(
                "Online Security",
                list(ENCODING_MAPS["OnlineSecurity"].keys()),
                key="osec",
            )

        with col3:
            st.markdown(
                '<p class="form-section-title">🛡️ Add-ons</p>',
                unsafe_allow_html=True,
            )
            online_bkp = st.selectbox(
                "Online Backup",
                list(ENCODING_MAPS["OnlineBackup"].keys()),
                key="obkp",
            )
            dev_prot = st.selectbox(
                "Device Protection",
                list(ENCODING_MAPS["DeviceProtection"].keys()),
                key="dprot",
            )
            tech_sup = st.selectbox(
                "Tech Support",
                list(ENCODING_MAPS["TechSupport"].keys()),
                key="tsup",
            )
            streaming_tv = st.selectbox(
                "Streaming TV", ["No", "Yes", "No internet service"], key="stv"
            )
            streaming_mov = st.selectbox(
                "Streaming Movies", ["No", "Yes", "No internet service"], key="smov"
            )

        # — Account & Billing —
        with col4:
            st.markdown(
                '<p class="form-section-title">💳 Account & Billing</p>',
                unsafe_allow_html=True,
            )
            contract = st.selectbox(
                "Contract", list(ENCODING_MAPS["Contract"].keys()), key="contract"
            )
            paperless = st.selectbox(
                "Paperless Billing",
                list(ENCODING_MAPS["PaperlessBilling"].keys()),
                key="paperless",
            )
            pay_method = st.selectbox(
                "Payment Method",
                list(ENCODING_MAPS["PaymentMethod"].keys()),
                key="paymethod",
            )
            tenure = st.number_input(
                "Tenure (months)", min_value=0, max_value=72, value=12, step=1, key="tenure"
            )
            monthly = st.number_input(
                "Monthly Charges ($)", min_value=18.0, max_value=120.0,
                value=65.0, step=0.5, key="monthly",
            )
            total = st.number_input(
                "Total Charges ($)", min_value=18.0, max_value=9000.0,
                value=780.0, step=10.0, key="total",
            )

        submitted = st.form_submit_button("🔮  Predict Churn", use_container_width=True)

    # ── Prediction flow ──────────────────────────────────────────
    if submitted:
        raw_input = {
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "OnlineSecurity": online_sec,
            "OnlineBackup": online_bkp,
            "DeviceProtection": dev_prot,
            "TechSupport": tech_sup,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": pay_method,
            "MonthlyCharges": monthly,
            "TotalCharges": total,
        }

        with st.spinner("Analysing customer profile…"):
            model = load_model()
            scaler = load_scaler()
            prediction, probability, scaled, encoded_df = predict_churn(
                model, scaler, raw_input
            )
            risk_label, risk_color, risk_emoji = get_risk_category(probability)

        st.markdown("---")

        # ── Result cards ─────────────────────────────────────────
        res_col1, res_col2 = st.columns([1.2, 1])

        with res_col1:
            card_cls = "prediction-churn" if prediction == 1 else "prediction-stay"
            result_text = (
                "⚠️ Customer is <b>likely to churn</b>"
                if prediction == 1
                else "✅ Customer is <b>likely to stay</b>"
            )
            risk_badge_cls = (
                "risk-high" if risk_label == "High Risk"
                else "risk-medium" if risk_label == "Medium Risk"
                else "risk-low"
            )

            st.markdown(
                f"""
                <div class="glass-card {card_cls}">
                    <h3 style="margin:0 0 0.8rem; color:var(--text-primary); font-size:1.2rem;">
                        Prediction Result
                    </h3>
                    <p style="font-size:1.15rem; color:var(--text-secondary); margin-bottom:1rem;">
                        {result_text}
                    </p>
                    <div style="display:flex; align-items:center; gap:1.2rem; flex-wrap:wrap;">
                        <div>
                            <span style="color:var(--text-muted); font-size:0.8rem; text-transform:uppercase;
                                letter-spacing:0.5px;">Churn Probability</span><br>
                            <span class="prob-display" style="color:{risk_color};">
                                {probability * 100:.1f}%
                            </span>
                        </div>
                        <div>
                            <span class="risk-badge {risk_badge_cls}">
                                {risk_emoji} {risk_label}
                            </span>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Progress bar
            st.markdown(
                f"""
                <div style="margin-top:0.5rem;">
                    <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                        <span style="color:var(--text-muted); font-size:0.78rem;">Stay</span>
                        <span style="color:var(--text-muted); font-size:0.78rem;">Churn</span>
                    </div>
                    <div style="background:rgba(255,255,255,0.06); border-radius:8px;
                                height:12px; overflow:hidden;">
                        <div style="width:{probability * 100:.1f}%; height:100%;
                                    background:linear-gradient(90deg, #10b981, #f59e0b, #f43f5e);
                                    border-radius:8px; transition:width 0.8s ease;"></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with res_col2:
            gauge_fig = create_gauge(probability, risk_color)
            st.plotly_chart(gauge_fig, use_container_width=True, key="gauge")

        # ── Local SHAP explanation ───────────────────────────────
        st.markdown("---")
        st.markdown(
            """
            <div class="section-title">🔍 SHAP Explanation</div>
            <div class="section-subtitle">
                Understanding which factors influenced this prediction
            </div>
            """,
            unsafe_allow_html=True,
        )

        explainer = get_explainer(model)
        shap_vals = compute_local_shap(explainer, scaled)
        shap_row = shap_vals[0]

        # Waterfall plot
        fig_wf = create_waterfall_plot(
            explainer, shap_row, encoded_df.iloc[0].values
        )
        st.pyplot(fig_wf, use_container_width=True)
        plt.close(fig_wf)

        # Top contributors
        positive, negative = get_top_contributors(shap_row)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                '<p style="color:#9f1239; font-weight:600; font-size:0.95rem; '
                'margin-bottom:0.6rem;">⬆️ Top Factors Increasing Churn Risk</p>',
                unsafe_allow_html=True,
            )
            if positive:
                for name, val in positive:
                    st.markdown(
                        f"""
                        <div class="contributor-pos">
                            <span style="color:#9f1239; font-size:0.9rem;">{name}</span>
                            <span style="color:#9f1239; font-weight:600;
                                font-family:'JetBrains Mono',monospace; font-size:0.85rem;">
                                +{val:.4f}
                            </span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            else:
                st.info("No features significantly increasing churn risk.")

        with c2:
            st.markdown(
                '<p style="color:#065f46; font-weight:600; font-size:0.95rem; '
                'margin-bottom:0.6rem;">⬇️ Top Factors Reducing Churn Risk</p>',
                unsafe_allow_html=True,
            )
            if negative:
                for name, val in negative:
                    st.markdown(
                        f"""
                        <div class="contributor-neg">
                            <span style="color:#065f46; font-size:0.9rem;">{name}</span>
                            <span style="color:#065f46; font-weight:600;
                                font-family:'JetBrains Mono',monospace; font-size:0.85rem;">
                                {val:.4f}
                            </span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            else:
                st.info("No features significantly reducing churn risk.")

        # ── Business recommendations ─────────────────────────────
        st.markdown("---")
        st.markdown(
            """
            <div class="section-title">💡 Business Recommendations</div>
            <div class="section-subtitle">
                Actionable retention strategies based on this customer's risk profile
            </div>
            """,
            unsafe_allow_html=True,
        )

        recs = generate_recommendations(positive, negative)
        for text, icon in recs:
            st.markdown(
                f"""
                <div class="rec-card">
                    <span style="font-size:1.3rem; margin-right:0.6rem;">{icon}</span>
                    <span style="color:var(--text-primary); font-size:0.92rem;">{text}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
