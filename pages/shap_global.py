import streamlit as st

from utils.shap_utils import (
    get_global_shap_data,
    create_summary_bar_plot,
    create_beeswarm_plot,
    create_dependence_plot,
    generate_insights
)

def render():
    st.markdown('<div class="animate-in">', unsafe_allow_html=True)
    
    st.markdown(
        """
        <div class="section-title">SHAP Global Interpretation</div>
        <div class="section-subtitle">
            SHAP (SHapley Additive exPlanations) is a game-theoretic approach to explain the output of any machine learning model. It connects optimal credit allocation with local explanations using the classical Shapley values from game theory.
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        with st.spinner("Loading global SHAP values..."):
            shap_values, X_display = get_global_shap_data()
    except Exception as e:
        st.error(f"Error loading precomputed SHAP data. Did you run precompute_shap.py? Details: {e}")
        st.markdown('</div>', unsafe_allow_html=True)
        return

    col1, col2 = st.columns(2)
    
    with col1:
        # 1. Feature Importance Plot
        st.markdown("### Feature Importance Plot")
        st.markdown(
            "**What this plot is about:** This bar chart shows the global importance of each feature by "
            "averaging the absolute SHAP values across all customers. It identifies which variables the "
            "model relies on most to make its predictions."
        )
        fig_bar = create_summary_bar_plot(shap_values, X_display)
        st.pyplot(fig_bar, clear_figure=True)
        st.markdown(
            "**Insights from this plot:** The model relies most heavily on features like Contract type and Tenure "
            "to determine whether a customer will stay or leave. This ranking allows the business to prioritize "
            "strategic interventions on the top drivers rather than low-impact variables."
        )
        
    with col2:
        # 2. SHAP Beeswarm Plot
        st.markdown("### SHAP Beeswarm Plot")
        st.markdown(
            "**What this plot is about:** This plot displays the distribution of SHAP values for every feature. "
            "Each dot is a customer. The color represents the value of the feature (Red = High, Blue = Low), "
            "and the horizontal axis shows the impact on the churn prediction."
        )
        fig_beeswarm = create_beeswarm_plot(shap_values, X_display)
        st.pyplot(fig_beeswarm, clear_figure=True)
        st.markdown(
            "**Insights from this plot:** High Tenure (shown in red) consistently pushes the churn risk lower, "
            "showing strong customer loyalty, while wide horizontal spreads suggest feature impact varies"
            "significantly across diverse customer groups."
        )

    st.markdown("---")
    
    # ── SHAP Dependence Analysis ──
    st.subheader("SHAP Dependence Analysis")
    
    st.markdown(
        "**What this plot is about:** This plot shows how a feature's values (x-axis) relate to its SHAP "
        "values (y-axis). It helps identify whether the relationship between the feature and the risk of churn "
        "is linear, non-linear, or has specific thresholds."
    )
    
    selected_feature = st.selectbox(
        "Select Feature for Dependence Analysis",
        options=["Contract", "tenure", "MonthlyCharges"],
        index=0
    )
    
    fig_dep = create_dependence_plot(shap_values, selected_feature, X_display)
    st.pyplot(fig_dep, clear_figure=True)
    
    # Dynamic insights based on selected feature
    if selected_feature == "Contract":
        dep_insight = (
            "**Insights from this plot:** Month-to-month contracts strongly push the churn risk higher (positive SHAP values). "
            "In contrast, signing 1-year or 2-year contracts significantly drops the churn risk, indicating that longer-term contracts "
            "are highly effective for retention."
        )
    elif selected_feature == "tenure":
        dep_insight = (
            "**Insights from this plot:** As customer tenure increases, the churn risk decreases significantly. The sharpest drop in risk "
            "occurs within the first 12 to 20 months, showing that early customer relationship management is critical."
        )
    else:  # MonthlyCharges
        dep_insight = (
            "**Insights from this plot:** Higher monthly charges are generally associated with a higher likelihood of churn. "
            "There is a notable increase in churn risk once charges exceed $70-$80, suggesting a key price-sensitivity threshold."
        )
        
    st.markdown(dep_insight)

    st.markdown("---")
    
    # ── Insights Section ──
    st.subheader("Insights Section")
    st.markdown("Automatically summarized findings from the global SHAP analysis.")
    
    insights = generate_insights(shap_values, X_display)
    
    # Display insights inside KPI cards
    cols = st.columns(min(len(insights), 4))
    for idx, (title, desc, icon) in enumerate(insights[:4]):
        with cols[idx]:
            st.markdown(
                f"""
                <div class="insight-card">
                    <div class="insight-icon">{icon}</div>
                    <div class="insight-title">{title}</div>
                    <div class="insight-desc">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown('</div>', unsafe_allow_html=True)
