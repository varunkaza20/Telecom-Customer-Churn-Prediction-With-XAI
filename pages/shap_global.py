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
            Provide explainability of the entire model. Understand the driving factors behind customer churn.
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

    st.markdown("---")
    
    # ── Feature Importance Section ──
    st.subheader("Feature Importance Section")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### SHAP Summary Bar Plot")
        st.markdown("Shows top influential features and global importance ranking.")
        fig_bar = create_summary_bar_plot(shap_values, X_display)
        st.pyplot(fig_bar, clear_figure=True)
        
    with col2:
        st.markdown("##### SHAP Beeswarm Plot")
        st.markdown("Shows feature impact distribution and positive/negative contributions.")
        fig_beeswarm = create_beeswarm_plot(shap_values, X_display)
        st.pyplot(fig_beeswarm, clear_figure=True)

    st.markdown("---")
    
    # ── SHAP Dependence Analysis ──
    st.subheader("SHAP Dependence Analysis")
    
    selected_feature = st.selectbox(
        "Select Feature",
        options=["Contract", "tenure", "MonthlyCharges"],
        index=0
    )
    
    st.markdown(f"Displaying SHAP dependence plot for **{selected_feature}**.")
    fig_dep = create_dependence_plot(shap_values, selected_feature, X_display)
    st.pyplot(fig_dep, clear_figure=True)

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
