import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from config.settings import MODEL_METRICS, PLOTLY_LAYOUT, CHART_COLORS

def render():
    st.markdown('<div class="animate-in">', unsafe_allow_html=True)
    
    st.markdown(
        """
        <div class="section-title">Model Comparison</div>
        <div class="section-subtitle">
            Compare all trained models and their performance metrics.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Metrics Table ──
    st.subheader("Metrics Table")
    
    # Styled dataframe
    st.dataframe(
        MODEL_METRICS.style.background_gradient(cmap="Blues", subset=["Accuracy", "F1 Score", "ROC-AUC"])
                          .format(precision=4),
        use_container_width=True
    )

    st.markdown("---")

    # ── Comparison Visualizations ──
    st.subheader("Comparison Visualizations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Accuracy Comparison
        fig_acc = px.bar(
            MODEL_METRICS, 
            x="Model", 
            y="Accuracy", 
            title="Accuracy Comparison",
            color="Model",
            color_discrete_sequence=CHART_COLORS
        )
        fig_acc.update_layout(**PLOTLY_LAYOUT)
        fig_acc.update_layout(showlegend=False)
        st.plotly_chart(fig_acc, use_container_width=True)
        
        # ROC-AUC Comparison
        fig_roc = px.bar(
            MODEL_METRICS, 
            x="Model", 
            y="ROC-AUC", 
            title="ROC-AUC Comparison",
            color="Model",
            color_discrete_sequence=CHART_COLORS
        )
        fig_roc.update_layout(**PLOTLY_LAYOUT)
        fig_roc.update_layout(showlegend=False)
        st.plotly_chart(fig_roc, use_container_width=True)

    with col2:
        # F1 Score Comparison
        fig_f1 = px.bar(
            MODEL_METRICS, 
            x="Model", 
            y="F1 Score", 
            title="F1 Score Comparison",
            color="Model",
            color_discrete_sequence=CHART_COLORS
        )
        fig_f1.update_layout(**PLOTLY_LAYOUT)
        fig_f1.update_layout(showlegend=False)
        st.plotly_chart(fig_f1, use_container_width=True)
        
        # Precision vs Recall
        fig_pr = go.Figure()
        fig_pr.add_trace(go.Bar(
            x=MODEL_METRICS["Model"], 
            y=MODEL_METRICS["Precision"], 
            name="Precision", 
            marker_color=CHART_COLORS[0]
        ))
        fig_pr.add_trace(go.Bar(
            x=MODEL_METRICS["Model"], 
            y=MODEL_METRICS["Recall"], 
            name="Recall", 
            marker_color=CHART_COLORS[1]
        ))
        fig_pr.update_layout(
            barmode='group', 
            title="Precision vs Recall",
            **PLOTLY_LAYOUT
        )
        st.plotly_chart(fig_pr, use_container_width=True)

    st.markdown("---")
    
    # ── Best Model Section ──
    st.subheader("Best Model Section")
    
    best_model_name = "XGBoost"
    best_model_row = MODEL_METRICS[MODEL_METRICS["Model"] == best_model_name].iloc[0]
    
    st.markdown(
        f"""
        <div class="best-model-card">
            <h3 style="margin-top:0; color:var(--text-primary);">🏆 {best_model_name}</h3>
            <p style="color:var(--text-secondary); margin-bottom:1rem;">
                The selected model achieved the highest overall predictive performance and was chosen for deployment.
            </p>
            <div style="display:flex; justify-content:center; gap:2rem;">
                <div>
                    <div style="font-size:0.8rem; color:var(--text-muted); text-transform:uppercase;">Accuracy</div>
                    <div style="font-size:1.5rem; font-weight:bold; color:var(--text-primary);">{best_model_row["Accuracy"]:.4f}</div>
                </div>
                <div>
                    <div style="font-size:0.8rem; color:var(--text-muted); text-transform:uppercase;">F1 Score</div>
                    <div style="font-size:1.5rem; font-weight:bold; color:var(--text-primary);">{best_model_row["F1 Score"]:.4f}</div>
                </div>
                <div>
                    <div style="font-size:0.8rem; color:var(--text-muted); text-transform:uppercase;">ROC-AUC</div>
                    <div style="font-size:1.5rem; font-weight:bold; color:var(--text-primary);">{best_model_row["ROC-AUC"]:.4f}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('</div>', unsafe_allow_html=True)
