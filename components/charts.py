"""
components/charts.py
Reusable chart components using Plotly.
"""

import plotly.graph_objects as go
from config.settings import PLOTLY_LAYOUT


def create_gauge(probability: float, risk_color: str):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number=dict(
                suffix="%",
                font=dict(size=48, family="JetBrains Mono, monospace", color="#f1f5f9"),
            ),
            title=dict(
                text="Churn Probability",
                font=dict(size=16, color="#94a3b8"),
            ),
            gauge=dict(
                axis=dict(range=[0, 100], tickwidth=1, tickcolor="#334155",
                          tickfont=dict(color="#64748b")),
                bar=dict(color=risk_color, thickness=0.25),
                bgcolor="rgba(255,255,255,0.03)",
                borderwidth=0,
                steps=[
                    dict(range=[0, 35],  color="rgba(16,185,129,0.12)"),
                    dict(range=[35, 65], color="rgba(245,158,11,0.12)"),
                    dict(range=[65, 100], color="rgba(244,63,94,0.12)"),
                ],
                threshold=dict(
                    line=dict(color=risk_color, width=3),
                    thickness=0.8,
                    value=probability * 100,
                ),
            ),
        )
    )
    
    # Filter out margin from PLOTLY_LAYOUT to prevent conflicts
    layout_kwargs = {k: v for k, v in PLOTLY_LAYOUT.items() if k != "margin"}
    
    fig.update_layout(
        height=280,
        **layout_kwargs,
        margin=dict(l=30, r=30, t=60, b=20),
    )
    return fig
