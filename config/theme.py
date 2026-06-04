"""
config/theme.py
CSS theme loader for the Streamlit application.
"""

import os
import streamlit as st


def load_css():
    """Inject the custom CSS theme into the Streamlit app."""
    css_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "assets", "style.css"
    )
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
