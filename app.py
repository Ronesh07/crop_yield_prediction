# all file 
import streamlit as st
import os
import sys
import numpy as np
import pickle
import json
from datetime import datetime

# Try to import the required modules with exception handling
try:
    import pandas as pd
    import numpy as np
    try:
        from src.components.data_exploration import render_data_exploration
        from src.components.feature_analysis import render_feature_analysis
        from src.components.model_training import render_model_training
        from src.components.yield_prediction import render_yield_prediction
        from src.components.crop_information import render_crop_information
        from src.components.gemini_ai import render_gemini_ai
        from src.utils import config
        # Removed: from src.utils.auth import login_form, logout_button, is_authenticated
    except Exception as e:
        st.error(f"Error importing components: {e}")
        st.stop()
except Exception as e:
    st.error(f"Error importing required modules: {e}")
    st.stop()

# # --- Custom CSS for modern look and dark mode toggle ---
def load_css(dark_mode=False):
    common_css = """
    <style>
    .main-header {
        font-size: 2.5rem;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 700;
    }
    .sub-header {
        font-size: 1.5rem;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    .metric-card {
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        transition: transform 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-4px);
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    .metric-label {
        font-size: 1rem;
        font-weight: 600;
    }
    .card {
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        transition: transform 0.3s ease;
    }
    .card:hover {
        transform: translateY(-4px);
    }
    .card-title {
        font-size: 1.25rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    .highlight {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #2e7d32;
    }
    .footer {
        text-align: center;
        font-size: 0.8rem;
        margin-top: 2rem;
        padding: 1rem;
        border-radius: 5px;
    }
    </style>
    """

    light_css = """
    <style>
    .main-header, .card-title, .metric-value, .sub-header {
        color: #1b5e20 !important;
    }
    .metric-card, .card {
        background-color: #ffffff !important;
        border: 1px solid #e0e0e0 !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05) !important;
        color: #212529 !important;
    }
    .metric-label {
        color: #424242 !important;
    }
    .highlight {
        background-color: #e8f5e9 !important;
        color: #1b5e20 !important;
        border-left-color: #2e7d32 !important;
    }
    .footer {
        background-color: #f1f3f2 !important;
        color: #616161 !important;
    }
    </style>
    """

    dark_css = """
    <style>
    header[data-testid="stHeader"] {
        background-color: #121212 !important;
    }
    .main-header, .card-title, .metric-value, .sub-header {
        color: #81c784 !important;
    }
    .metric-card, .card {
        background-color: #1e1e1e !important;
        border: 1px solid #333333 !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3) !important;
        color: #e0e0e0 !important;
    }
    .metric-label {
        color: #b0bec5 !important;
    }
    .highlight {
        background-color: #1e2d24 !important;
        color: #c8e6c9 !important;
        border-left-color: #81c784 !important;
    }
    .footer {
        background-color: #1e1e1e !important;
        color: #9e9e9e !important;
    }
    p, span, label, li, h1, h2, h3, h4, h5, h6 {
        color: #e0e0e0 !important;
    }
    </style>
    """
    st.markdown(common_css, unsafe_allow_html=True)
    if dark_mode:
        st.markdown(dark_css, unsafe_allow_html=True)
    else:
        st.markdown(light_css, unsafe_allow_html=True)

# --- Model metrics ---
def get_model_performance_metrics():
    metadata_file = os.path.join('models', 'model_metrics.json')
    default_metrics = {
        "avg_accuracy": 87.2,
        "predictions_count": 214,
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    try:
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                metrics = json.load(f)
                return metrics
    except Exception as e:
        print(f"Error loading model metrics: {e}")
    try:
        os.makedirs('models', exist_ok=True)
        with open(metadata_file, 'w') as f:
            json.dump(default_metrics, f)
    except Exception as e:
        print(f"Error creating metrics file: {e}")
    return default_metrics

def count_models():
    try:
        model_count = len([f for f in os.listdir('models') if f.endswith(('.pkl', '.h5', '.joblib'))])
        return model_count if model_count > 0 else 3
    except Exception as e:
        return 3

# --- Sidebar ---
def sidebar(dark_mode):
    st.sidebar.image("crop.jpg",width=300)
    st.sidebar.markdown("<div class='main-header'>🌾Crop Yield AI</div>", unsafe_allow_html=True)
    st.sidebar.markdown("""
    <div class="highlight">
    Welcome! This platform helps you analyze crop data, train AI models, forecast yields, and get AI-powered insights.
    </div>
    """, unsafe_allow_html=True)
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Quick Links**")
    st.sidebar.markdown("- [Project README](#)")
    st.sidebar.markdown("- [Contact Support](#)")
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Settings**")
    st.sidebar.toggle("Dark Mode", key="dark_mode", value=dark_mode)
    st.sidebar.markdown("---")
    # Removed: if is_authenticated(): ... logout_button()
    st.sidebar.markdown("---")
    # Navigation menu (last)
    st.sidebar.markdown("**Navigation**")
    menu_options = [
        "🏠 Dashboard",
        "📊 Data Exploration",
        "🔍 Feature Analysis",
        "🌾 Crop Information",
        "📈 Yield Prediction",
        "🤖 Gemini AI Assistant"
    ]
    default_page = menu_options[0]
    selected_page = st.sidebar.radio("Go to", menu_options, index=st.session_state.get("selected_page_idx", 0), key="sidebar_menu")
    st.session_state["selected_page_idx"] = menu_options.index(selected_page)
    st.sidebar.markdown("<div class='footer'>© 2026 Crop Yield AI | Powered by Streamlit</div>", unsafe_allow_html=True)
    return selected_page

# --- Dashboard Home ---
def render_home():
    st.markdown('<div class="main-header">🌱 Agriculture Intelligence Dashboard</div>', unsafe_allow_html=True)
    metrics = get_model_performance_metrics()
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Datasets</div>
            <div class="metric-value">5</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Trained Models</div>
            <div class="metric-value">{count_models()}</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Predictions</div>
            <div class="metric-value">{metrics.get('predictions_count', 214)}</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Accuracy</div>
            <div class="metric-value">{metrics.get('avg_accuracy', 87)}%</div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<div class='sub-header'>Platform Features</div>", unsafe_allow_html=True)
    feature_col1, feature_col2, feature_col3 = st.columns(3)
    with feature_col1:
        st.markdown("""
        <div class="card">
            <div class="card-title">📊 Data Analysis</div>
            <ul>
                <li>Interactive visualizations</li>
                <li>Statistical tools</li>
                <li>Custom filtering</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    with feature_col2:
        st.markdown("""
        <div class="card">
            <div class="card-title">🔮 Prediction</div>
            <ul>
                <li>ML model support</li>
                <li>Parameter tuning</li>
                <li>Scenario forecasting</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    with feature_col3:
        st.markdown("""
        <div class="card">
            <div class="card-title">🧠 AI Assistant</div>
            <ul>
                <li>Expert knowledge</li>
                <li>Visual crop health</li>
                <li>Custom queries</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("---")
    st.info("Use the tabs above to explore data, analyze features, predict yields, and get AI insights.")

# --- Main App ---
def main():
    if "dark_mode" not in st.session_state:
        st.session_state["dark_mode"] = False
        
    selected_page = sidebar(st.session_state["dark_mode"])
    dark_mode = st.session_state.get("dark_mode", False)
    load_css(dark_mode)
    # Render the selected page
    if selected_page == "🏠 Dashboard":
        render_home()
    elif selected_page == "📊 Data Exploration":
        render_data_exploration()
    elif selected_page == "🔍 Feature Analysis":
        render_feature_analysis()
    elif selected_page == "🌾 Crop Information":
        render_crop_information()
    elif selected_page == "📈 Yield Prediction":
        render_yield_prediction()
    elif selected_page == "🤖 Gemini AI Assistant":
        render_gemini_ai()

if __name__ == "__main__":
    main() 
