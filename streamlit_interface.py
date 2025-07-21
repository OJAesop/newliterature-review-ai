# Literature Review AI - Week 5-6: Web Interface using Streamlit (Free)
# Run with: streamlit run app.py

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import io
from datetime import datetime
import sys
import os

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our modules
try:
    from lit_review_pipeline import LiteraturePipeline  # The data pipeline
    from ai_processing_module import LiteratureAI      # The AI processor
except ImportError:
    st.error("Please ensure lit_review_pipeline.py and ai_processing_module.py are in the same directory")
    st.stop()

# Configure Streamlit page
st.set_page_config(
    page_title="Literature Review AI",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'papers_df' not in st.session_state:
    st.session_state.papers_df = pd.DataFrame()
if 'ai_processor' not in st.session_state:
    st.session_state.ai_processor = None
if 'search_history' not in st.session_state:
    st.session_state.search_history = []

# Sidebar
with st.sidebar:
    st.title("📚 Literature Review AI")
    st.markdown("---")
    
    # Navigation
    page = st.selectbox(
        "Choose a page",
        ["🔍 Search Papers", "🤖 AI Analysis", "📊 Visualizations", "❓ Q&A Assistant", "📈 Summary Report"]