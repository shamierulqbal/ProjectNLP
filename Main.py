import streamlit as st
import pandas as pd
import re
import nltk
import torch
import numpy as np
import random
from nltk.corpus import stopwords
from transformers import pipeline
import plotly.express as px
import plotly.graph_objects as go

# =========================================
# GLOBAL SETTINGS & REPRODUCIBILITY
# =========================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# =========================================
# PAGE CONFIGURATION & PREMIUM UI STYLING
# =========================================
st.set_page_config(
    page_title="Sentiment Intelligence Analytics Dashboard",
    layout="wide"
)

# Advanced CSS for modern Dashboard UI
st.markdown("""
    <style>
    /* Main Background and Sidebar */
    .stApp {
        background-color: #0d1117;
        color: #c9d1d9;
    }
    section[data-testid="stSidebar"] {
        background-color: #161b22 !important;
        border-right: 1px solid #30363d;
    }

    /* Metric Card Styling - Mimicking the Reference Image */
    div[data-testid="metric-container"] {
        background-color: #161b22;
        border: 1px solid #30363d;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.5);
        transition: transform 0.2s ease-in-out;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-5px);
        border-color: #58a6ff;
    }

    /* Typography for Metrics */
    [data-testid="stMetricValue"] {
        color: #58a6ff !important;
        font-size: 2.5rem !important;
        font-weight: 800 !important;
    }
    [data-testid="stMetricLabel"] p {
        color: #8b949e !important;
        font-size: 0.9rem !important;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }

    /* Professional Button Styling */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #1f6feb 0%, #1158c7 100%);
        color: white;
        border-radius: 8px;
        border: none;
        font-weight: 600;
        padding: 0.6rem 2rem;
        width: 100%;
    }

    /* Tabs Customization */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        color: #8b949e;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f6feb !important;
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)

# =========================================
# MODELS & UTILS
# =========================================
@st.cache_resource
def load_engine():
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    sentiment_pipe = pipeline(
        "sentiment-analysis", 
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        device=-1
    )
    return stop_words, sentiment_pipe

STOP_WORDS, SENTIMENT_MODEL = load_engine()

def process_label(label):
    label = label.lower()
    if 'positive' in label: return 'Positive'
    if 'negative' in label: return 'Negative'
    return 'Neutral'

# =========================================
# MAIN DASHBOARD INTERFACE
# =========================================
st.title("Analytics Intelligence Dashboard")
st.markdown("---")

tab_manual, tab_batch = st.tabs(["Real-time Analysis", "Dataset Intelligence"])

# --- INDIVIDUAL ANALYSIS ---
with tab_manual:
    col_input, col_viz = st.columns([1, 1], gap="large")
    
    with col_input:
        st.subheader("Data Input")
        user_input = st.text_area("Customer Content", height=200, placeholder="Input text for neural processing...")
        if st.button("Initialize Analysis"):
            if user_input:
                res = SENTIMENT_MODEL(user_input[:512])[0]
                label = process_label(res['label'])
                score = res['score']
                
                with col_viz:
                    st.subheader("Neural Outcome")
                    c1, c2 = st.columns(2)
                    c1.metric("Classification", label)
                    c2.metric("Confidence", f"{score:.2%}")
                    
                    # Modern Gauge Chart
                    fig_gauge = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = score * 100,
                        title = {'text': "Certainty Score", 'font': {'color': "#8b949e"}},
                        gauge = {
                            'axis': {'range': [None, 100], 'tickcolor': "#8b949e"},
                            'bar': {'color': "#1f6feb"},
                            'bgcolor': "#161b22",
                            'borderwidth': 2,
                            'bordercolor': "#30363d",
                            'steps': [
                                {'range': [0, 50], 'color': '#21262d'},
                                {'range': [50, 100], 'color': '#30363d'}]
                        }
                    ))
                    fig_gauge.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': "#c9d1d9"}, height=300)
                    st.plotly_chart(fig_gauge, use_container_width=True)

# --- BATCH PROCESS ---
with tab_batch:
    uploaded_file = st.sidebar.file_uploader("Upload Core Asset (CSV)", type=["csv"])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        
        st.subheader("Asset Configuration")
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            id_col = st.selectbox("Identifier Column", df.columns)
        with col_c2:
            text_col = st.selectbox("Content Column", df.columns)
            
        if st.button("Process Intelligence Pipeline"):
            with st.spinner("Processing Large-scale Asset..."):
                texts = df[text_col].astype(str).tolist()
                results = SENTIMENT_MODEL(texts, truncation=True, batch_size=8)
                
                df['Sentiment'] = [process_label(r['label']) for r in results]
                df['Confidence'] = [r['score'] for r in results]
                
                # --- METRIC CARDS SECTION ---
                st.markdown("### Executive Summary")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Total Processing", len(df))
                m2.metric("Positive Volume", len(df[df['Sentiment'] == 'Positive']))
                m3.metric("Neutral Volume", len(df[df['Sentiment'] == 'Neutral']))
                m4.metric("Negative Volume", len(df[df['Sentiment'] == 'Negative']))
                
                # --- VISUAL CHARTS SECTION ---
                st.markdown("---")
                chart_left, chart_right = st.columns(2)
                
                with chart_left:
                    st.subheader("Sentiment Distribution")
                    fig_pie = px.pie(
                        df, names='Sentiment', 
                        hole=0.6,
                        color='Sentiment',
                        color_discrete_map={'Positive': '#238636', 'Neutral': '#8b949e', 'Negative': '#da3633'}
                    )
                    fig_pie.update_layout(paper_bgcolor='rgba(0,0,0,0)', legend_font_color="#c9d1d9")
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                with chart_right:
                    st.subheader("Confidence Spread")
                    fig_hist = px.histogram(
                        df, x='Confidence', color='Sentiment',
                        marginal="box", barmode="overlay",
                        color_discrete_map={'Positive': '#238636', 'Neutral': '#8b949e', 'Negative': '#da3633'}
                    )
                    fig_hist.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="#c9d1d9")
                    st.plotly_chart(fig_hist, use_container_width=True)

                # --- DATA TABLE ---
                st.subheader("Detailed Intelligent Ledger")
                st.dataframe(
                    df[[id_col, text_col, 'Sentiment', 'Confidence']], 
                    use_container_width=True, 
                    hide_index=True
                )
                
                # Export Button
                csv = df[[id_col, text_col, 'Sentiment', 'Confidence']].to_csv(index=False).encode('utf-8')
                st.download_button("Export Processed Ledger", csv, "intelligence_report.csv", "text/csv")
    else:
        st.info("System Ready. Please upload a CSV dataset via the Control Panel to begin.")

# =========================================
# FOOTER
# =========================================
st.markdown("<br><br><br>", unsafe_allow_html=True)
st.caption("Intelligence Core 3.0 | Secure Analysis Mode | 2026 Enterprise Edition")
