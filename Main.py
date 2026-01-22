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
# PAGE CONFIGURATION & ADVANCED UI STYLING
# =========================================
st.set_page_config(
    page_title="Customer Sentiment Intelligence",
    layout="wide"
)

# Custom CSS for Deep Blue Professional Theme
st.markdown("""
    <style>
    /* Main App Background */
    .stApp {
        background-color: #0a192f;
        color: #e6f1ff;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #112240 !important;
        border-right: 1px solid #233554;
    }

    /* Metric Card Styling - Dark Blue Glassmorphism */
    [data-testid="metric-container"] {
        background-color: #112240;
        border: 1px solid #233554;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }

    /* Metric Text Colors */
    [data-testid="stMetricValue"] {
        color: #64ffda !important;
        font-family: 'Inter', sans-serif;
        font-weight: 700;
    }

    [data-testid="stMetricLabel"] p {
        color: #8892b0 !important;
        font-size: 14px !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        background-color: transparent;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: transparent;
        border-radius: 4px;
        color: #8892b0;
        font-weight: 600;
    }

    .stTabs [aria-selected="true"] {
        color: #64ffda !important;
        border-bottom-color: #64ffda !important;
    }

    /* Button Styling */
    div.stButton > button:first-child {
        background-color: #64ffda;
        color: #0a192f;
        font-weight: bold;
        border: none;
        padding: 10px 24px;
        border-radius: 4px;
        transition: all 0.3s ease;
    }

    div.stButton > button:hover {
        background-color: #4cd3b5;
        box-shadow: 0 0 15px rgba(100, 255, 218, 0.4);
    }

    /* Dataframe & Table Adjustments */
    .stDataFrame {
        border: 1px solid #233554;
        border-radius: 8px;
    }
    
    /* Input field styling */
    .stTextArea textarea {
        background-color: #112240 !important;
        color: #ccd6f6 !important;
        border: 1px solid #233554 !important;
    }

    hr {
        border-top: 1px solid #233554;
    }
    </style>
    """, unsafe_allow_html=True)

# =========================================
# DATA & MODEL LOADING (CACHED)
# =========================================
@st.cache_resource
def load_resources():
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    
    # Model: RoBERTa for Neutral, Positive, and Negative detection
    sentiment_pipe = pipeline(
        "sentiment-analysis", 
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        device=-1
    )
    return stop_words, sentiment_pipe

STOP_WORDS, SENTIMENT_MODEL = load_resources()

# =========================================
# CORE FUNCTIONS
# =========================================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|[^a-z\s]", "", text)
    tokens = [word for word in text.split() if word not in STOP_WORDS]
    return " ".join(tokens)

def process_sentiment_label(label):
    label = label.lower()
    if 'positive' in label: return 'Positive'
    if 'negative' in label: return 'Negative'
    return 'Neutral'

# =========================================
# MAIN DASHBOARD INTERFACE
# =========================================
st.title("Sentiment Intelligence Systems")
st.markdown("<p style='color: #8892b0;'>Advanced Neural Network Analysis for Customer Feedback</p>", unsafe_allow_html=True)

tab_manual, tab_batch = st.tabs(["Individual Analysis", "Batch Dataset Processing"])

# --- TAB 1: INDIVIDUAL ANALYSIS ---
with tab_manual:
    col_in, col_out = st.columns(2, gap="large")
    with col_in:
        st.subheader("Input Stream")
        user_input = st.text_area("Review Content", height=180, placeholder="Type or paste customer feedback here...")
        run_single = st.button("Run Intelligence Check")
    
    if run_single and user_input:
        res = SENTIMENT_MODEL(user_input[:512])[0]
        label = process_sentiment_label(res['label'])
        with col_out:
            st.subheader("Classification Outcome")
            st.metric("Detected Sentiment", label)
            st.metric("Model Confidence", f"{res['score']:.2%}")

# --- TAB 2: BATCH PROCESS ---
with tab_batch:
    uploaded_file = st.sidebar.file_uploader("Upload CSV Asset", type=["csv"])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        
        st.subheader("Schema Mapping")
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            id_col = st.selectbox("Customer Identifier Column", df.columns)
        with col_c2:
            text_col = st.selectbox("Review Text Column", df.columns)
            
        if st.button("Execute Pipeline"):
            with st.spinner("Analyzing high-volume data..."):
                # Optimize speed using list processing
                texts = df[text_col].astype(str).tolist()
                results = SENTIMENT_MODEL(texts, truncation=True, batch_size=8)
                
                df['Sentiment'] = [process_sentiment_label(r['label']) for r in results]
                df['Confidence'] = [r['score'] for r in results]
                
                # Visual Metrics Section
                st.divider()
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Total Volume", len(df))
                m2.metric("Positive Hits", len(df[df['Sentiment'] == 'Positive']))
                m3.metric("Neutral Hits", len(df[df['Sentiment'] == 'Neutral']))
                m4.metric("Negative Hits", len(df[df['Sentiment'] == 'Negative']))
                
                # Visual Analytics
                st.subheader("Sentiment Distribution Profile")
                dist_df = df['Sentiment'].value_counts().reset_index()
                dist_df.columns = ['Label', 'Count']
                
                fig = px.bar(
                    dist_df, x='Label', y='Count', color='Label',
                    color_discrete_map={'Positive': '#64ffda', 'Neutral': '#8892b0', 'Negative': '#f07178'},
                    template="plotly_dark"
                )
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='#ccd6f6'
                )
                st.plotly_chart(fig, use_container_width=True)

                st.subheader("Aggregated Results Table")
                # Structure: ID, Text, Sentiment, Confidence
                result_display = df[[id_col, text_col, 'Sentiment', 'Confidence']]
                st.dataframe(result_display, use_container_width=True, hide_index=True)
                
                # Export functionality
                csv = result_display.to_csv(index=False).encode('utf-8')
                st.download_button("Export Processed Dataset", csv, "processed_intelligence.csv", "text/csv")
    else:
        st.info("Awaiting CSV asset upload via the control panel.")

# =========================================
# FOOTER
# =========================================
st.markdown("<br><hr>", unsafe_allow_html=True)
st.caption("Intelligence Core: RoBERTa Base | Seed: 42 | Response Mode: High-Speed Batching")
