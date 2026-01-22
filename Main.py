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
# PAGE CONFIGURATION & CUSTOM CSS
# =========================================
st.set_page_config(
    page_title="Customer Sentiment Intelligence",
    layout="wide"
)

# Professional CSS - Diperkemas untuk tulisan HITAM dalam metrik
st.markdown("""
    <style>
    /* Mengubah warna latar belakang aplikasi (gelap) */
    .stApp {
        background-color: #0e1117;
    }
    
    /* Mengubah kotak metrik menjadi putih dengan tulisan HITAM */
    [data-testid="stMetricValue"], [data-testid="stMetricLabel"] {
        color: #000000 !important;
    }
    
    [data-testid="metric-container"] {
        background-color: #ffffff;
        border: 1px solid #e6e9ef;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
    }
    
    /* Warna butang profesional */
    div.stButton > button:first-child {
        background-color: #00416d;
        color: white;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

# =========================================
# DATA & MODEL LOADING (OPTIMIZED)
# =========================================
@st.cache_resource
def load_resources():
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
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
# MAIN INTERFACE
# =========================================
st.title("Customer Sentiment Intelligence")

tab_manual, tab_batch = st.tabs(["Individual Analysis", "Batch Dataset Processing"])

with tab_manual:
    col_in, col_out = st.columns(2, gap="large")
    with col_in:
        user_input = st.text_area("Review Text", height=150, placeholder="Enter review...")
        run_single = st.button("Analyze Review")
    
    if run_single and user_input:
        res = SENTIMENT_MODEL(user_input[:512])[0]
        label = process_sentiment_label(res['label'])
        with col_out:
            st.metric("Sentiment", label)
            st.metric("Confidence Score", f"{res['score']:.2%}")

with tab_batch:
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        
        st.subheader("Configuration")
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            id_col = st.selectbox("Select Customer ID Column", df.columns)
        with col_c2:
            text_col = st.selectbox("Select Review Text Column", df.columns)
            
        if st.button("Execute Batch Analysis"):
            with st.spinner("Processing..."):
                texts = df[text_col].astype(str).tolist()
                results = SENTIMENT_MODEL(texts, truncation=True, batch_size=8)
                
                df['Sentiment'] = [process_sentiment_label(r['label']) for r in results]
                df['Confidence'] = [r['score'] for r in results]
                
                # METRICS SECTION - Tulisan akan berwarna HITAM
                st.divider()
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Total Reviews", len(df))
                m2.metric("Positive", len(df[df['Sentiment'] == 'Positive']))
                m3.metric("Neutral", len(df[df['Sentiment'] == 'Neutral']))
                m4.metric("Negative", len(df[df['Sentiment'] == 'Negative']))
                
                st.subheader("Sentiment Distribution")
                dist_df = df['Sentiment'].value_counts().reset_index()
                dist_df.columns = ['Label', 'Count']
                fig = px.bar(dist_df, x='Label', y='Count', color='Label',
                             color_discrete_map={'Positive': '#2ecc71', 'Neutral': '#95a5a6', 'Negative': '#e74c3c'})
                st.plotly_chart(fig, use_container_width=True)

                st.subheader("Analysis Results")
                st.dataframe(df[[id_col, text_col, 'Sentiment', 'Confidence']], use_container_width=True, hide_index=True)
    else:
        st.info("Upload a CSV file in the sidebar to begin batch analysis.")
