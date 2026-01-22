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
# PAGE CONFIGURATION
# =========================================
st.set_page_config(
    page_title="Customer Sentiment Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS for a clean dashboard look
st.markdown("""
    <style>
    .reportview-container {
        background: #F0F2F6;
    }
    .main {
        padding-top: 2rem;
    }
    div.stButton > button:first-child {
        width: 100%;
        border-radius: 4px;
        height: 3em;
        background-color: #00416d;
        color: white;
    }
    .metric-container {
        border: 1px solid #e6e9ef;
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: white;
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
    
    # Using 'fast' tokenizers and specific revisions for speed
    sentiment_pipe = pipeline(
        "sentiment-analysis", 
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=-1 # Set to 0 if GPU is available on your deployment
    )
    
    emotion_pipe = pipeline(
        "text-classification", 
        model="j-hartmann/emotion-english-distilroberta-base", 
        return_all_scores=True,
        device=-1
    )
    return stop_words, sentiment_pipe, emotion_pipe

STOP_WORDS, SENTIMENT_MODEL, EMOTION_MODEL = load_resources()

# =========================================
# CORE FUNCTIONS
# =========================================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|[^a-z\s]", "", text)
    tokens = [word for word in text.split() if word not in STOP_WORDS]
    return " ".join(tokens)

def get_analysis(text):
    # Combined function to reduce redundant calls
    truncated_text = text[:512]
    sentiment = SENTIMENT_MODEL(truncated_text)[0]
    emotions = EMOTION_MODEL(truncated_text)[0]
    return {
        'sentiment': sentiment['label'],
        'confidence': sentiment['score'],
        'emotions': {e['label']: e['score'] for e in emotions}
    }

# =========================================
# SIDEBAR NAVIGATION
# =========================================
st.sidebar.title("Analysis Control Panel")
st.sidebar.markdown("Upload your dataset for batch processing or use the main panel for manual entry.")

uploaded_file = st.sidebar.file_uploader("Upload CSV Dataset", type=["csv"])

# =========================================
# MAIN INTERFACE
# =========================================
st.title("Customer Sentiment Intelligence")
st.markdown("Professional NLP Tool for Customer Feedback Analysis")

tab_manual, tab_batch = st.tabs(["Manual Entry Analysis", "Batch Process Dataset"])

# --- MANUAL ENTRY ---
with tab_manual:
    col_input, col_output = st.columns([1, 1], gap="large")
    
    with col_input:
        st.subheader("Input Text")
        user_input = st.text_area(
            "Paste customer review below:",
            height=200,
            placeholder="Enter review text here..."
        )
        analyze_btn = st.button("Run Analysis")

    if analyze_btn and user_input:
        results = get_analysis(user_input)
        
        with col_output:
            st.subheader("Analysis Summary")
            
            # Metric Display
            m_col1, m_col2 = st.columns(2)
            with m_col1:
                st.metric("Primary Sentiment", results['sentiment'])
            with m_col2:
                st.metric("Confidence Score", f"{results['confidence']:.2%}")

            # Emotion Distribution
            emo_df = pd.DataFrame(results['emotions'].items(), columns=['Emotion', 'Intensity'])
            fig = px.bar(
                emo_df, 
                x='Intensity', 
                y='Emotion', 
                orientation='h',
                title="Emotion Probability Distribution",
                template="plotly_white",
                color_discrete_sequence=['#00416d']
            )
            fig.update_layout(margin=dict(l=0, r=0, t=40, b=0), height=300)
            st.plotly_chart(fig, use_container_width=True)

# --- BATCH PROCESS ---
with tab_batch:
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        text_column = st.selectbox("Select Column for Analysis", df.columns)
        
        if st.button("Execute Batch Analysis"):
            with st.spinner("Processing Dataset..."):
                # Fast processing using list comprehension for better performance over .apply()
                raw_texts = df[text_column].astype(str).tolist()
                cleaned_texts = [clean_text(t) for t in raw_texts]
                
                # Perform Sentiment Analysis
                results = SENTIMENT_MODEL(cleaned_texts, truncation=True)
                df['Sentiment'] = [r['label'] for r in results]
                df['Confidence'] = [r['score'] for r in results]

                # Visual Summary
                st.subheader("Aggregate Results")
                c1, c2, c3 = st.columns(3)
                
                total = len(df)
                pos_pct = (df['Sentiment'] == 'POSITIVE').sum() / total
                
                c1.metric("Total Records", total)
                c2.metric("Positive Ratio", f"{pos_pct:.1%}")
                c3.metric("Avg Confidence", f"{df['Confidence'].mean():.2%}")

                st.divider()

                # Graphs
                g1, g2 = st.columns(2)
                with g1:
                    fig_pie = px.pie(
                        df, names='Sentiment', 
                        title="Sentiment Share",
                        color_discrete_map={'POSITIVE':'#00416d', 'NEGATIVE':'#d1d1d1'},
                        hole=0.5
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                with g2:
                    fig_hist = px.histogram(
                        df, x='Confidence', 
                        title="Confidence Distribution",
                        template="plotly_white",
                        color_discrete_sequence=['#00416d']
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)

                st.subheader("Data Preview")
                st.dataframe(df[[text_column, 'Sentiment', 'Confidence']].head(50), use_container_width=True)
    else:
        st.info("Please upload a CSV file via the sidebar to enable batch processing.")

# =========================================
# FOOTER
# =========================================
st.divider()
st.caption("Intelligence System | Version 2.0 | Processed using DistilBERT")
