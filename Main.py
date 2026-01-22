import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from transformers import pipeline
import plotly.express as px

# =========================================
# PAGE CONFIG
# =========================================
st.set_page_config(
    page_title="Customer Sentiment AI",
    page_icon="📊",
    layout="wide"
)

# Custom CSS untuk mencantikkan UI
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_html=True)

# =========================================
# NLTK SETUP & MODELS
# =========================================
@st.cache_resource
def download_nltk():
    nltk.download('stopwords')
    return set(stopwords.words('english'))

stop_words = download_nltk()

@st.cache_resource
def load_models():
    sentiment_pipe = pipeline(
        "sentiment-analysis", 
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )
    emotion_pipe = pipeline(
        "text-classification", 
        model="j-hartmann/emotion-english-distilroberta-base", 
        return_all_scores=True
    )
    return sentiment_pipe, emotion_pipe

sentiment_model, emotion_model = load_models()

# =========================================
# HELPER FUNCTIONS
# =========================================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|[^a-z\s]", "", text)
    text = " ".join(word for word in text.split() if word not in stop_words)
    return text

def get_sentiment(text):
    result = sentiment_model(text[:512])[0]
    return result['label'], result['score']

def get_emotions(text):
    emotions = emotion_model(text[:512])[0]
    return {e['label']: e['score'] for e in emotions}

# =========================================
# SIDEBAR
# =========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/9438/9438567.png", width=100)
    st.title("Settings")
    uploaded_file = st.file_uploader("Upload Customer Reviews (CSV)", type=["csv"])
    
    st.info("Aplikasi ini menggunakan AI untuk menganalisis sentimen dan emosi pelanggan secara mendalam.")

# =========================================
# MAIN CONTENT
# =========================================
st.title("📊 Customer Sentiment Analysis Dashboard")
st.markdown("Analisis maklum balas pelanggan dengan teknologi *Deep Learning*.")

tab1, tab2 = st.tabs(["🔍 Analisis Individu", "📂 Analisis Batch (CSV)"])

# --- TAB 1: INDIVIDUAL ANALYSIS ---
with tab1:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Write a Review")
        user_input = st.text_area(
            "Masukkan komen pelanggan di sini:",
            placeholder="Contoh: The product quality is amazing and delivery was fast!",
            height=150
        )
        analyze_btn = st.button("Analyze Sentiment", type="primary")

    if analyze_btn and user_input:
        sentiment, score = get_sentiment(user_input)
        emotions = get_emotions(user_input)
        
        with col2:
            st.subheader("Results")
            
            # Metric Display
            m1, m2 = st.columns(2)
            color = "normal" if sentiment == "POSITIVE" else "inverse"
            m1.metric("Sentiment", sentiment)
            m2.metric("Confidence", f"{score:.2%}")

            # Emotion Chart
            emo_df = pd.DataFrame(emotions.items(), columns=['Emotion', 'Score'])
            fig = px.bar(emo_df, x='Score', y='Emotion', orientation='h', 
                         color='Score', color_continuous_scale='RdBu')
            fig.update_layout(height=300, margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig, use_container_width=True)

# --- TAB 2: BATCH ANALYSIS ---
with tab2:
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        st.success(f"Berjaya memuat naik {len(df)} data.")
        text_col = st.selectbox("Pilih ruangan (column) teks:", df.columns)
        
        if st.button("Start Batch Analysis"):
            with st.spinner("Sedang memproses... Sila tunggu."):
                # Processing
                df['Cleaned'] = df[text_col].apply(clean_text)
                
                # Apply Sentiment
                results = df['Cleaned'].apply(lambda x: pd.Series(get_sentiment(x)))
                df['Sentiment'] = results[0]
                df['Confidence'] = results[1]
                
                # Stats
                pos_count = len(df[df['Sentiment'] == 'POSITIVE'])
                neg_count = len(df[df['Sentiment'] == 'NEGATIVE'])
                
                # Visuals
                c1, c2, c3 = st.columns(3)
                c1.metric("Total Reviews", len(df))
                c2.metric("Positive", pos_count, delta=f"{(pos_count/len(df)):.1%}")
                c3.metric("Negative", neg_count, delta=f"-{(neg_count/len(df)):.1%}", delta_color="inverse")
                
                st.divider()
                
                col_chart1, col_chart2 = st.columns(2)
                
                with col_chart1:
                    st.write("### Sentiment Distribution")
                    fig_pie = px.pie(df, names='Sentiment', hole=0.4, 
                                     color_discrete_map={'POSITIVE':'#2ecc71', 'NEGATIVE':'#e74c3c'})
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                with col_chart2:
                    st.write("### Confidence Score Analysis")
                    fig_hist = px.histogram(df, x='Confidence', color='Sentiment', nbins=20)
                    st.plotly_chart(fig_hist, use_container_width=True)

                st.write("### Raw Data Preview")
                st.dataframe(df[[text_col, 'Sentiment', 'Confidence']].head(100), use_container_width=True)
    else:
        st.warning("Sila muat naik fail CSV di bahagian Sidebar untuk menggunakan fungsi ini.")

# =========================================
# FOOTER
# =========================================
st.markdown("---")
st.caption("Dikuasakan oleh HuggingFace Transformers & Streamlit")
