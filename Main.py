import streamlit as st
import pandas as pd
import plotly.express as px
from transformers import pipeline
import random
import numpy as np
import torch

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Movie Sentiment Dashboard",
    page_icon="🎬",
    layout="wide"
)

# ---------------- RANDOM SEED ----------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_models():
    sentiment_model = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment"
    )

    emotion_model = pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        return_all_scores=True
    )
    return sentiment_model, emotion_model

sentiment_model, emotion_model = load_models()

# ---------------- UI HEADER ----------------
st.markdown("""
<h1 style='text-align:center; background:linear-gradient(90deg,#38bdf8,#818cf8,#f472b6);
-webkit-background-clip:text; color:transparent;'>
🎬 AI Movie Review Sentiment Dashboard
</h1>
<p style='text-align:center; color:#94a3b8;'>
Upload your dataset and analyze movie review sentiments using Deep Learning
</p>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR SETTINGS ----------------
st.sidebar.header("⚙ Experiment Settings")
seed_input = st.sidebar.number_input("Random Seed", value=42, step=1)

random.seed(seed_input)
np.random.seed(seed_input)
torch.manual_seed(seed_input)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed_input)

max_rows = st.sidebar.slider("Max Reviews to Analyze", 10, 500, 200)

# ---------------- FILE UPLOAD ----------------
st.markdown("## 📁 Upload Movie Review Dataset (CSV)")
uploaded_file = st.file_uploader("Upload CSV file with a 'review' column", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if "review" not in df.columns:
        st.error("❌ Dataset must contain a column named 'review'")
        st.stop()

    st.success("✅ Dataset uploaded successfully!")
    st.dataframe(df.head(), use_container_width=True)

    if st.button("Analyze Dataset"):

        with st.spinner("🤖 AI is analyzing reviews..."):
            sentiment_count = {"Positive": 0, "Neutral": 0, "Negative": 0}
            emotion_count = {"joy": 0, "anger": 0, "sadness": 0, "fear": 0, "surprise": 0}

            analyzed_reviews = []

            for review in df["review"].astype(str).head(max_rows):
                sentiment_result = sentiment_model(review)[0]
                emotion_result = emotion_model(review)[0]

                # Sentiment
                label = sentiment_result["label"].lower()
                if "positive" in label:
                    sentiment_count["Positive"] += 1
                elif "neutral" in label:
                    sentiment_count["Neutral"] += 1
                else:
                    sentiment_count["Negative"] += 1

                # Emotion
                top_emotion = max(emotion_result, key=lambda x: x["score"])["label"].lower()
                if top_emotion in emotion_count:
                    emotion_count[top_emotion] += 1

                analyzed_reviews.append({
                    "Review": review,
                    "Sentiment": sentiment_result["label"],
                    "Emotion": top_emotion.capitalize()
                })

        # ---------------- DASHBOARD ----------------
        st.markdown("## 📊 Analysis Results")

        col1, col2 = st.columns(2)

        # Sentiment Chart
        sentiment_df = pd.DataFrame({
            "Sentiment": sentiment_count.keys(),
            "Count": sentiment_count.values()
        })

        fig_sentiment = px.pie(
            sentiment_df,
            names="Sentiment",
            values="Count",
            title="Sentiment Distribution",
            hole=0.4
        )

        col1.plotly_chart(fig_sentiment, use_container_width=True)

        # Emotion Chart
        emotion_df = pd.DataFrame({
            "Emotion": emotion_count.keys(),
            "Count": emotion_count.values()
        })

        fig_emotion = px.bar(
            emotion_df,
            x="Emotion",
            y="Count",
            title="Emotion Distribution"
        )

        col2.plotly_chart(fig_emotion, use_container_width=True)

        # ---------------- REVIEW TABLE ----------------
        st.markdown("## 📝 Analyzed Reviews")
        result_df = pd.DataFrame(analyzed_reviews)
        st.dataframe(result_df, use_container_width=True)

# ---------------- FOOTER ----------------
st.markdown("""
<hr>
<p style='text-align:center; color:#64748b;'>
AI Sentiment Dashboard | Powered by HuggingFace & Streamlit
</p>
""", unsafe_allow_html=True)
