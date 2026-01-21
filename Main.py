import streamlit as st
import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
from transformers import pipeline

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

# ---------------- LOAD MODELS (CPU ONLY) ----------------
@st.cache_resource
def load_models():
    sentiment_model = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment",
        device=-1   # Force CPU
    )

    emotion_model = pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        return_all_scores=True,
        device=-1   # Force CPU
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

# ---------------- SIDEBAR ----------------
st.sidebar.header("⚙ Experiment Settings")
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

        # Sentiment Pie Chart
        fig1, ax1 = plt.subplots()
        ax1.pie(
            sentiment_count.values(),
            labels=sentiment_count.keys(),
            autopct='%1.1f%%',
            startangle=90
        )
        ax1.set_title("Sentiment Distribution")
        col1.pyplot(fig1)

        # Emotion Bar Chart
        fig2, ax2 = plt.subplots()
        ax2.bar(emotion_count.keys(), emotion_count.values())
        ax2.set_title("Emotion Distribution")
        ax2.set_ylabel("Count")
        ax2.set_xlabel("Emotion")
        col2.pyplot(fig2)

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
