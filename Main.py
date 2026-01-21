import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from transformers import pipeline

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Movie Sentiment Analysis", page_icon="🎬")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    model = pipeline("sentiment-analysis")
    return model

sentiment_model = load_model()

# ---------------- TITLE ----------------
st.title("🎬 Movie Review Sentiment Analysis")
st.write("Upload your movie review dataset and analyze the sentiment using AI.")

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader("Upload CSV file (must contain a column named 'review')", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if "review" not in df.columns:
        st.error("Dataset must contain a column named 'review'")
        st.stop()

    st.success("Dataset uploaded successfully!")
    st.dataframe(df.head())

    if st.button("Analyze Reviews"):

        with st.spinner("Analyzing reviews..."):

            sentiment_count = {"POSITIVE": 0, "NEGATIVE": 0}
            results = []

            for review in df["review"].astype(str).head(200):
                result = sentiment_model(review)[0]
                sentiment = result["label"]

                sentiment_count[sentiment] += 1

                results.append({
                    "Review": review,
                    "Sentiment": sentiment,
                    "Score": round(result["score"], 3)
                })

        # ---------------- RESULTS ----------------
        st.subheader("Sentiment Distribution")

        fig, ax = plt.subplots()
        ax.bar(sentiment_count.keys(), sentiment_count.values())
        ax.set_ylabel("Count")
        ax.set_xlabel("Sentiment")
        st.pyplot(fig)

        st.subheader("Analyzed Reviews")
        result_df = pd.DataFrame(results)
        st.dataframe(result_df)

# ---------------- FOOTER ----------------
st.markdown("---")
st.write("AI Sentiment Analysis Dashboard using Streamlit & HuggingFace")
