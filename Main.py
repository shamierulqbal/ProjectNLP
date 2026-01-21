import streamlit as st
import pandas as pd
import plotly.express as px
from transformers import pipeline

# --- AI MODEL SETUP ---
@st.cache_resource
def load_sentiment_model():
    # DistilBERT is highly accurate for Pang & Lee's movie review style
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

classifier = load_sentiment_model()

# --- UI CONFIGURATION ---
st.set_page_config(page_title="Pang & Lee Movie Sentiment", layout="wide")
st.title("🎬 Movie Review Sentiment Analyzer")
st.markdown("### NLP Project: Analyzing the Pang & Lee Polarity Dataset")

# --- FILE UPLOAD ---
uploaded_file = st.file_uploader("Upload your movie_reviews_dataset.csv", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    col1, col2 = st.columns(2)
    with col1:
        text_col = st.selectbox("Select Review Text Column", df.columns)
    with col2:
        label_col = st.selectbox("Select Actual Label Column", df.columns)

    if st.button("🚀 Run Sentiment Analysis"):
        with st.spinner("Analyzing reviews using Transformers..."):
            
            # Prediction function with character limit to avoid model errors
            def get_ai_result(text):
                res = classifier(str(text)[:512])[0]
                return res['label']

            df['AI_Prediction'] = df[text_col].apply(get_ai_result)

            # --- RESULTS & METRICS ---
            st.success("Analysis Complete!")
            
            # Calculate Accuracy
            correct = (df['AI_Prediction'] == df[label_col]).sum()
            accuracy = (correct / len(df)) * 100

            m1, m2, m3 = st.columns(3)
            m1.metric("Total Reviews", len(df))
            m2.metric("Correct Predictions", correct)
            m3.metric("Model Accuracy", f"{accuracy:.1f}%")

            # --- VISUALIZATION ---
            st.subheader("Sentiment Distribution Comparison")
            # Creating a comparison chart between Actual vs AI
            fig = px.histogram(df, x="AI_Prediction", color=label_col, 
                               barmode="group", 
                               color_discrete_map={'POS': '#2ecc71', 'NEG': '#e74c3c'},
                               labels={'AI_Prediction': 'AI Predicted Sentiment', 'actual_label': 'Original Label'})
            st.plotly_chart(fig, use_container_width=True)

            # --- DATA TABLE ---
            st.subheader("Detailed Review Analysis")
            st.dataframe(df[[text_col, label_col, 'AI_Prediction']], use_container_width=True)

            # Download Result
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Full Results", data=csv, file_name="nlp_results.csv")
else:
    st.info("Waiting for CSV upload. Please upload the dataset converted from the Pang & Lee text files.")
