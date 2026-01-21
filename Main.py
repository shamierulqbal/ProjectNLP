import streamlit as st
import pandas as pd
import plotly.express as px
from transformers import pipeline

# --- MODEL LOADING ---
# Caching ensures the model loads once and stays in memory
@st.cache_resource
def load_sentiment_model():
    # Using DistilBERT: It's small, fast, and perfect for Streamlit Cloud
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

classifier = load_sentiment_model()

# --- APP INTERFACE ---
st.set_page_config(page_title="NLP Movie Review Analyzer", layout="wide")
st.title("🎬 Movie Review Sentiment Analyzer")
st.write("Upload a movie review dataset (CSV) and get instant sentiment results.")

# --- FILE UPLOADER ---
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the data
    df = pd.read_csv(uploaded_file)
    
    st.subheader("Data Preview")
    st.dataframe(df.head())

    # Column Selection
    text_col = st.selectbox("Select the column with reviews:", df.columns)

    if st.button("Analyze Sentiments"):
        with st.spinner("Processing reviews with AI..."):
            # Sentiment Logic
            def get_ai_sentiment(text):
                if pd.isna(text) or text == "": return "Neutral"
                # Keep text within model limits (max 512 tokens)
                result = classifier(str(text)[:512])[0]
                return result['label']

            # Apply to dataframe
            df['Sentiment'] = df[text_col].apply(get_ai_sentiment)

            # --- RESULTS DISPLAY ---
            st.success("Analysis Finished!")
            
            # Show Metrics
            counts = df['Sentiment'].value_counts()
            m1, m2 = st.columns(2)
            m1.metric("Positive Reviews", counts.get('POSITIVE', 0))
            m2.metric("Negative Reviews", counts.get('NEGATIVE', 0))

            # Plotly Chart
            fig = px.pie(df, names='Sentiment', color='Sentiment',
                         color_discrete_map={'POSITIVE': 'green', 'NEGATIVE': 'red'},
                         title="Overall Sentiment Distribution")
            st.plotly_chart(fig)

            # Download Result
            st.write("### Final Results")
            st.dataframe(df[[text_col, 'Sentiment']])
            
            csv_data = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Analysis as CSV", data=csv_data, file_name="results.csv")

else:
    st.info("👆 Please upload a CSV file to begin.")
