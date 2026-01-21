import streamlit as st
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import plotly.express as px

# Download the VADER lexicon (required for sentiment analysis)
nltk.download('vader_lexicon')

# Initialize the analyzer
analyzer = SentimentIntensityAnalyzer()

def get_sentiment(text):
    if isinstance(text, str):
        score = analyzer.polarity_scores(text)['compound']
        if score >= 0.05:
            return 'Positive'
        elif score <= -0.05:
            return 'Negative'
        else:
            return 'Neutral'
    return 'None'

# Streamlit UI
st.set_page_config(page_title="Movie Review Sentiment Analyzer", layout="wide")

st.title("🎬 Movie Review Sentiment Analyzer")
st.markdown("Upload a CSV file containing movie reviews to analyze their sentiment automatically.")

# 1. File Upload Section
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    st.write("### Preview of Uploaded Data")
    st.dataframe(df.head())

    # User selects the column containing the reviews
    column_name = st.selectbox("Select the column containing the reviews:", df.columns)

    if st.button("Run Sentiment Analysis"):
        with st.spinner('Analyzing sentiments...'):
            # Apply analysis
            df['Sentiment'] = df[column_name].apply(get_sentiment)
            
            # 2. Result Section
            st.success("Analysis Complete!")
            
            # Show Metrics
            col1, col2, col3 = st.columns(3)
            counts = df['Sentiment'].value_counts()
            col1.metric("Positive Reviews", counts.get('Positive', 0))
            col2.metric("Negative Reviews", counts.get('Negative', 0))
            col3.metric("Neutral Reviews", counts.get('Neutral', 0))

            # Visualizations
            st.write("### Sentiment Distribution")
            fig = px.pie(df, names='Sentiment', title='Review Sentiment Breakdown',
                         color_discrete_map={'Positive':'#2ecc71', 'Negative':'#e74c3c', 'Neutral':'#95a5a6'})
            st.plotly_chart(fig)

            # Show Data with Results
            st.write("### Detailed Results")
            st.dataframe(df[[column_name, 'Sentiment']])

            # Download Option
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download results as CSV",
                data=csv,
                file_name="sentiment_results.csv",
                mime="text/csv",
            )
else:
    st.info("Please upload a CSV file to get started.")
