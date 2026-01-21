import streamlit as st
from transformers import pipeline

# Load a pre-trained sentiment analysis model (e.g., DistilBERT fine-tuned on SST-2)
classifier = pipeline("sentiment-analysis")

# Streamlit app title and description
st.title("Movie Review Sentiment Analyzer")
st.markdown("Enter a movie review below to analyze its sentiment (Positive or Negative). This app uses a pre-trained NLP model for classification.")

# Text input for the movie review
review = st.text_area("Enter your movie review here:", height=150)

# Button to analyze
if st.button("Analyze Sentiment"):
    if review:
        # Perform sentiment analysis
        result = classifier(review)[0]
        label = result['label']
        score = result['score']
        
        # Display results
        if label == "POSITIVE":
            st.success(f"Positive Sentiment! Confidence: {score:.2f}")
        else:
            st.error(f"Negative Sentiment! Confidence: {score:.2f}")
        
        st.markdown("### Analysis Details")
        st.write(f"Review: {review}")
        st.write(f"Predicted Label: {label}")
        st.write(f"Confidence Score: {score:.2f}")
    else:
        st.warning("Please enter a review to analyze.")

# Footer with project info
st.markdown("---")
st.markdown("This is a basic NLP project for sentiment analysis on movie reviews, similar to IMDB classification tasks<sup>1</sup><sup>2</sup><sup>9</sup>.")
