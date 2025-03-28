import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

# Ensure this is the first Streamlit command
st.set_page_config(page_title="Sentiment Analysis", layout="centered")

# Load Pretrained Model & Tokenizer
MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"  # Pretrained sentiment model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

# App Title
st.title("🔥 Sentiment Analysis with Transformers")
st.write("Enter text below to analyze sentiment using a BERT-based model.")

# Text Input
user_input = st.text_area("Enter text here:", "")

if st.button("Analyze Sentiment"):
    if user_input.strip():
        # Tokenize and Convert Input to Tensor
        inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)
        
        # Perform Sentiment Analysis
        with torch.no_grad():
            outputs = model(**inputs)
            scores = F.softmax(outputs.logits, dim=-1).squeeze().tolist()
        
        # Interpret Results
        labels = ["Very Negative 😡", "Negative 😞", "Neutral 😐", "Positive 🙂", "Very Positive 😍"]
        sentiment = labels[scores.index(max(scores))]  # Select sentiment with highest score

        # Display Results
        st.subheader("Sentiment Analysis Result:")
        st.write(f"**Predicted Sentiment:** {sentiment}")
        st.write("**Confidence Scores:**")
        for label, score in zip(labels, scores):
            st.write(f"- {label}: {score:.2%}")

    else:
        st.warning("⚠️ Please enter text before analyzing.")

# Footer
st.markdown("---")
st.markdown("💡 *Built with PyTorch, Hugging Face Transformers & Streamlit*")

