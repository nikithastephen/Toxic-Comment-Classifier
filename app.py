import streamlit as st
import pickle
import nltk
from nltk.stem.snowball import SnowballStemmer
import re
import pandas as p

# Load pre-trained model
with open("lr_pipeline.pkl", "rb") as model_file:
    LR_pipeline = pickle.load(model_file)

# Load column labels
labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]  # Adjust based on your dataset

# Text Preprocessing
stemmer = SnowballStemmer('english')


def clean_text(text):
    text = text.lower()
    text = re.sub(r"\W", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def stemming(sentence):
    return " ".join(stemmer.stem(word) for word in sentence.split())


# Streamlit UI
st.title("Toxic Comment Classification")
st.write("Enter a comment and check for toxicity levels!")

user_input = st.text_area("Enter text:")

if st.button("Analyze"):
    if user_input:
        processed_text = stemming(clean_text(user_input))
        results = LR_pipeline.predict([processed_text])[0]

        # Display results
        st.subheader("Classification Results")
        for label, result in zip(labels, results):
            st.write(f"**{label}**: {'Yes' if result else 'No'}")
    else:
        st.warning("Please enter some text to analyze.")
