import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import joblib

# Load and preprocess the data
@st.cache_data
def load_model_and_vectorizer():
    # Load data
    data = pd.read_csv("emotions.csv")
    X = data["text"]
    y = data["label"]

    # Fit vectorizer
    vectorizer = TfidfVectorizer(stop_words="english", max_features=10000)
    X_tfidf = vectorizer.fit_transform(X)

    # Fit LinearSVC
    model = LinearSVC(random_state=42, dual=False, max_iter=1000)
    model.fit(X_tfidf, y)

    return model, vectorizer

# Map numeric labels to emotion names
label_mapping = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise"
}

# Load model and vectorizer
model, vectorizer = load_model_and_vectorizer()

# Streamlit app layout
st.title("Tweet Emotion Classifier (SVM)")
st.write("Enter a tweet and predict its emotion using a trained Support Vector Machine.")

tweet = st.text_area("Enter tweet text here:")

if st.button("Classify"):
    if tweet.strip() == "":
        st.warning("Please enter a tweet.")
    else:
        tweet_tfidf = vectorizer.transform([tweet])
        prediction = model.predict(tweet_tfidf)[0]
        emotion = label_mapping.get(prediction, "Unknown")
        st.success(f"Predicted Emotion: {emotion}")
