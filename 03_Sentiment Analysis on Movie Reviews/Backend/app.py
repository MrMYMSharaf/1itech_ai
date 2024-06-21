from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
from bs4 import BeautifulSoup
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import logging

nltk.download('stopwords')

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Load the model and vectorizer
try:
    classifier = joblib.load('../Model/sentiment_classifier.pkl')
    vectorizer = joblib.load('../Model/tfidf_vectorizer.pkl')
    app.logger.debug("Model and vectorizer loaded successfully.")
except Exception as e:
    app.logger.error(f"Error loading model or vectorizer: {e}")

stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

def remove_html_tags(text):
    return BeautifulSoup(text, "lxml").text

def remove_punctuation(sentence):
    return ''.join([letters.lower() for letters in sentence if letters not in string.punctuation])

def remove_stopwords(sentence):
    return ' '.join([words for words in sentence.split() if words.lower() not in stop_words])

def stem_sentence(sentence):
    return ' '.join([ps.stem(word) for word in sentence.split()])

def preprocess_review(review):
    try:
        review = remove_html_tags(review)
        review = remove_punctuation(review)
        review = remove_stopwords(review)
        review = stem_sentence(review)
    except Exception as e:
        app.logger.error(f"Error during preprocessing: {e}")
    return review

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        review = data['review']
        app.logger.debug(f"Received review: {review}")
        preprocessed_review = preprocess_review(review)
        review_vectorized = vectorizer.transform([preprocessed_review])
        prediction = classifier.predict(review_vectorized)
        sentiment = 'Positive' if prediction[0] == 1 else 'Negative'
        app.logger.debug(f"Prediction: {sentiment}")
        return jsonify({'sentiment': sentiment})
    except Exception as e:
        app.logger.error(f"Error during prediction: {e}")
        return jsonify({'error': 'Internal Server Error'}), 500

if __name__ == '__main__':
    app.run(debug=True)
