"""
Contains functions for preprocessing, vectorization, and classification.
"""

import os
import pickle
import pandas as pd
import numpy as np
import spacy
import gensim.downloader as api
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


class NewsClassifier:
    """Class for classifying news articles as Real or Fake using Word2Vec embeddings."""

    def __init__(self, model_path=None):
        print("Loading Word2Vec model...")
        self.wv = api.load("word2vec-google-news-300")
        print("Loading spaCy model...")
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            raise OSError(
                "spaCy model 'en_core_web_sm' not found. "
                "Please run: python -m spacy download en_core_web_sm"
            )
        self.clf = None
        self.model_path = os.path.abspath(model_path) if model_path else None
        if self.model_path and os.path.exists(self.model_path):
            print(f"Loading saved classifier from {self.model_path}...")
            self.load_model(self.model_path)
        else:
            print("No saved model found. Training new model...")
            self.train_model()

    def preprocess_and_vectorize(self, text):
        """
        Preprocess text and convert to vector using Word2Vec embeddings. [Input: string, Output: numpy array of shape (300,) representing the text vector]
        """
        if not text or not isinstance(text, str):
            return np.zeros(300)
        # Remove stop words and lemmatize the text
        doc = self.nlp(text)
        filtered_tokens = []
        for token in doc:
            if token.is_stop or token.is_punct:
                continue
            filtered_tokens.append(token.lemma_)
        if not filtered_tokens:
            return np.zeros(300)
        try:
            return self.wv.get_mean_vector(filtered_tokens, pre_normalize=False)
        except KeyError:
            return np.zeros(300)

    def train_model(self, data_path="fake_and_real_news.csv"):
        """
        Train the GradientBoostingClassifier on the dataset.
        """
        if not os.path.isabs(data_path):
            base_dir = os.path.dirname(os.path.abspath(__file__))
            data_path = os.path.join(base_dir, data_path)
        if not os.path.exists(data_path):
            print(f"Warning: Dataset {data_path} not found. Model will not be trained.")
            print("Please ensure the dataset is available or provide a saved model.")
            return
        print("Loading dataset...")
        df = pd.read_csv(data_path)
        df["label_num"] = df["label"].map({"Fake": 0, "Real": 1})
        print("Vectorizing texts (this may take a few minutes)...")
        df["vector"] = df["Text"].apply(
            lambda text: self.preprocess_and_vectorize(text)
        )
        X = np.stack(df["vector"].values)
        y = df["label_num"].values
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=2022, stratify=y
        )
        print("Training GradientBoostingClassifier...")
        self.clf = GradientBoostingClassifier()
        self.clf.fit(X_train, y_train)
        y_pred = self.clf.predict(X_test)
        print("\nModel Performance:")
        print(classification_report(y_test, y_pred))
        if self.model_path:
            self.save_model(self.model_path)

    def predict(self, text):
        """
        Classify a text as Real (1) or Fake (0).
        """
        if self.clf is None:
            raise ValueError("Classifier not trained. Please train the model first.")
        vector = self.preprocess_and_vectorize(text)
        vector_2d = vector.reshape(1, -1)
        prediction = self.clf.predict(vector_2d)[0]
        probabilities = self.clf.predict_proba(vector_2d)[0]
        confidence = probabilities[prediction]
        label_text = "Real" if prediction == 1 else "Fake"
        return prediction, confidence, label_text

    def save_model(self, path):
        """Save the trained classifier to disk."""
        if self.clf is None:
            raise ValueError("No model to save. Train the model first.")
        with open(path, "wb") as f:
            pickle.dump(self.clf, f)
        print(f"Model saved to {path}")

    def load_model(self, path):
        """Load a trained classifier from disk."""
        with open(path, "rb") as f:
            self.clf = pickle.load(f)
        print(f"Model loaded from {path}")


classifier = None


def get_classifier(model_name: str = "news_classifier.pkl"):
    """
    Get or create the global classifier instance.
    """
    global classifier
    # Resolve the pickle path relative to this module's directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, model_name)
    if classifier is None:
        classifier = NewsClassifier(model_path=model_path)
    return classifier
