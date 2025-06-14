import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from visualization import PLOT_PATH


class SentimentAnalyzer:
    def __init__(self):
        self.vectorizer = CountVectorizer(stop_words='english')
        self.model = MultinomialNB()

    def prepare_features(self, texts):
        """Convert text to feature vectors"""
        print("\nConverting text to feature vectors...")
        X = self.vectorizer.fit_transform(texts)
        print(
            f"Vocabulary size: {len(self.vectorizer.get_feature_names_out())}")
        return X

    def train(self, X, y):
        """Train the model"""
        print("\nSplitting data into train and test sets...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

        print(f"Training set size: {X_train.shape[0]}")
        print(f"Test set size: {X_test.shape[0]}")

        print("\nTraining the model...")
        self.model.fit(X_train, y_train)

        # Evaluate on test set
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nModel accuracy: {accuracy * 100:.2f}%")

        # Print classification report
        print("\nDetailed classification report:")
        print(classification_report(y_test, y_pred))

        # Plot confusion matrix
        self._plot_confusion_matrix(y_test, y_pred)

        return X_test, y_test

    def _plot_confusion_matrix(self, y_true, y_pred):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Negative', 'Positive'],
                    yticklabels=['Negative', 'Positive'])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig(os.path.join(PLOT_PATH, 'confusion_matrix.png'))
        plt.close()

    def predict(self, text):
        """Predict sentiment for new text"""
        processed_text = text.lower()
        vect = self.vectorizer.transform([processed_text])
        prediction = self.model.predict(vect)[0]
        return "Positive" if prediction == "pos" else "Negative"
