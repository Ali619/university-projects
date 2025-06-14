import os
import string

import matplotlib.pyplot as plt
import nltk
import pandas as pd
import seaborn as sns
from nltk.corpus import movie_reviews
from visualization import PLOT_PATH

CURR_DIR = os.path.dirname(os.path.abspath(__file__))
DOWNLOAD_PATH = f"{CURR_DIR}/data/"
os.makedirs(DOWNLOAD_PATH, exist_ok=True)


def download_data():
    """Download required NLTK data"""
    print("Downloading NLTK movie reviews dataset...")
    try:
        # First try to download to the default location
        nltk.download('movie_reviews', quiet=True, download_dir=DOWNLOAD_PATH)
    except Exception as e:
        print(f"Error downloading to default location: {e}")
        # If that fails, try to download to a specific directory
        nltk_data_dir = os.path.join(os.path.dirname(__file__), 'nltk_data')
        os.makedirs(nltk_data_dir, exist_ok=True)
        nltk.download('movie_reviews', download_dir=nltk_data_dir, quiet=True)
        nltk.data.path.append(nltk_data_dir)
    print("Download complete!")


def load_data():
    """Load and prepare the movie reviews dataset"""
    print("\nLoading movie reviews dataset...")
    try:
        # Try to load the data
        documents = [(movie_reviews.raw(fileid), category)
                     for category in movie_reviews.categories()
                     for fileid in movie_reviews.fileids(category)]
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Attempting to re-download the dataset...")
        download_data()
        # Try loading again after re-download
        documents = [(movie_reviews.raw(fileid), category)
                     for category in movie_reviews.categories()
                     for fileid in movie_reviews.fileids(category)]

    # Create a DataFrame for better analysis
    df = pd.DataFrame(documents, columns=['text', 'sentiment'])

    # Display basic statistics
    print("\nDataset Statistics:")
    print(f"Total number of reviews: {len(df)}")
    print("\nDistribution of sentiments:")
    print(df['sentiment'].value_counts())

    # Plot sentiment distribution
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x='sentiment')
    plt.title('Distribution of Movie Reviews by Sentiment')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.savefig(os.path.join(PLOT_PATH, 'sentiment_distribution.png'))
    plt.close()

    return df


def preprocess(text):
    """Preprocess the text data"""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text


def prepare_data(df):
    """Prepare the data for model training"""
    print("\nPreprocessing the text data...")
    df['processed_text'] = df['text'].apply(preprocess)

    # Display sample of processed text
    print("\nSample of processed text:")
    print(df[['text', 'processed_text']].head())

    return df['processed_text'], df['sentiment']
