import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from wordcloud import WordCloud

CURR_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_PATH = os.path.join(CURR_DIR, 'plots')
os.makedirs(PLOT_PATH, exist_ok=True)


def plot_word_frequency(texts, n_words=20):
    """Plot the most frequent words in the dataset"""
    print("\nAnalyzing word frequencies...")

    # Combine all texts and split into words
    all_words = ' '.join(texts).split()
    word_freq = Counter(all_words)

    # Get the most common words
    most_common = word_freq.most_common(n_words)
    words, frequencies = zip(*most_common)

    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.bar(words, frequencies)
    plt.xticks(rotation=45, ha='right')
    plt.title(f'Top {n_words} Most Frequent Words')
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_PATH, 'word_frequency.png'))
    plt.close()

    print(
        f"Word frequency plot saved as '{os.path.join(PLOT_PATH, 'word_frequency.png')}'")


def create_wordcloud(texts):
    """Create a word cloud from the texts"""
    print("\nGenerating word cloud...")

    # Combine all texts
    text = ' '.join(texts)

    # Create and generate a word cloud image
    wordcloud = WordCloud(width=800, height=400,
                          background_color='white',
                          max_words=100).generate(text)

    # Display the word cloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud of Movie Reviews')
    plt.savefig(os.path.join(PLOT_PATH, 'wordcloud.png'))
    plt.close()
    print(PLOT_PATH)

    print(f"Word cloud saved as '{os.path.join(PLOT_PATH, 'wordcloud.png')}'")


def plot_review_length_distribution(texts, labels):
    """Plot the distribution of review lengths"""
    print("\nAnalyzing review lengths...")

    # Calculate review lengths
    lengths = [len(text.split()) for text in texts]

    # Create the plot
    plt.figure(figsize=(10, 6))
    sns.histplot(data=pd.DataFrame({'length': lengths, 'sentiment': labels}),
                 x='length', hue='sentiment', bins=50)
    plt.title('Distribution of Review Lengths by Sentiment')
    plt.xlabel('Number of Words')
    plt.ylabel('Count')
    plt.savefig(os.path.join(PLOT_PATH, 'review_length_distribution.png'))
    plt.close()

    print(
        f"Review length distribution plot saved as '{os.path.join(PLOT_PATH, 'review_length_distribution.png')}'")

    # Print some statistics
    print("\nReview Length Statistics:")
    print(f"Average length: {np.mean(lengths):.2f} words")
    print(f"Median length: {np.median(lengths):.2f} words")
    print(f"Minimum length: {min(lengths)} words")
    print(f"Maximum length: {max(lengths)} words")
