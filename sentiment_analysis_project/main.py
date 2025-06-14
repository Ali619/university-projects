from data_loader import download_data, load_data, prepare_data
from model import SentimentAnalyzer
from visualization import (create_wordcloud, plot_review_length_distribution,
                           plot_word_frequency)


def main():
    print("=" * 50)
    print("Movie Review Sentiment Analysis Project")
    print("=" * 50)

    # Download and load data
    download_data()
    df = load_data()

    # Prepare data
    texts, labels = prepare_data(df)

    # Create visualizations
    plot_word_frequency(texts)
    create_wordcloud(texts)
    plot_review_length_distribution(texts, labels)

    # Train and evaluate model
    analyzer = SentimentAnalyzer()
    X = analyzer.prepare_features(texts)
    X_test, y_test = analyzer.train(X, labels)

    # Test with some example reviews
    print("\n" + "=" * 50)
    print("Testing the model with example reviews:")
    print("=" * 50)

    test_reviews = [
        "This movie was absolutely fantastic! I loved every minute of it.",
        "The acting was terrible and the plot made no sense.",
        "A masterpiece of modern cinema, with brilliant performances.",
        "I've never been more bored in my life. Complete waste of time."
    ]

    for review in test_reviews:
        sentiment = analyzer.predict(review)
        print(f"\nReview: {review}")
        print(f"Predicted sentiment: {sentiment}")

    print("\n" + "=" * 50)
    print("Analysis complete! Check the generated plots:")
    print("- sentiment_distribution.png")
    print("- word_frequency.png")
    print("- wordcloud.png")
    print("- review_length_distribution.png")
    print("- confusion_matrix.png")
    print("=" * 50)


if __name__ == "__main__":
    main()
