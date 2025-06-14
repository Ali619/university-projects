# Movie Review Sentiment Analysis

This project implements a sentiment analysis system for movie reviews using Natural Language Processing (NLP) and Machine Learning techniques. The system can classify movie reviews as either positive or negative.

## Project Structure

- `main.py`: The main script that runs the entire analysis
- `data_loader.py`: Handles data loading and preprocessing
- `model.py`: Contains the sentiment analysis model implementation
- `visualization.py`: Creates various visualizations of the data and results
- `requirements.txt`: Lists all required Python packages

## Features

- Data preprocessing and text cleaning
- Feature extraction using Bag-of-Words
- Naive Bayes classification
- Multiple visualizations:
  - Sentiment distribution
  - Word frequency analysis
  - Word cloud
  - Review length distribution
  - Confusion matrix
- Model evaluation metrics
- Example predictions

## Installation

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

Run the main script:
```bash
python main.py
```

The script will:
1. Download the required NLTK data
2. Load and preprocess the movie reviews
3. Create various visualizations
4. Train the sentiment analysis model
5. Evaluate the model
6. Test with example reviews

## Output

The script generates several visualization files:
- `sentiment_distribution.png`: Distribution of positive and negative reviews
- `word_frequency.png`: Most frequent words in the dataset
- `wordcloud.png`: Word cloud visualization of the reviews
- `review_length_distribution.png`: Distribution of review lengths
- `confusion_matrix.png`: Model performance visualization

## Model Performance

The model's performance is evaluated using:
- Accuracy score
- Classification report (precision, recall, F1-score)
- Confusion matrix 