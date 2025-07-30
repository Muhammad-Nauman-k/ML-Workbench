# IMDB Sentiment Analysis
A sentiment analysis for the IMDB movie 50k reviews dataset using LogisticRegression with TF-IDF features and achieves ~87.42% test accuracy.

## Features
- **Preprocessing**: Removes HTML tags, punctuation, and converts to lowercase and lemmatization 
- **Vectorization**: Uses TF-IDF with unigrams and bigrams 
- **Model**: LogisticRegression with hyperparameter tuning (`C=1.9`).

## Installation
```bash
pip install -r requirements.txt