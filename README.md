# Movie Genre Classification

This project aims to classify movies into their respective genres based on their plot summaries using natural language processing (NLP) and machine learning techniques.

## Project Overview

Movie genre classification can be challenging due to the ambiguity and overlap between genres. By using plot summaries, we aim to create a machine learning model that can predict the genre(s) of a movie.

## Dataset

The dataset used contains plot summaries of movies along with their associated genres. The dataset was cleaned and preprocessed for training.

## Model

We applied several machine learning and deep learning algorithms, including:

- **Logistic Regression**
- **Support Vector Machines (SVM)**
- **Random Forest**
- **Deep Learning Models (e.g., LSTM, BERT)**

## Steps

1. **Data Preprocessing**:
   - Cleaning the text data (removing stopwords, lemmatization, etc.).
   - Tokenizing and vectorizing the text.
   
2. **Exploratory Data Analysis (EDA)**:
   - Visualizing the distribution of movie genres.
   - Analyzing the most common words in each genre.
   
3. **Model Training**:
   - Training multiple models and evaluating their performance using metrics like accuracy, F1-score, precision, and recall.
   
4. **Hyperparameter Tuning**:
   - Tuning the models using GridSearchCV to optimize performance.
   
5. **Evaluation**:
   - Testing the best model on a held-out test dataset.
   - Analyzing misclassifications.

## Results

The best-performing model achieved an accuracy of `XX%` with an F1-score of `YY%`. Deep learning models like LSTM and BERT showed promising results but required more computational power.

## Future Work

- Implement multi-label classification for movies with multiple genres.
- Experiment with more advanced NLP techniques such as transformers (e.g., BERT, GPT).

## Installation

To run this project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/movie-genre-classification.git
   cd movie-genre-classification
