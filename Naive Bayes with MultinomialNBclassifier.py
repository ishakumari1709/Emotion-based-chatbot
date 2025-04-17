# -*- coding: utf-8 -*-
"""

Naive Bayes with MultinomialNB classifier

Data Preperation and Libraries

"""

# Importing the required libraries
#pip install nltk

import re
import nltk

# Download NLTK data
nltk.download('punkt')
nltk.download('words')
nltk.download('movie_reviews')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

from nltk.tokenize import word_tokenize
from nltk.corpus import words
from nltk.stem import WordNetLemmatizer
from nltk.corpus import movie_reviews

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

"""Data Cleaning"""

# Define a function to check if a word is an English word
def is_english_word(word):
    return word.lower() in english_words

# Define a set of English words
english_words = set(words.words())

# Initialize the WordNet Lemmatizer
lemmatizer = WordNetLemmatizer()

# Modify the preprocess_text function to use the WordNet Lemmatizer for all categories
def preprocess_text(text):
    # Remove non-alphanumeric characters (excluding spaces)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

    # Tokenization
    words = word_tokenize(text.lower())  # Convert to lowercase and tokenize

    # Lemmatize all words using NLTK's WordNet lemmatizer
    cleaned_words = []
    for word in words:
        lemma = lemmatizer.lemmatize(word)
        if lemma.isalpha() and is_english_word(lemma):
            cleaned_words.append(lemma)

    return ' '.join(cleaned_words)

# movie_reviews dataset
positive_reviews = movie_reviews.fileids('pos')
negative_reviews = movie_reviews.fileids('neg')

# Combine positive and negative reviews with preprocessing (including cleaning)
all_reviews = [(preprocess_text(movie_reviews.raw(fileid)), 'pos') for fileid in positive_reviews] + \
              [(preprocess_text(movie_reviews.raw(fileid)), 'neg') for fileid in negative_reviews]

"""*   Test - Train Data Split
*   Classification Report



"""

# Splitting the data into training and testing sets
train_reviews, test_reviews = train_test_split(all_reviews, test_size=0.2, random_state=42)

# Initialize the vectorizer and classifier
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform([text for text, _ in train_reviews])  # Convert text data to numerical features
X_test = vectorizer.transform([text for text, _ in test_reviews])  # Transform test data using the same vectorizer

# Get the vocabulary from the vectorizer
vocabulary = vectorizer.get_feature_names_out()

# Extract true labels for training and testing data (pos/neg)
y_train = [label for _, label in train_reviews]
y_test = [label for _, label in test_reviews]

# Initialize and train the Naive Bayes classifier
nb_classifier = MultinomialNB() #Naive Bayes machine learning algorithm
nb_classifier.fit(X_train, y_train)  # Train the classifier using training data

# Predict using the trained Naive Bayes model - predict sentiment labels for the test data, generating an array of predictions stored in nb_predictions.
nb_predictions = nb_classifier.predict(X_test)  # Make predictions on the test data

# Calculate accuracy - Accuracy of the predictions by comparing them to the actual test labels (y_test).
accuracy = accuracy_score(y_test, nb_predictions)
print(f"Accuracy: {accuracy:.2f}")

# Generate Classification Report - detailed metrics about the performance of our classification model.
print("Classification Report for Naive Bayes Sentiment Analysis:\n")
print(classification_report(y_test, nb_predictions))

"""Top Sentiment Words - Problems about Data Cleaning / Lemmatization"""

# POS-Tagging with NLTK

def print_top_sentiment_words(words_list, category_name, num_words=10):
    word_sentiment_scores = {word: nb_classifier.predict_proba(vectorizer.transform([word]))[0] for word in words_list if is_english_word(word)}

    positive_words = [(word, sentiment[1]) for word, sentiment in word_sentiment_scores.items() if sentiment[1] > sentiment[0]]
    negative_words = [(word, sentiment[0]) for word, sentiment in word_sentiment_scores.items() if sentiment[0] > sentiment[1]]

    positive_words.sort(key=lambda x: x[1], reverse=True)
    negative_words.sort(key=lambda x: x[1], reverse=True)

    print(f"Top {num_words} Positive {category_name}:")
    for word, sentiment in positive_words[:num_words]:
        print(f"{word}: {sentiment:.4f} sentiment")

    print(f"\nTop {num_words} Negative {category_name}:")
    for word, sentiment in negative_words[:num_words]:
        print(f"{word}: {sentiment:.4f} sentiment")


# Separate words based on their POS tags - NLTK POS
adjectives = [word for word in vocabulary if nltk.pos_tag([word])[0][1].startswith('JJ')]
verbs = [word for word in vocabulary if nltk.pos_tag([word])[0][1].startswith('VB')]
nouns = [word for word in vocabulary if nltk.pos_tag([word])[0][1].startswith('NN')]


# Print top sentiment words for each category
print_top_sentiment_words(adjectives, "Adjectives")
print()
print_top_sentiment_words(verbs, "Verbs")
print()
print_top_sentiment_words(nouns, "Nouns")

"""Sentiment Analysis"""

while True:
    user_input = input("Enter your text (or 'exit' to quit): ")

    if user_input.lower() == 'exit':
        print("Exiting the program.")
        break

    # Convert the input sentence to a numerical feature vector
    user_input_vector = vectorizer.transform([user_input])

    # Predict sentiment probabilities using the trained Naive Bayes model
    sentiment_probabilities = nb_classifier.predict_proba(user_input_vector)[0]

    # Get the predicted sentiment class
    predicted_sentiment_class = nb_classifier.predict(user_input_vector)[0]

    if predicted_sentiment_class == 'pos':
        predicted_emotion = "positive"
    else:
        predicted_emotion = "negative"

    # Print the predicted sentiment and confidence for each class
    print(f"Predicted Sentiment: {predicted_emotion}")
    print(f"Positive Confidence: {sentiment_probabilities[1]:.4f}")
    print(f"Negative Confidence: {sentiment_probabilities[0]:.4f}")
