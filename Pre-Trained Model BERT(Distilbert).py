
"""
This model takes a long time to load and Data Split (>45 Minutes). 
For this reason, the BERT file results faster when run through GoogleColab (with T4 GPU - 2 Minutes). 
Since GoogleColab runs this code using GPU, the Train-Test split and model loading takes 2 minutes, 
but when you run this code on your own computer on a platform like VS Code, this part of the code takes a very long time (>45 Minutes).

Pre-Trained Model BERT (Distilbert)

Data Preperation and Libraries
"""

# Importing the required libraries
#pip install nltk
#pip install transformers

import re
import nltk

# Download NLTK data
nltk.download('punkt')
nltk.download('words')
nltk.download('movie_reviews')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

from nltk.tokenize import word_tokenize
from nltk.corpus import words
from nltk.stem import WordNetLemmatizer
from nltk.corpus import movie_reviews
from collections import defaultdict

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

    # Tokenization#
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

"""Modal Loading and Test - Train Data Split"""

# Load pre-trained BERT model and tokenizer
model_name = "distilbert-base-uncased"
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertForSequenceClassification.from_pretrained(model_name)


# Data preprocessing
positive_reviews = movie_reviews.fileids('pos')
negative_reviews = movie_reviews.fileids('neg')

all_reviews = [movie_reviews.raw(fileid) for fileid in positive_reviews] + \
              [movie_reviews.raw(fileid) for fileid in negative_reviews]
labels = [1] * len(positive_reviews) + [0] * len(negative_reviews)  # 1 for positive, 0 for negative

# Tokenize and encode the reviews
encoded_reviews = tokenizer(all_reviews, padding=True, truncation=True, return_tensors='pt')

# Split the data into training and testing sets
train_inputs, test_inputs, train_labels, test_labels = train_test_split(encoded_reviews['input_ids'],
                                                                      torch.tensor(labels),
                                                                      test_size=0.2,
                                                                      random_state=42)

# Define a data loader for training
batch_size = 8
train_dataset = torch.utils.data.TensorDataset(train_inputs, train_labels)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Fine-tune BERT on the sentiment analysis task
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
criterion = torch.nn.CrossEntropyLoss()

num_epochs = 1  # Increase this for better performance
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.train()

for epoch in range(num_epochs):
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs, labels=labels)
        loss = criterion(outputs.logits, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss:.4f}")

"""Classification Report"""

# Evaluation on the test set
model.eval()

# Move test inputs and labels to the device
test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)

# Initialize variables to store predictions
all_predicted_labels = []
batch_size = 8  # Adjust the batch size for inference

# Perform inference in batches to reduce GPU memory usage
with torch.no_grad():
    for i in range(0, len(test_inputs), batch_size):
        batch_inputs = test_inputs[i:i+batch_size]

        # Forward pass for the batch
        batch_outputs = model(batch_inputs)
        batch_logits = batch_outputs.logits

        # Convert logits to probabilities and get predicted labels
        batch_probs = torch.softmax(batch_logits, dim=1)
        batch_predicted_labels = torch.argmax(batch_probs, dim=1).cpu().numpy()

        # Append predicted labels for this batch to the list
        all_predicted_labels.extend(batch_predicted_labels)

# Convert the list of predicted labels to a numpy array
predicted_labels = np.array(all_predicted_labels)

# Calculate accuracy
accuracy = accuracy_score(test_labels.cpu().numpy(), predicted_labels)
print(f"Accuracy: {accuracy:.2f}")

# Generate Classification Report
class_names = ['negative', 'positive']
report = classification_report(test_labels.cpu().numpy(), predicted_labels, target_names=class_names)
print("Classification Report for BERT Sentiment Analysis:\n")
print(report)

"""Sentiment Analysis

"""

while True:
    user_input = input("Enter your text (or 'exit' to quit): ")

    if user_input.lower() == 'exit':
        print("Exiting the program.")
        break

    # Tokenize and encode the user input
    user_input_tokens = tokenizer(user_input, padding=True, truncation=True, return_tensors='pt')
    user_input_encoded = {key: val.to(device) for key, val in user_input_tokens.items()}

    # Predict sentiment probabilities
    with torch.no_grad():
        user_output = model(**user_input_encoded)
        user_logits = user_output.logits
        user_probs = torch.softmax(user_logits, dim=1).cpu().numpy()

    # Modify the class names
    class_names = {0: "Negative", 1: "Positive"}

    # Find the sentiment class with the highest probability
    predicted_sentiment_idx = np.argmax(user_probs)
    predicted_sentiment = class_names[predicted_sentiment_idx].capitalize()

    # Print the predicted sentiment and sentiment probabilities
    print(f"Predicted Sentiment: {predicted_sentiment}")
    for class_idx, class_name in class_names.items():
        print(f"{class_name.capitalize()} Sentiment Probability: {user_probs[0][class_idx]:.4f}")
