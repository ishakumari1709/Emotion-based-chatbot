"""
Prototype Chatbot with Sentiments - BERT
"""

#pip install transformers
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BlenderbotSmallForConditionalGeneration, BlenderbotSmallTokenizer

# Load DistilBERT for sentiment analysis
sentiment_model_name = "distilbert-base-uncased"
sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_name)
sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_name)

# Load BlenderBot chatbot
chatbot_model_name = "facebook/blenderbot_small-90M"
chatbot_tokenizer = BlenderbotSmallTokenizer.from_pretrained(chatbot_model_name)
chatbot_model = BlenderbotSmallForConditionalGeneration.from_pretrained(chatbot_model_name)

while True:
    user_input = input("Enter your text (or 'exit' to quit): ")

    if user_input.lower() == 'exit':
        print("Exiting the program.")
        break

    # Sentiment analysis using DistilBERT
    user_input_tokens = sentiment_tokenizer(user_input, padding=True, truncation=True, return_tensors='pt')
    user_input_encoded = {key: val for key, val in user_input_tokens.items()}

    with torch.no_grad():
        user_output = sentiment_model(**user_input_encoded)
        user_logits = user_output.logits
        user_probs = torch.softmax(user_logits, dim=1).cpu().numpy()

    class_names = {0: "Negative", 1: "Positive"}
    predicted_sentiment_idx = np.argmax(user_probs)
    predicted_sentiment = class_names[predicted_sentiment_idx].capitalize()

    # Generate chatbot response based on sentiment
    chatbot_input = f"Predicted Sentiment: {predicted_sentiment}. {user_input}"
    chatbot_input_ids = chatbot_tokenizer.encode(chatbot_input, return_tensors="pt")

    chatbot_response_ids = chatbot_model.generate(chatbot_input_ids, max_length=50, num_return_sequences=1, pad_token_id=chatbot_tokenizer.eos_token_id)
    chatbot_response = chatbot_tokenizer.decode(chatbot_response_ids[0], skip_special_tokens=True)

    # Print the predicted sentiment, sentiment probabilities, and chatbot response
    print(f"Predicted Sentiment: {predicted_sentiment}")
    for class_idx, class_name in class_names.items():
        print(f"{class_name.capitalize()} Sentiment Probability: {user_probs[0][class_idx]:.4f}")
    print(f"Chatbot: {chatbot_response}")
