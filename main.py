# Importing necessary libraries
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from textblob import TextBlob
import random
from gtts import gTTS
import os
import IPython.display as ipd

# Create a larger sample dataset
data = pd.DataFrame({
    'interaction': [
        "I am very disappointed with your service.",
        "Can you please help me with my issue?",
        "I need to speak with a supervisor.",
        "Thank you for your assistance!",
        "This is the worst experience I've ever had.",
        "Your service is excellent, keep it up!",
        "I am frustrated with the delay in response.",
        "I appreciate your help, thanks!",
        "I need urgent assistance.",
        "Everything is fine, no issues at all.",
        "Fuck you, asshole!",
        "This service is terrible, I want to speak to a manager.",
        "Why is my issue not resolved yet?",
        "You guys are the best, thank you so much!",
        "I can't believe how bad this service is.",
        "I am extremely happy with the service.",
        "Can someone competent help me?",
        "I love the support I received, thank you!",
        "This is unacceptable, I need to escalate this now.",
        "Great job on solving my issue!"
    ],
    'escalation_needed': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0]
})

# Preprocessing function to clean text data
def preprocess_text(text):
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.lower()
    return text

# Apply preprocessing
data['interaction'] = data['interaction'].apply(preprocess_text)

# Function to get sentiment polarity
def get_sentiment(text):
    return TextBlob(text).sentiment.polarity

# Apply sentiment analysis
data['sentiment'] = data['interaction'].apply(get_sentiment)

# Normalize sentiment values to 0-1 range
data['sentiment'] = (data['sentiment'] - data['sentiment'].min()) / (data['sentiment'].max() - data['sentiment'].min())

# Feature Engineering: Create TF-IDF vectors
tfidf = TfidfVectorizer(max_features=5000)
X_tfidf = tfidf.fit_transform(data['interaction']).toarray()

# Combine TF-IDF features with sentiment
X = np.hstack((X_tfidf, data['sentiment'].values.reshape(-1, 1)))

# Labels (assuming regression target is the sentiment score for demonstration)
y = data['sentiment']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the regression model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predictions
y_pred = regressor.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'Mean Absolute Error: {mae}')
print(f'R-squared: {r2}')

# Define a threshold for accuracy
threshold = 0.1

# Calculate the number of accurate predictions
accurate_predictions = np.sum(np.abs(y_pred - y_test) <= threshold)

# Calculate the accuracy
accuracy = accurate_predictions / len(y_test) * 100

print(f'Custom Accuracy (within ±{threshold}): {accuracy:.2f}%')

# Rule-based Empathetic Responses
def generate_response(predicted_sentiment):
    if predicted_sentiment > 0.5:  # Arbitrary threshold for positive sentiment
        response = "Thank you for your feedback. How else can we assist you today?"
    else:
        response = "I understand your frustration. Let me connect you to an agent who can help you further."
    return response

def ivr_system(interaction):
    processed_interaction = preprocess_text(interaction)
    sentiment = get_sentiment(processed_interaction)
    sentiment_normalized = (sentiment - data['sentiment'].min()) / (data['sentiment'].max() - data['sentiment'].min())
    vectorized_interaction = tfidf.transform([processed_interaction]).toarray()
    interaction_features = np.hstack((vectorized_interaction, [[sentiment_normalized]]))
    predicted_sentiment = regressor.predict(interaction_features)[0]
    response = generate_response(predicted_sentiment)
    
    # Convert response to speech
    tts = gTTS(response)
    tts.save("response.mp3")
    ipd.display(ipd.Audio("response.mp3", autoplay=True))
    
    return response

# Test with your own input
interaction = input("Enter a customer interaction: ")
print("IVR Response:", ivr_system(interaction))
