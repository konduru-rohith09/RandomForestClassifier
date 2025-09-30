Random Forest Text Classifier

A machine learning project to classify text messages as spam or ham using Random Forest and TF-IDF vectorization. This project demonstrates how to preprocess text, train a classifier, and evaluate its performance, making it ideal for spam detection or general text classification tasks.

Project Overview

In this project:

Text preprocessing: Convert text messages into numerical vectors using TF-IDF, which captures the importance of words in each message.

Model training: Train a Random Forest classifier, which combines multiple decision trees for accurate and robust classification.

Evaluation: Use metrics like accuracy, precision, recall, and F1-score to measure model performance.

This project can be extended for real-world applications such as email spam detection, SMS classification, or chatbot intent recognition.

Required Libraries:
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

Sample Dataset:
data = {
    'text': [
       "Congratulations! You have won $1000 cash prize",
        "Hello, are we still meeting tomorrow?",
        "Get cheap loans now!!!",
        "Don't forget to bring the documents",
        "Free entry in 2 a weekly competition to win tickets"],
    'label': ['spam', 'ham', 'spam', 'ham', 'spam']
}
This dataset contains a mix of spam and ham messages to demonstrate the workflow.

Workflow / Usage

Load the dataset as a Pandas DataFrame.

Split data into training and testing sets (70% train, 30% test).

Convert text to numerical features using TF-IDF vectorization.

Train a Random Forest classifier on the training set.

Predict on the test set and evaluate performance.

# Load dataset
df = pd.DataFrame(data)

# Features and labels
X = df['text']
y = df['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# TF-IDF vectorization
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train Random Forest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_vec, y_train)

# Predict and evaluate
y_pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

Evaluation

The model is evaluated using:

Accuracy – overall correctness of predictions

Precision & Recall – effectiveness of detecting spam vs ham

F1-score – balance between precision and recall

this project getting 0.5 accuracy because of we have small data here.

