# Random Forest Text Classifier

A machine learning project to classify text messages as spam or ham using Random Forest and TF-IDF vectorization. This project demonstrates text preprocessing, model training, and evaluation, making it suitable for spam detection, SMS classification, or chatbot intent recognition.

---

## Project Overview

In this project:

1. Text preprocessing – Convert text messages into numerical vectors using TF-IDF, capturing the importance of words.
2. Model training – Train a Random Forest classifier, which combines multiple decision trees for accurate and robust predictions.
3. Evaluation – Measure performance using accuracy, precision, recall, and F1-score.

> ⚠️ Note: This project uses a small sample dataset, so accuracy may be low (~0.5). For real-world applications, larger datasets are recommended.

---

## Required Libraries

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
```

---

## Sample Dataset

```python
data = {
    'text': [
        "Congratulations! You have won $1000 cash prize",
        "Hello, are we still meeting tomorrow?",
        "Get cheap loans now!!!",
        "Don't forget to bring the documents",
        "Free entry in 2 a weekly competition to win tickets"
    ],
    'label': ['spam', 'ham', 'spam', 'ham', 'spam']
}
```

This dataset contains a mix of spam and ham messages to demonstrate the workflow.

---

## Workflow / Code

```python
# Load dataset
df = pd.DataFrame(data)

# Features and labels
X = df['text']
y = df['label']

# Train-test split (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# TF-IDF vectorization
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train Random Forest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_vec, y_train)

# Predict on test set
y_pred = model.predict(X_test_vec)

# Evaluate model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
```

---

## Evaluation

The model is evaluated using:

* Accuracy – Overall correctness of predictions
* Precision & Recall – Effectiveness in detecting spam vs ham
* F1-score – Balance between precision and recall

> ⚠️ Example result on this small dataset:

```
Accuracy: 0.5
Classification Report:
              precision    recall  f1-score   support
         ham       0.50      1.00      0.67         1
        spam       0.00      0.00      0.00         1
```

> With larger datasets, model performance is expected to improve significantly.

---

## Insights

* TF-IDF captures word importance in text messages.
* Random Forest reduces overfitting compared to a single decision tree.
* Small datasets limit model performance; this pipeline is a foundation for real-world spam detection systems.

