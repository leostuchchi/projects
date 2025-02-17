import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import nltk
from nltk.corpus import stopwords
import re
import mlflow
import mlflow.sklearn

# Download Russian stopwords
nltk.download('stopwords')
russian_stopwords = set(stopwords.words('russian'))

# Load the JSONL file
def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return pd.DataFrame(data)

# Preprocess text
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Remove stopwords
    words = text.split()
    words = [word for word in words if word not in russian_stopwords]
    return ' '.join(words)

# Load the data
#df = load_jsonl('kinopoisk.jsonl')
df = pd.read_json('kinopoisk.jsonl')

# Preprocess the content
df['processed_content'] = df['content'].apply(preprocess_text)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    df['processed_content'], df['grade3'], test_size=0.2, random_state=42
)

# Create bag of words representation
vectorizer = CountVectorizer()
X_train_bow = vectorizer.fit_transform(X_train)
X_test_bow = vectorizer.transform(X_test)

# Train the classifier
clf = MultinomialNB()
clf.fit(X_train_bow, y_train)

# Make predictions
y_pred = clf.predict(X_test_bow)

# Evaluate the model
report = classification_report(y_test, y_pred, output_dict=True)
print(classification_report(y_test, y_pred))


# Установка URI трекинга
mlflow.set_tracking_uri("http://localhost:5000")  # Убедитесь, что сервер MLflow запущен на этом адресе
# Установка URI трекинга
#mlflow.set_tracking_uri("file:./mlruns")

# Start MLflow run
with mlflow.start_run():
    # Log parameters
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("random_state", 42)
    mlflow.log_param("model_type", "MultinomialNB")
    
    # Log metrics
    mlflow.log_metric("accuracy", report['accuracy'])
    for label, metrics in report.items():
        if isinstance(metrics, dict):
            for metric_name, value in metrics.items():
                mlflow.log_metric(f"{label}_{metric_name}", value)
    
    # Log model
    mlflow.sklearn.log_model(clf, "model")
    
    # Log vectorizer
    mlflow.sklearn.log_model(vectorizer, "vectorizer")

# Function to classify new reviews
def classify_review(review_text):
    processed_text = preprocess_text(review_text)
    bow_representation = vectorizer.transform([processed_text])
    prediction = clf.predict(bow_representation)
    return prediction[0]

# Example usage
# example_review = "Ваш текст рецензии на русском языке"
# print(f"Классификация: {classify_review(example_review)}")