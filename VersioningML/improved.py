import json
import pickle
from datetime import datetime
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
from mlflow.models.signature import infer_signature
from nltk.stem import SnowballStemmer

# Download Russian stopwords
nltk.download('stopwords')
russian_stopwords = set(stopwords.words('russian'))

# Препроцессинг текста
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    words = text.split()
    words = [word for word in words if word not in russian_stopwords]
    return ' '.join(words)

# Load the data
df = pd.read_json('kinopoisk.jsonl')
print('shape', df.shape)
print('попуски', df.isna().sum())
print('grade3', df['grade3'].value_counts())

# Preprocess the content
df['processed_content'] = df['content'].apply(preprocess_text)

# Удаление строк с меткой 'Neutral'
df = df[df['grade3'] != 'Neutral']
# Преобразование значений в 'grade2'
df['grade2'] = df['grade3'].map({'Bad': 0, 'Good': 1})
print('grade2', df['grade2'].value_counts())
target_dict = {'Bad': 0, 'Good': 1}

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    df['processed_content'], df['grade2'], test_size=0.2, random_state=42
)

# Create bag of words representation
vectorizer = CountVectorizer()
X_train_bow = vectorizer.fit_transform(X_train)
X_test_bow = vectorizer.transform(X_test)

# Train the classifier
model = MultinomialNB()
model.fit(X_train_bow, y_train)

# Make predictions
y_pred = model.predict(X_test_bow)

# Evaluate the model
report = classification_report(y_test, y_pred, output_dict=True)
print(classification_report(y_test, y_pred))

# Получаем текущую дату в формате YYYY-MM-DD
current_date = datetime.now().strftime("%Y-%m-%d")
filename = f"models/model_{current_date}.pkl"

# Сохраняем модель в файл
with open(filename, 'wb') as file:
    pickle.dump(model, file)

# Установка URI трекинга
mlflow.set_tracking_uri("http://localhost:5000")

# Start MLflow run
with mlflow.start_run():
    # Log parameters
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("random_state", 42)
    mlflow.log_param("model_type", "MultinomialNB")
    mlflow.log_param("vectorizer_type", "CountVectorizer")
    mlflow.log_param("target", df['grade3'].value_counts())
    mlflow.log_param("target_dict", target_dict)

    # Log metrics
    mlflow.log_metric("accuracy", report['accuracy'])
    for label, metrics in report.items():
        if isinstance(metrics, dict):
            for metric_name, value in metrics.items():
                mlflow.log_metric(f"{label}_{metric_name}", value)
    
    # Определение сигнатуры модели
    signature = infer_signature(X_test_bow, y_pred)
    
    # Логирование модели с сигнатурой и примером входных данных
    mlflow.sklearn.log_model(
        model,
        "model",
        signature=signature,
    )

    # Логируем файл модели
    mlflow.log_artifact(filename)  # Логируем артефакт с моделью

# Function to classify new reviews
def classify_review(review_text):
    processed_text = preprocess_text(review_text)
    bow_representation = vectorizer.transform([processed_text])
    prediction = model.predict(bow_representation)
    return prediction[0]

# Example usage
# example_review = "Ваш текст рецензии на русском языке"
# print(f"Классификация: {classify_review(example_review)}")
