import json
import pickle
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_auc_score
import nltk
from nltk.corpus import stopwords
import re
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from nltk.stem import SnowballStemmer
import argparse
import os
import joblib

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

def main(data_path, test_size, random_state, vectorizer_type, model_type):
    # Load the data
    df = pd.read_json(data_path)
    print('shape', df.shape)
    print('попуски', df.isna().sum())
    print('grade3', df['grade3'].value_counts())

    # Preprocess the content
    df['processed_content'] = df['content'].apply(preprocess_text)

    # Удаление строк с меткой 'Neutral'
    df = df[df['grade3'] != 'Neutral']
    df['grade2'] = df['grade3'].map({'Bad': 0, 'Good': 1})
    print('grade2', df['grade2'].value_counts())
    target_dict = {'Bad': 0, 'Good': 1}

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        df['processed_content'], df['grade2'], test_size=test_size, random_state=random_state
    )

    # Create bag of words representation based on vectorizer type
    if vectorizer_type == 'count':
        vectorizer = CountVectorizer()
    elif vectorizer_type == 'tfidf':
        vectorizer = TfidfVectorizer()
    else:
        raise ValueError("Invalid vectorizer type. Choose 'count' or 'tfidf'.")

    X_train_bow = vectorizer.fit_transform(X_train)
    X_test_bow = vectorizer.transform(X_test)

    # Train the classifier based on model type
    if model_type == 'logistic':
        model = LogisticRegression(class_weight='balanced')
    elif model_type == 'naive_bayes':
        model = MultinomialNB()
    else:
        raise ValueError("Invalid model type. Choose 'logistic' or 'naive_bayes'.")

    model.fit(X_train_bow, y_train)

    # Make predictions
    y_pred = model.predict(X_test_bow)

    # Evaluate the model
    report = classification_report(y_test, y_pred)
    print(report)

    # Calculate additional metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    try:
        roc_auc = roc_auc_score(y_test, y_pred)
    except ValueError:
        roc_auc = np.nan
        print("ROC AUC score could not be calculated due to only one class present in y_true.")

    current_date = datetime.now().strftime("%Y-%m-%d")
    
    # Create the "models" directory if it doesn't exist
    if not os.path.exists("models"):
        os.makedirs("models")

    # Установка URI трекинга
    mlflow.set_tracking_uri("http://localhost:5000")

    # Start MLflow run
    with mlflow.start_run() as run:
        # Log parameters
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("vectorizer_type", vectorizer_type)
        mlflow.log_param("target", df['grade3'].value_counts().to_dict())
        mlflow.log_param("target_dict", target_dict)

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)
        if not np.isnan(roc_auc):
            mlflow.log_metric("roc_auc", roc_auc)

        # Log vectorizer
        mlflow.sklearn.log_model(vectorizer, "vectorizer")

        # Log the model
        signature = infer_signature(X_test_bow, y_pred)
        
        # Generate an example input for logging
        #input_example = X_train_bow[0:1].toarray()  # Convert sparse matrix to dense array

        mlflow.sklearn.log_model(
            model,
            "model",
            signature=signature,
            #input_example=input_example,
        )

        # Save the model
        model_path = "model.pkl"
        joblib.dump(model, model_path)
        mlflow.log_artifact(model_path)

        # Save the vectorizer
        vectorizer_path = "vectorizer.pkl"
        joblib.dump(vectorizer, vectorizer_path)
        mlflow.log_artifact(vectorizer_path)

        run_id = run.info.run_uuid
        experiment_id = run.info.experiment_id

        print(f"Logged model with run_id {run_id}")
        print(f"Logged to experiment_id {experiment_id}")

# Function to classify new reviews
def classify_review(review_text, vectorizer, model):
    processed_text = preprocess_text(review_text)
    bow_representation = vectorizer.transform([processed_text])
    prediction = model.predict(bow_representation)
    return prediction[0]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="kinopoisk.jsonl")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--vectorizer_type", type=str, choices=['count', 'tfidf'], default='count', help="Choose 'count' or 'tfidf' for vectorization.")
    parser.add_argument("--model_type", type=str, choices=['logistic', 'naive_bayes'], default='logistic', help="Choose 'logistic' or 'naive_bayes' for modeling.")
    args = parser.parse_args()

    main(args.data_path, args.test_size, args.random_state, args.vectorizer_type, args.model_type)

