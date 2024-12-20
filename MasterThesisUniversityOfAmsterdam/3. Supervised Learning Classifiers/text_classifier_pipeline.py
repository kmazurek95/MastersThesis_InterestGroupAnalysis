import pandas as pd
import numpy as np
import re
import json
import pickle
import csv
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
import joblib

# Text Preprocessing Function
def preprocess_text(text):
    """
    Cleans and tokenizes text by:
    - Removing punctuation and converting to lowercase.
    - Removing stop words (common, unimportant words).
    """
    text = re.sub(r'[^\w\s]', '', text.lower().strip())  # Remove punctuation and lowercase
    stop_words = set(stopwords.words('english'))  # Load English stop words
    tokens = [word for word in word_tokenize(text) if word not in stop_words]  # Tokenize and remove stop words
    return " ".join(tokens)

# Load Labeled Dataset
def load_labeled_data(file_path):
    """
    Reads and preprocesses labeled data from a CSV file.
    - Processes the text column and encodes target labels.
    """
    df = pd.read_csv(file_path).dropna()  # Read CSV and drop missing values
    df['p1_processed'] = df['p1_original'].apply(preprocess_text)  # Preprocess text column
    le = LabelEncoder()  # Initialize label encoder
    df['prominence_numerical'] = le.fit_transform(df['prominence'].astype(str))  # Encode target labels
    return df

# Load Unlabeled Dataset
def load_unlabeled_data(file_path):
    """
    Reads and preprocesses unlabeled data from a CSV file.
    - Processes the text column only.
    """
    df = pd.read_csv(file_path)  # Read CSV
    df['p1_processed'] = df['p1_original'].apply(preprocess_text)  # Preprocess text column
    return df

# Split Data into Train, Validation, and Test Sets
def split_data(df):
    """
    Splits the dataset into training, validation, and test subsets.
    """
    X = df['p1_processed']  # Feature: processed text
    y = df['prominence_numerical']  # Target: encoded labels
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)  # Train/Test split
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)  # Val/Test split
    return X_train, X_val, X_test, y_train, y_val, y_test

# Model Configurations
configs = [
    ('Naive Bayes + CountVectorizer', CountVectorizer(min_df=5, max_df=0.5), MultinomialNB(), {}),
    ('Logistic Regression + TfidfVectorizer', TfidfVectorizer(min_df=5, max_df=0.5), LogisticRegression(solver='liblinear'), 
     {'classifier__C': [0.1, 1, 10], 'classifier__penalty': ['l1', 'l2']}),
    ('SVM + CountVectorizer', CountVectorizer(min_df=5, max_df=0.5), SVC(), 
     {'classifier__C': [0.1, 1, 10], 'classifier__gamma': [1, 0.1, 0.01], 'classifier__kernel': ['linear', 'rbf']}),
    ('Random Forest + TfidfVectorizer', TfidfVectorizer(min_df=5, max_df=0.5), RandomForestClassifier(), 
     {'classifier__n_estimators': [10, 50, 100], 'classifier__max_depth': [5, 15, None]})
]

# Run Grid Search for Each Model Configuration
def run_grid_search(X_train, y_train, configs):
    """
    Finds the best model and hyperparameters using Grid Search.
    """
    best_estimators = []
    for name, vectorizer, classifier, param_grid in configs:
        print(f"Running Grid Search for {name}...")
        pipeline = Pipeline([
            ('vectorizer', vectorizer),
            ('classifier', classifier)
        ])
        grid_search = GridSearchCV(pipeline, param_grid, scoring='accuracy', cv=5, n_jobs=-1)
        grid_search.fit(X_train, y_train)
        print(f"Best parameters for {name}: {grid_search.best_params_}")
        best_estimators.append((name, grid_search.best_estimator_))
    return best_estimators

# Evaluate Models
def evaluate_models(best_estimators, X_val, y_val, X_test, y_test):
    """
    Tests the performance of models on validation and test datasets.
    """
    for name, model in best_estimators:
        print(f"Evaluating {name}...")
        val_preds = model.predict(X_val)
        test_preds = model.predict(X_test)
        print(f"Validation Results for {name}:\n{classification_report(y_val, val_preds)}")
        print(f"Test Results for {name}:\n{classification_report(y_test, test_preds)}\n")

# Label Unlabeled Data
def label_unlabeled(best_model, unlabeled_df):
    """
    Uses the best model to predict labels for unlabeled data.
    """
    X_unlabeled = unlabeled_df['p1_processed']
    predictions = best_model.predict(X_unlabeled)
    unlabeled_df['predicted_label'] = predictions
    return unlabeled_df

# Save Predictions
def save_predictions(predictions, output_path):
    """
    Exports labeled predictions to CSV and JSON files.
    """
    predictions.to_csv(output_path + ".csv", index=False)
    with open(output_path + ".json", 'w') as f:
        json.dump(predictions.to_dict(orient='records'), f, indent=4)

# Main Workflow
if __name__ == "__main__":
    labeled_path = "C://Users//kaleb//OneDrive//Desktop//Classifier//Labeled_Data__Version_____1.csv"
    unlabeled_path = "C://Users//kaleb//OneDrive//Desktop//Classifier//Un-labeled_data.csv"

    # Load and preprocess data
    labeled_data = load_labeled_data(labeled_path)
    unlabeled_data = load_unlabeled_data(unlabeled_path)

    # Split labeled data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(labeled_data)

    # Run grid search to find the best models
    best_models = run_grid_search(X_train, y_train, configs)

    # Evaluate models on validation and test sets
    evaluate_models(best_models, X_val, y_val, X_test, y_test)

    # Label unlabeled data using the best model
    best_model = best_models[0][1]  # Select the first model as an example
    labeled_unlabeled_data = label_unlabeled(best_model, unlabeled_data)

    # Save the labeled data
    save_predictions(labeled_unlabeled_data, "labeled_unlabeled_data")

    print("Process completed successfully!")
