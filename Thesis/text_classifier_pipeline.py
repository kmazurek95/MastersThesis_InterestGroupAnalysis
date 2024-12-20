import sys
import csv
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import (RandomForestClassifier)
from sklearn.model_selection import train_test_split
from sklearn import metrics

from sklearn.feature_extraction.text import (
    CountVectorizer, TfidfVectorizer)
from sklearn.linear_model import (
    LogisticRegression)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import (
    make_pipeline, Pipeline)
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import joblib
import eli5
from nltk.sentiment import vader
import nltk
from nltk.corpus import stopwords
import nltk
from nltk.tokenize import (TreebankWordTokenizer, 
                           WhitespaceTokenizer)
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import (
    CountVectorizer, TfidfVectorizer)
from nltk.tokenize import word_tokenize
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn import preprocessing 
import json
import os
import sys
import urllib
import urllib.request
import re
import regex
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
import joblib
nltk.download('stopwords')
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from tqdm.auto import tqdm
import time
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.feature_extraction.text import HashingVectorizer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as imblearnPipeline
from itertools import product
from sklearn.metrics import f1_score





data_labeled = pd.read_csv("C:\\Users\\kaleb\\OneDrive\\Desktop\\UVA_RMSS_THESIS_MAZUREK\\Labeled_Training_Data.csv")
column_names = data_labeled.columns.tolist()
print(column_names)

# drop columns except the ones we want to keep
columns_to_drop = set(data_labeled.columns) - {'p1_original', 'uuid_paragraph', 'prominence', '10_or_more_org_mentioned'}
data_labeled.drop(columns_to_drop, axis=1, inplace=True)

def pre_processing (text):
    text = text.lower().replace('\n',' ').strip() #convert all uppercase letters to lowercase, replace line breaks with a single space, and remove all spaces at the beginning or end of the article
    text = re.sub(' +', ' ', text) #regex expression that deletes mutliple spaces and replaces with one space
    text = re.sub(r'[^\w\s]','',text) #regex expression that removes all characters that aren't part of the alphabet 
    
    
    stop_words = set(stopwords.words('english')) 
    tokenized_words = word_tokenize(text) # a list that contains all the words in a given news article   
    
    #processed_sentence = [w for w in tokenized_words if not w in stop_words] # removes all stop words from word_tokens
    

    processed_sentence = [] #initially an empty list
    for w in tokenized_words:   #these three rows remove the stop words from the list tokenized_words
        if w not in stop_words: 
            processed_sentence.append(w) 
    
    text = " ".join(processed_sentence) #filtered_sentence is list of words, but it needs to be in sentence form so we join it
    return text # returns the proccessed text

data_labeled['p1_proccessed'] = data_labeled['p1_original'].apply(pre_processing)

#proccessed text, we can see that the preprocessing was successful
data_labeled.iloc[0][4]

def prominence(prominence):
    if prominence == "TRUE":
        return 1
    elif prominence == "FALSE":
        return 0
    else:
        return prominence

data_labeled["prominence_binary"] = data_labeled["prominence"].apply(prominence)

label_encoder = preprocessing.LabelEncoder() 
data_labeled['prominence_numerical']= label_encoder.fit_transform(data_labeled['prominence_binary']) 

# look and see to what value politics was assign: 0 is no prominence and 1 is prominence
data_labeled.head()

data_labeled['prominence_numerical'].unique()

X_train, X_test, y_train, y_test = train_test_split(data_labeled['p1_proccessed'], 
                                                    data_labeled['prominence_numerical'],
                                                    shuffle = True, 
                                                    test_size=0.2, 
                                                    random_state=13)

#Check that the datatsets are the correct size

print(len(y_train),  len(X_train), len(y_test),  len(X_test))

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, 
    test_size=0.25, random_state= 8)

#Check that the datatsets are the correct size

print(len(y_train),  len(X_train), len(y_test),  len(X_test), len(y_val), len(X_val))



import joblib
from sklearn.naive_bayes import MultinomialNB

class FloatTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X.astype(np.float64)


def evaluate_models(configs, X_train, y_train, X_test, y_test):
    results = {}
    best_parameters = {}

    total_configs = len(configs)
    progress_bar = tqdm(total=total_configs, desc="Grid Search Progress")

    for config in configs:
        model_name, vectorizer, classifier, param_grid = config
        print(f"Performing grid search for model: {model_name}")

        pipeline = Pipeline([
            ('vectorizer', vectorizer),
            ('classifier', classifier)
        ])

        search = GridSearchCV(estimator=pipeline, n_jobs=-1, param_grid=param_grid, scoring="roc_auc", cv=5)
        with tqdm(total=len(X_train)) as pbar:
            search.fit(X_train, y_train)
            pbar.update(len(X_train))

        print(f"Best parameters for {model_name}: {search.best_params_}")
        best_parameters[model_name] = search.best_params_

        pred = search.predict(X_test)
        report = metrics.classification_report(y_test, pred, output_dict=True)
        roc_auc = metrics.roc_auc_score(y_test, pred)
        print(f"ROC AUC: {roc_auc}")
        print(metrics.classification_report(y_test, pred))

        # Save the grid search object to a pickle file
        joblib.dump(search, f"{model_name}_grid_search.pkl")

        # Store the results in a dictionary
        results[model_name] = {
            'best_params': search.best_params_,
            'best_score': search.best_score_,
            'report': report,
            'roc_auc': roc_auc
        }

        progress_bar.update(1)

    # Save the results dictionary to a pickle file
    joblib.dump(results, "grid_search_results.pkl")

    return best_parameters, results

param_grids = {
    'NB': {},  # No parameters for MultinomialNB
    'LogReg': {'classifier__C': [0.1, 1, 10], 'classifier__penalty': ['l1', 'l2'], 'classifier__class_weight': ['balanced']},
    'SVM': {'classifier__C': [0.1, 1, 10], 'classifier__gamma': [1, 0.1, 0.01], 'classifier__kernel': ['linear', 'rbf'], 'classifier__class_weight': ['balanced']},
    'Random': {'classifier__n_estimators': [10, 50, 100, 200], 'classifier__criterion': ['gini', 'entropy'], 'classifier__max_depth': [5, 8, 15, 25, 30], 'classifier__class_weight': ['balanced']},    'SGD': {'classifier__alpha': [0.0001, 0.001, 0.01, 0.1], 'classifier__loss': ['hinge', 'log_loss', 'modified_huber', 'squared_hinge']},  # Replaced 'log' with 'log_loss'
    'XGBoost': {'classifier__learning_rate': [0.1, 0.01, 0.001], 'classifier__max_depth': [3, 5, 7], 'classifier__n_estimators': [100, 200, 300]},
    'LightGBM': {'classifier__learning_rate': [0.1, 0.01, 0.001], 'classifier__max_depth': [3, 5, 7], 'classifier__n_estimators': [100, 200, 300]}
}

# Define the model configurations as a list
configs = [
    ('NB with Count', CountVectorizer(min_df=5, max_df=.5), MultinomialNB(), param_grids['NB']),
    ('NB with TfIdf', TfidfVectorizer(min_df=5, max_df=.5), MultinomialNB(), param_grids['NB']),
    ('LogReg with Count', CountVectorizer(min_df=5, max_df=.5), LogisticRegression(solver='liblinear', class_weight='balanced'), param_grids['LogReg']),
    ('LogReg with TfIdf', TfidfVectorizer(min_df=5, max_df=.5), LogisticRegression(solver='liblinear', class_weight='balanced'), param_grids['LogReg']),
    ('SVM with Count - rbf kernel', CountVectorizer(min_df=5, max_df=.5), SVC(kernel='rbf', class_weight='balanced'), param_grids['SVM']),
    ('SVM with Count - linear kernel', CountVectorizer(min_df=5, max_df=.5), SVC(kernel='linear', class_weight='balanced'), param_grids['SVM']),
    ('SVM with Tfidf - rbf kernel', TfidfVectorizer(min_df=5, max_df=.5), SVC(kernel='rbf', class_weight='balanced'), param_grids['SVM']),
    ('SVM with Tfidf - linear kernel', TfidfVectorizer(min_df=5, max_df=.5), SVC(kernel='linear', class_weight='balanced'), param_grids['SVM']),
    ('Random with Count', CountVectorizer(min_df=5, max_df=.5), RandomForestClassifier(class_weight='balanced'), param_grids['Random']),
    ('Random with Tfidf', TfidfVectorizer(min_df=5, max_df=.5), RandomForestClassifier(class_weight='balanced'), param_grids['Random']),
    ('SGD with Count', CountVectorizer(min_df=5, max_df=.5), SGDClassifier(), param_grids['SGD']),
    ('SGD with TfIdf', TfidfVectorizer(min_df=5, max_df=.5), SGDClassifier(), param_grids['SGD']),
    ('XGBoost with Count', CountVectorizer(min_df=5, max_df=.5), XGBClassifier(), param_grids['XGBoost']),
    ('XGBoost with TfIdf', TfidfVectorizer(min_df=5, max_df=.5), XGBClassifier(), param_grids['XGBoost']),
    ('LightGBM with Count', Pipeline([
        ('vectorizer', CountVectorizer(min_df=5, max_df=.5)),
        ('to_float', FloatTransformer())
    ]), LGBMClassifier(), param_grids['LightGBM']),
    ('LightGBM with TfIdf', TfidfVectorizer(min_df=5, max_df=.5), LGBMClassifier(), param_grids['LightGBM'])
]

# Call the evaluate_models function with the model configurations and datasets
best_parameters, results = evaluate_models(configs, X_train, y_train, X_test, y_test)




'''
Resample your data: As I mentioned in the previous response, you can use techniques such as over-sampling the minority class 
    (e.g., using SMOTE) or under-sampling the majority class to balance your dataset.

Use class weights: Many machine learning algorithms allow you to set class weights, which can make the algorithm pay more
    attention to the minority class during training. For example, in scikit-learn's implementation of SVM, you can set the class_weight parameter to balanced to automatically adjust weights inversely proportional to class frequencies. Given your code, it would look something like this: {'classifier__C': 0.1, 'classifier__gamma': 1, 'classifier__kernel': 'rbf', 'classifier__class_weight': 'balanced'}.

'''

class FloatTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X.astype(np.float64)


def evaluate_models(configs, X_train, y_train, X_test, y_test):
    results = {}
    best_parameters = {}

    total_configs = len(configs)
    progress_bar = tqdm(total=total_configs, desc="Grid Search Progress")

    for config in configs:
        model_name, vectorizer, transformer, classifier, param_grid = config
        print(f"Performing grid search for model: {model_name}")

        # Incorporate SMOTE in the pipeline
        pipeline = imblearnPipeline([
            ('vectorizer', vectorizer),
            ('to_float', transformer),  # add this line
            ('sampling', SMOTE()),
            ('classifier', classifier)
        ])

        # If the model supports class_weight parameter, include it in the grid search
        if 'classifier__class_weight' in classifier.get_params():
            param_grid['classifier__class_weight'] = ['balanced']

        search = GridSearchCV(estimator=pipeline, n_jobs=-1, param_grid=param_grid, scoring="roc_auc", cv=5)

        with tqdm(total=len(X_train)) as pbar:
            search.fit(X_train, y_train)
            pbar.update(len(X_train))

        print(f"Best parameters for {model_name}: {search.best_params_}")
        best_parameters[model_name] = search.best_params_

        pred = search.predict(X_test)
        report = metrics.classification_report(y_test, pred, output_dict=True)
        f1_score = report['1']['f1-score']
        roc_auc = metrics.roc_auc_score(y_test, pred)
        print(f"F1-score: {f1_score}")
        print(f"ROC AUC: {roc_auc}")
        print(metrics.classification_report(y_test, pred))

        # Save the grid search object to a pickle file
        joblib.dump(search, f"{model_name}_grid_search.pkl")

        # Store the results in a dictionary
        results[model_name] = {
            'best_params': search.best_params_,
            'best_score': search.best_score_,
            'report': report,
            'roc_auc': roc_auc
        }

        progress_bar.update(1)

    # Save the results dictionary to a pickle file
    joblib.dump(results, "grid_search_results_PARAMATERS.pkl")

    return best_parameters, results


param_grids = {
    'NB': {},  # No parameters for MultinomialNB
    'LogReg': {'classifier__C': [0.1, 1, 10], 'classifier__penalty': ['l1', 'l2']},
    'SVM': {'classifier__C': [0.1, 1, 10], 'classifier__gamma': [1, 0.1, 0.01], 'classifier__kernel': ['linear', 'rbf']},
    'Random': {'classifier__n_estimators': [10, 50, 100, 200], 'classifier__criterion': ['gini', 'entropy'], 'classifier__max_depth': [5, 8, 15, 25, 30]},
    'SGD': {'classifier__alpha': [0.0001, 0.001, 0.01, 0.1], 'classifier__loss': ['hinge', 'log', 'modified_huber', 'squared_hinge']},  # Replaced 'log' with 'log_loss'
    'XGBoost': {'classifier__learning_rate': [0.1, 0.01, 0.001], 'classifier__max_depth': [3, 5, 7], 'classifier__n_estimators': [100, 200, 300]},
    'LightGBM': {'classifier__learning_rate': [0.1, 0.01, 0.001], 'classifier__max_depth': [3, 5, 7], 'classifier__n_estimators': [100, 200, 300]}
}

# Define the model configurations as a list
configs = [
    ('NB with Count', CountVectorizer(min_df=5, max_df=.5), FloatTransformer(), MultinomialNB(), param_grids['NB']),
    ('NB with TfIdf', TfidfVectorizer(min_df=5, max_df=.5), FloatTransformer(), MultinomialNB(), param_grids['NB']),
    ('LogReg with Count', CountVectorizer(min_df=5, max_df=.5), FloatTransformer(), LogisticRegression(solver='liblinear'), param_grids['LogReg']),
    ('LogReg with TfIdf', TfidfVectorizer(min_df=5, max_df=.5), FloatTransformer(), LogisticRegression(solver='liblinear'), param_grids['LogReg']),
    ('SVM with Count - rbf kernel', CountVectorizer(min_df=5, max_df=.5), FloatTransformer(), SVC(kernel='rbf'), param_grids['SVM']),
    ('SVM with Count - linear kernel', CountVectorizer(min_df=5, max_df=.5), FloatTransformer(), SVC(kernel='linear'), param_grids['SVM']),
    ('SVM with Tfidf - rbf kernel', TfidfVectorizer(min_df=5, max_df=.5), FloatTransformer(), SVC(kernel='rbf'), param_grids['SVM']),
    ('SVM with Tfidf - linear kernel', TfidfVectorizer(min_df=5, max_df=.5), FloatTransformer(), SVC(kernel='linear'), param_grids['SVM']),
    ('Random with Count', CountVectorizer(min_df=5, max_df=.5), FloatTransformer(), RandomForestClassifier(), param_grids['Random']),
    ('Random with Tfidf', TfidfVectorizer(min_df=5, max_df=.5), FloatTransformer(), RandomForestClassifier(), param_grids['Random']),
    ('SGD with Count', CountVectorizer(min_df=5, max_df=.5), FloatTransformer(), SGDClassifier(), param_grids['SGD']),
    ('SGD with TfIdf', TfidfVectorizer(min_df=5, max_df=.5), FloatTransformer(), SGDClassifier(), param_grids['SGD']),
    ('XGBoost with Count', CountVectorizer(min_df=5, max_df=.5), FloatTransformer(), XGBClassifier(), param_grids['XGBoost']),
    ('XGBoost with TfIdf', TfidfVectorizer(min_df=5, max_df=.5), FloatTransformer(), XGBClassifier(), param_grids['XGBoost']),
    ('LightGBM with Count', CountVectorizer(min_df=5, max_df=.5), FloatTransformer(), LGBMClassifier(), param_grids['LightGBM']),
    ('LightGBM with TfIdf', TfidfVectorizer(min_df=5, max_df=.5), FloatTransformer(), LGBMClassifier(), param_grids['LightGBM'])
]

# Call the evaluate_models function with the model configurations and datasets
best_parameters, results = evaluate_models(configs, X_train, y_train, X_test, y_test)



# More comopicated grid search:
'''
This setup includes more parameters and different models, vectorizers according to your request. Please keep in mind that each additional parameter will increase the time required for the GridSearchCV. 
If the number of hyperparameters or the values tested are too large, it might not be practical to perform an exhaustive search. In such cases, RandomizedSearchCV can be a good alternative as it allows to
 specify a number of iterations and will sample parameters from the given distributions, allowing for better control over the time complexity.

'''
# Ensure reproducibility
np.random.seed(0)

# Create a dictionary to store pipelines and parameter grids
model_dict = {

    "Logistic Regression - CountVectorizer": {
        "pipeline": Pipeline(steps=[("vectorizer", CountVectorizer()), ("classifier", LogisticRegression())]),
        "params": {
            "vectorizer__ngram_range": [(1,1), (1,2), (2,2)], 
            "vectorizer__max_df": [0.5, 0.75, 1.0],
            "vectorizer__min_df": [0, 5, 10],
            "vectorizer__stop_words": [None, "english"],
            "classifier__C": [0.1, 1, 10],
            "classifier__penalty": ["l1", "l2"],
            "classifier__fit_intercept": [True, False],
            "classifier__class_weight": [None, "balanced"],
            "classifier__solver": ['liblinear', 'lbfgs']
        }
    },
    "Logistic Regression - TfidfVectorizer": {
        "pipeline": Pipeline(steps=[("vectorizer", TfidfVectorizer()), ("classifier", LogisticRegression())]),
        "params": {
            "vectorizer__ngram_range": [(1,1), (1,2), (2,2)], 
            "vectorizer__max_df": [0.5, 0.75, 1.0],
            "vectorizer__min_df": [0, 5, 10],
            "vectorizer__stop_words": [None, "english"],
            "classifier__C": [0.1, 1, 10],
            "classifier__penalty": ["l1", "l2"],
            "classifier__fit_intercept": [True, False],
            "classifier__class_weight": [None, "balanced"],
            "classifier__solver": ['liblinear', 'lbfgs']
        }
    },
    "Multinomial Naive Bayes - CountVectorizer": {
        "pipeline": Pipeline(steps=[("vectorizer", CountVectorizer()), ("classifier", MultinomialNB())]),
        "params": {
            "vectorizer__ngram_range": [(1,1), (1,2), (2,2)], 
            "vectorizer__max_df": [0.5, 0.75, 1.0],
            "vectorizer__min_df": [0, 5, 10],
            "vectorizer__stop_words": [None, "english"],
            "classifier__alpha": [0.5, 1, 10, 100],
            "classifier__fit_prior": [True, False],
            "classifier__class_prior": [None, [0.5, 0.5]]
        }
    },

    "Multinomial Naive Bayes - TfidfVectorizer": {
        "pipeline": Pipeline(steps=[("vectorizer", TfidfVectorizer()), ("classifier", MultinomialNB())]),
        "params": {
            "vectorizer__ngram_range": [(1,1), (1,2), (2,2)], 
            "vectorizer__max_df": [0.5, 0.75, 1.0],
            "vectorizer__min_df": [0, 5, 10],
            "vectorizer__stop_words": [None, "english"],
            "classifier__alpha": [0.5, 1, 10, 100],
            "classifier__fit_prior": [True, False],
            "classifier__class_prior": [None, [0.5, 0.5]]  # This depends on your class distribution
        }
    },
    "SVM - TfidfVectorizer": {
        "pipeline": Pipeline(steps=[("vectorizer", TfidfVectorizer()), ("classifier", SVC())]),
        "params": {
            "vectorizer__ngram_range": [(1,1), (1,2), (2,2)], 
            "vectorizer__max_df": [0.5, 0.75, 1.0],
            "vectorizer__min_df": [0, 5, 10],
            "vectorizer__stop_words": [None, "english"],
            "classifier__C": [0.1, 1, 10, 100],
            "classifier__kernel": ["linear", "rbf", "poly"],
            "classifier__degree": [3, 4],
            "classifier__gamma": [0.1, 1, "scale", "auto"]
        }
    },
    "Random Forest - CountVectorizer": {
        "pipeline": Pipeline(steps=[("vectorizer", CountVectorizer()), ("classifier", RandomForestClassifier())]),
        "params": {
            "vectorizer__ngram_range": [(1,1), (1,2), (2,2)],
            "vectorizer__max_df": [0.5, 0.75, 1.0],
            "vectorizer__min_df": [0, 5, 10],
            "vectorizer__stop_words": [None, "english"],
            "classifier__n_estimators": [10, 50, 100, 200],
            "classifier__criterion": ["gini", "entropy"],
            "classifier__bootstrap": [True, False],
            "classifier__max_depth": [5, 8, 15, 25, 30],
            "classifier__min_samples_split": [2, 5, 10],
            "classifier__min_samples_leaf": [1, 2, 4],
            "classifier__max_features": ["sqrt", "log2", None]
        }
    },
    "Random Forest - TfidfVectorizer": {
        "pipeline": Pipeline(steps=[("vectorizer", TfidfVectorizer()), ("classifier", RandomForestClassifier())]),
        "params": {
            "vectorizer__ngram_range": [(1,1), (1,2), (2,2)],
            "vectorizer__max_df": [0.5, 0.75, 1.0],
            "vectorizer__min_df": [0, 5, 10],
            "vectorizer__stop_words": [None, "english"],
            "classifier__n_estimators": [10, 50, 100, 200],
            "classifier__criterion": ["gini", "entropy"],
            "classifier__bootstrap": [True, False],
            "classifier__max_depth": [5, 8, 15, 25, 30],
            "classifier__min_samples_split": [2, 5, 10],
            "classifier__min_samples_leaf": [1, 2, 4],
            "classifier__max_features": ["sqrt", "log2", None]
        }
    },
      "SGDClassifier - HashingVectorizer": {
        "pipeline": Pipeline(steps=[("vectorizer", HashingVectorizer()), ("classifier", SGDClassifier())]),
        "params": {
            "vectorizer__ngram_range": [(1,1), (1,2), (2,2)], 
            "vectorizer__non_negative": [True, False],
            "classifier__loss": ["hinge", "log", "modified_huber"],
            "classifier__penalty": ["l2", "l1", "elasticnet"],
            "classifier__alpha": [0.0001, 0.001, 0.01, 0.1]
        }
    },
    "XGBClassifier - TfidfVectorizer": {
        "pipeline": Pipeline(steps=[("vectorizer", TfidfVectorizer()), ("classifier", XGBClassifier())]),
        "params": {
            "vectorizer__ngram_range": [(1,1), (1,2), (2,2)], 
            "vectorizer__max_df": [0.5, 0.75, 1.0],
            "vectorizer__min_df": [0, 5, 10],
            "vectorizer__stop_words": [None, "english"],
            "classifier__n_estimators": [100, 200, 500],
            "classifier__learning_rate": [0.01, 0.1, 0.2],
            "classifier__max_depth": [3, 5, 7]
        }
    },
    "LGBMClassifier - CountVectorizer": {
        "pipeline": Pipeline(steps=[("vectorizer", CountVectorizer()), ("classifier", LGBMClassifier())]),
        "params": {
            "vectorizer__ngram_range": [(1,1), (1,2), (2,2)], 
            "vectorizer__max_df": [0.5, 0.75, 1.0],
            "vectorizer__min_df": [0, 5, 10],
            "vectorizer__stop_words": [None, "english"],
            "classifier__n_estimators": [100, 200, 500],
            "classifier__learning_rate": [0.01, 0.1, 0.2],
            "classifier__max_depth": [-1, 5, 7],
            "classifier__num_leaves": [31, 50, 100]
        }
    }
}

for model_name, model_info in model_dict.items():
    print(f"Performing grid search for model: {model_name}")
    pipeline = model_info["pipeline"]
    param_grid = model_info["params"]

    # Update the pipeline to include class weights and SMOTE
    if model_name.startswith("Logistic Regression") or model_name.startswith("Multinomial Naive Bayes"):
        # Add class weights to the classifier
        pipeline.steps[-1][1].class_weight = ["balanced", None]
    elif model_name.startswith("SVM"):
        # Add class weights to the classifier
        pipeline.steps[-1][1].class_weight = ["balanced", None]
        # Add SMOTE to the pipeline after vectorization
        pipeline.steps.insert(-1, ("smote", SMOTE(random_state=0)))
    
    # Check if the grid search results have already been saved
    try:
        # Load the grid search object from the pickle file
        search = joblib.load(f"{model_name}_grid_search.pkl")
        print("Resuming grid search...")
    except FileNotFoundError:
        search = GridSearchCV(estimator=pipeline, param_grid=param_grid, scoring="f1", cv=5)

    # Calculate the total number of combinations in the parameter grid
    total_combinations = len(list(product(*param_grid.values())))

    with tqdm(total=total_combinations) as pbar:
        # Perform a new grid search
        search.fit(X_train, y_train)
        pbar.update(len(X_train))

    print(f"Best parameters for {model_name}: {search.best_params_}")
    pred = search.predict(X_test)
    report = metrics.classification_report(y_test, pred, output_dict=True)
    f1_score = report['1']['f1-score']
    roc_auc = metrics.roc_auc_score(y_test, pred)
    print(f"F1-score: {f1_score}")
    print(f"ROC AUC: {roc_auc}")
    print(metrics.classification_report(y_test, pred))

    # Save the grid search object to a pickle file
    joblib.dump(search, f"{model_name}_grid_search_HYPERPARAMETERS.pkl")


'''

For MultinomialNB, I added class_prior, but this parameter depends on your class distribution, so adjust it accordingly.

For the SVM, I added a range for the C parameter and included the gamma parameter for tuning.

For RandomForest, I included parameters for min_samples_split, min_samples_leaf, and max_features.

In all models, I also included the stop_words parameter in the vectorizers. If your texts contain many common English words that are not informative about the class, removing them might improve your performance.

As always, remember that this is a starting point and you may need to adjust based on your specific dataset and use case.

'''


best_models = {
    "Logistic Regression - TfidfVectorizer": {
        "pipeline": ImbPipeline(steps=[("vectorizer", TfidfVectorizer()), ("smote", SMOTE()), ("classifier", LogisticRegression())]),
        "params": {
            "vectorizer__ngram_range": [(1,1), (1,2), (2,2)],
            "vectorizer__max_df": [0.5, 0.75, 1.0],
            "vectorizer__min_df": [0, 5, 10],
            "vectorizer__stop_words": [None, "english"],
            "smote__sampling_strategy": ["minority", "not majority", 0.5],
            "classifier__C": [0.01, 0.1, 1, 10, 100],
            "classifier__penalty": ["l1", "l2"],
            "classifier__fit_intercept": [True, False],
            "classifier__class_weight": [None, "balanced"],
            "classifier__solver": ['liblinear', 'lbfgs', 'newton-cg', 'sag', 'saga'],
            "classifier__dual": [True, False]
        }
    },
    "Multinomial Naive Bayes - TfidfVectorizer": {
        "pipeline": ImbPipeline(steps=[("vectorizer", TfidfVectorizer()), ("smote", SMOTE()), ("classifier", MultinomialNB())]),
        "params": {
            "vectorizer__ngram_range": [(1,1), (1,2), (2,2)],
            "vectorizer__max_df": [0.5, 0.75, 1.0],
            "vectorizer__min_df": [0, 5, 10],
            "vectorizer__stop_words": [None, "english"],
            "smote__sampling_strategy": ["minority", "not majority", 0.5],
            "classifier__alpha": [0.1, 0.5, 1, 10, 100],
            "classifier__fit_prior": [True, False],
            "classifier__class_prior": [None, [0.5, 0.5]]
        }
    },
    "SGDClassifier - HashingVectorizer": {
        "pipeline": ImbPipeline(steps=[("vectorizer", HashingVectorizer()), ("smote", SMOTE()), ("classifier", SGDClassifier())]),
        "params": {
            "vectorizer__ngram_range": [(1,1), (1,2), (2,2)],
            "vectorizer__non_negative": [True, False],
            "smote__sampling_strategy": ["minority", "not majority", 0.5],
            "classifier__loss": ["hinge", "log", "modified_huber", "squared_hinge"],
            "classifier__penalty": ["l2", "l1", "elasticnet"],
            "classifier__alpha": [0.00001, 0.0001, 0.001, 0.01, 0.1, 1],
            "classifier__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
            "classifier__learning_rate": ['constant', 'optimal', 'invscaling', 'adaptive'],
            "classifier__eta0": [0.01, 0.1, 1, 10]
        }
    }
}

for model_name, model_data in best_models.items():
    pipeline = model_data["pipeline"]
    param_grid = model_data["params"]

    print(f"Performing grid search for model: {model_name}")

    # Calculate the total number of combinations in the parameter grid
    total_combinations = np.prod([len(v) for v in param_grid.values()])

    # Create a GridSearchCV object
    search = GridSearchCV(pipeline, param_grid, scoring="f1", cv=5, n_jobs=-1)

    # Fit the model (Without progress bar as it's not supported)
    search.fit(X_train, y_train)

    print(f"Best parameters for {model_name}: {search.best_params_}")

    pred = search.predict(X_test)
    f1_score = metrics.f1_score(y_test, pred)
    roc_auc = metrics.roc_auc_score(y_test, pred)

    print(f"F1-score: {f1_score}")
    print(f"ROC AUC: {roc_auc}")
    print(metrics.classification_report(y_test, pred))

    # Save the grid search object to a pickle file
    joblib.dump(search, f"{model_name}_grid_search.pkl")

    print("----------------------------------")


#VALIDATION:


# Define the vectorizer
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_df=0.75, min_df=10, stop_words=None)

# Define the classifiers with the best hyperparameters
classifier_lr = LogisticRegression(C=1, penalty='l2', class_weight="balanced")
classifier_nb = MultinomialNB()
classifier_sgd = SGDClassifier(alpha=0.01, loss='squared_hinge')

# Define the pipelines with SMOTE
pipeline_lr = ImbPipeline(steps=[("vectorizer", vectorizer), ("smote", SMOTE()), ("classifier", classifier_lr)])
pipeline_nb = ImbPipeline(steps=[("vectorizer", vectorizer), ("smote", SMOTE()), ("classifier", classifier_nb)])
pipeline_sgd = ImbPipeline(steps=[("vectorizer", vectorizer), ("smote", SMOTE()), ("classifier", classifier_sgd)])

# Fit the pipelines to the training data
pipeline_lr.fit(X_train, y_train)
pipeline_nb.fit(X_train, y_train)
pipeline_sgd.fit(X_train, y_train)

# Make predictions on the validation data
y_pred_lr = pipeline_lr.predict(X_val)
y_pred_nb = pipeline_nb.predict(X_val)
y_pred_sgd = pipeline_sgd.predict(X_val)

# Print the classification report for the predictions
rep_lr = classification_report(y_val, y_pred_lr)
rep_nb = classification_report(y_val, y_pred_nb)
rep_sgd = classification_report(y_val, y_pred_sgd)

print("Logistic Regression with TfIdfVectorizer and SMOTE:")
print(rep_lr)

print("Naive Bayes with TfIdfVectorizer and SMOTE:")
print(rep_nb)

print("SGDClassifier with TfIdfVectorizer and SMOTE:")
print(rep_sgd)

# Calculate and print the F1-scores
f1_lr = f1_score(y_val, y_pred_lr)
f1_nb = f1_score(y_val, y_pred_nb)
f1_sgd = f1_score(y_val, y_pred_sgd)

print(f"F1-score for Logistic Regression: {f1_lr}")
print(f"F1-score for Naive Bayes: {f1_nb}")
print(f"F1-score for SGDClassifier: {f1_sgd}")

# Calculate and print the ROC AUC scores
roc_auc_lr = roc_auc_score(y_val, y_pred_lr)
roc_auc_nb = roc_auc_score(y_val, y_pred_nb)
roc_auc_sgd = roc_auc_score(y_val, y_pred_sgd)

print(f"ROC AUC Score for Logistic Regression: {roc_auc_lr}")
print(f"ROC AUC Score for Naive Bayes: {roc_auc_nb}")
print(f"ROC AUC Score for SGDClassifier: {roc_auc_sgd}")


# Load and Proccess Unseen Data



# write a function to retrieve unlabeled data
def get_unlabeled_data(fn="C:\\Users\\kaleb\\OneDrive\\Desktop\\UVA_RMSS_THESIS_MAZUREK\\Unlabeled_Data.csv"):
    # read the csv file into a pandas dataframe
    df = pd.read_csv(fn)
    print("CSV file has been read.")

    # preprocess the text in the 'p1_original' column
    df['p1_original'] = df['p1_original'].apply(pre_processing)
    print("Text in 'p1_original' column has been preprocessed.")

    # convert the preprocessed text and uuids to lists
    text = df['p1_original'].tolist()
    uuids = df['uuid_paragraph'].tolist()
    print("Converted text and uuids to lists.")

    print(f"Number of elements in text: {len(text)}")
    print(f"Number of elements in uuids: {len(uuids)}")

    return text, uuids

# call the function and unpack the results
unlabeled, uuids = get_unlabeled_data()

# print the first few elements of each list
print("First few elements of 'unlabeled':", unlabeled[:5])
print("First few elements of 'uuids':", uuids[:5])



# Specify our final model with the correct hyper-parameters
vectorizer_tfidf_final = TfidfVectorizer(ngram_range=(1, 2), max_df=0.75, min_df=10, stop_words=None)
classifier_nb_final = MultinomialNB()

# Define the pipeline with SMOTE
pipeline_nb_final = ImbPipeline(steps=[("vectorizer", vectorizer_tfidf_final), ("smote", SMOTE()), ("classifier", classifier_nb_final)])

# Fit the pipeline to the training data
pipeline_nb_final.fit(X_train, y_train)

# Predict labels for the unlabeled data
y_pred_final = pipeline_nb_final.predict(unlabeled)

# Convert y_pred_final to a list of Python ints
y_pred_final_list = [int(y) for y in y_pred_final]



# Save the classifier and vectorizer that we used on the unseen data
with open("C:\\Users\\kaleb\\OneDrive\\Desktop\\UVA_RMSS_THESIS_MAZUREK\\vectorizer_tfidf.pkl", mode="wb") as f:
    pickle.dump(vectorizer_tfidf_final, f)
with open("C:\\Users\\kaleb\\OneDrive\\Desktop\\UVA_RMSS_THESIS_MAZUREK\\classifier_nb.pkl", mode="wb") as f:
    pickle.dump(classifier_nb_final, f)

# Make sure that we can open it
with open("C:\\Users\\kaleb\\OneDrive\\Desktop\\UVA_RMSS_THESIS_MAZUREK\\vectorizer_tfidf.pkl", mode="rb") as f:
    vectorizer_tfidf_final = pickle.load(f)
with open("C:\\Users\\kaleb\\OneDrive\\Desktop\\UVA_RMSS_THESIS_MAZUREK\\classifier_nb.pkl", mode="rb") as f:
    classifier_nb_final = pickle.load(f)



class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        return super(NumpyEncoder, self).default(obj)

# Save predictions as JSON
predictions_dict = dict(zip(uuids, y_pred_final))
with open("C:\\Users\\kaleb\\OneDrive\\Desktop\\UVA_RMSS_THESIS_MAZUREK\\predictions.json", mode="w") as fo:
    json.dump(predictions_dict, fo, cls=NumpyEncoder)

# Save predictions as CSV
with open("C:\\Users\\kaleb\\OneDrive\\Desktop\\UVA_RMSS_THESIS_MAZUREK\\predictions.csv", encoding="utf8", mode='w', newline='') as fo:
    writer = csv.writer(fo)
    writer.writerows(zip(uuids, unlabeled, y_pred_final))

