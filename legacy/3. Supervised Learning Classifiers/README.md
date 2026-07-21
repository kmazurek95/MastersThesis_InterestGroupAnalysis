# 3. Supervised Learning Classifiers

## Overview
The Supervised Learning Classifiers folder contains a single script, `text_classifier_pipeline.py`, designed to perform text classification using a variety of machine learning models. The script includes preprocessing, model training, hyperparameter optimization, and evaluation of labeled data. It predicts labels for unlabeled datasets and saves the results for further analysis.

---

## Script overview

### `text_classifier_pipeline.py`
Purpose: End-to-end pipeline for supervised text classification.

#### Key features:
1. Text preprocessing:
   - Cleans and tokenizes text by removing punctuation, converting to lowercase, and removing stopwords.
   - Prepares input data for machine learning models.

2. Data loading:
   - Reads labeled and unlabeled datasets from CSV files.
   - Encodes target labels using `LabelEncoder`.

3. Data splitting:
   - Splits labeled data into training, validation, and test sets.

4. Model training and hyperparameter tuning:
   - Trains multiple classifiers using Grid Search for hyperparameter optimization:
     - Naive Bayes
     - Logistic Regression
     - Support Vector Machines (SVM)
     - Random Forests
   - Evaluates models based on accuracy.

5. Evaluation:
   - Produces classification reports for validation and test datasets.
   - Compares model performance to identify the best classifier.

6. Labeling unlabeled data:
   - Uses the best-trained model to predict labels for an unlabeled dataset.

7. Export results:
   - Saves labeled predictions in both CSV and JSON formats for downstream analysis.

---

## Workflow

### 1. Preprocessing:
- Preprocess text in labeled and unlabeled datasets:
  ```python
  preprocess_text(text)
  ```

### 2. Data loading:
- Load datasets using:
  ```python
  load_labeled_data(file_path)
  load_unlabeled_data(file_path)
  ```

### 3. Data splitting:
- Split labeled data into training, validation, and test sets:
  ```python
  split_data(df)
  ```

### 4. Model training:
- Configure and train classifiers using Grid Search:
  ```python
  run_grid_search(X_train, y_train, configs)
  ```

### 5. Evaluation:
- Evaluate models on validation and test datasets:
  ```python
  evaluate_models(best_estimators, X_val, y_val, X_test, y_test)
  ```

### 6. Labeling unlabeled data:
- Label unlabeled data with the best model:
  ```python
  label_unlabeled(best_model, unlabeled_df)
  ```

### 7. Save results:
- Save labeled predictions:
  ```python
  save_predictions(predictions, output_path)
  ```

---

## Prerequisites

1. Python libraries:
   Install necessary dependencies using:
   ```bash
   pip install -r requirements.txt
   ```
   Key libraries:
   - `pandas`
   - `numpy`
   - `nltk`
   - `sklearn`
   - `tqdm`

2. Dataset structure:
   - Labeled Dataset:
     - Columns: `p1_original` (text), `prominence` (target labels).
   - Unlabeled Dataset:
     - Columns: `p1_original` (text).

3. File paths:
   Update paths for labeled and unlabeled datasets in the script:
   ```python
   labeled_path = "path/to/labeled_data.csv"
   unlabeled_path = "path/to/unlabeled_data.csv"
   ```

4. Stopwords:
   Ensure NLTK stopwords are downloaded:
   ```python
   nltk.download('stopwords')
   nltk.download('punkt')
   ```

---

## Outputs

1. Model performance:
   - Classification reports for validation and test datasets.

2. Labeled predictions:
   - CSV and JSON files containing predicted labels for unlabeled data:
     - Example: `labeled_unlabeled_data.csv` and `labeled_unlabeled_data.json`.

---

## Usage

1. Clone the repository and navigate to the script directory:
   ```bash
   git clone https://github.com/username/MastersThesis_InterestGroupAnalysis.git
   cd MasterThesisUniversityOfAmsterdam/3. Supervised Learning Classifiers/
   ```

2. Run the script:
   ```bash
   python text_classifier_pipeline.py
   ```

3. Review outputs in the specified output directory.

---

## Contact
For questions or issues, contact Kaleb Mazurek at kalebmazurek@gmail.com
