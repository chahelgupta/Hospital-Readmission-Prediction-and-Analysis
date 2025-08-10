# Hospital Readmission Prediction using Machine Learning

## Overview

This project aims to predict hospital readmissions for diabetic patients using supervised machine learning models. The goal is to assist healthcare systems in identifying high-risk patients before discharge, enabling better resource allocation and proactive care.

## Tech Stack

- **Language**: Python
- **Libraries**: 
  - Pandas
  - NumPy
  - Scikit-learn
  - Matplotlib
  - Seaborn
  - Imbalanced-learn (SMOTE)
- **Models Used**: 
  - Logistic Regression
  - Random Forest
  - Support Vector Machine (SVM)
  - K-Nearest Neighbors (KNN)
- **Tools**: 
  - Jupyter Notebook
  - Joblib (for model serialization)

## ML Workflow

### 1. Data Preprocessing
- Handled missing values, outliers, and categorical encodings.
- Converted age and diagnosis codes into clinically meaningful brackets.
- Applied SMOTE to address class imbalance.

### 2. Feature Engineering
- Domain-driven feature selection: lab test counts, medications, diagnosis categories.
- Encoded ordinal and binary clinical test results (e.g., A1C test, glucose levels).

### 3. Model Training & Evaluation
- Logistic Regression, Random Forest, Support Vector Machine (SVM) (final best-performing model), K-Nearest Neighbors (KNN)
- Compared classifiers using accuracy, precision, recall, and F1-score.
- Best performance achieved with **SVM + SMOTE** (Accuracy: 61.16%).
- Evaluated using confusion matrices, ROC-AUC, and statistical tests.

### 4. Model Persistence
- Serialized the entire pipeline (model, scaler, encoders) using Joblib for deployment.

## Results

- **Top Predictors**: Number of lab tests, hospital stay duration, and number of medications.
- **Best Model**: Support Vector Machine (SVM) with SMOTE balancing.
- **Balanced Error Distribution**: Suitable for healthcare use due to even false positive/negative rates.

## Authors

 [@AmishiDesai04](https://www.github.com/AmishiDesai04) [@chahelgupta](https://www.github.com/chahelgupta) 
