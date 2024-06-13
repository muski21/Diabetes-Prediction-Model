# Diabetes-Prediction-Model

# Diabetes Prediction Model using Machine Learning 

## Table of Contents

1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Data Preprocessing](#data-preprocessing)
4. [Exploratory Data Analysis (EDA)](#eda)
5. [Model Building](#model-building)
6. [Model Evaluation](#model-evaluation)
7. [Conclusion](#conclusion)

## Introduction

In this project, we aim to predict the onset of diabetes in patients using machine learning techniques. We explore different algorithms such as Random Forest, Decision Tree, and XGBoost to build and evaluate our models.

## Dataset

The dataset used for this project is sourced from Kaggle and consists of several health parameters of individuals, including glucose level, blood pressure, skin thickness, insulin level, BMI, age, and diabetes pedigree function.

## Data Preprocessing

Before building the models, we performed the following preprocessing steps:

- Handling missing values by imputing means and medians.
- Data normalization using StandardScaler.
- Splitting the dataset into training and testing sets.

## Exploratory Data Analysis (EDA)

We conducted exploratory data analysis to gain insights into the dataset. This included:

- Descriptive statistics of the dataset.
- Visualizations such as histograms and correlation matrices to understand the relationships between variables.

## Model Building

We trained the following machine learning models:

- **Random Forest Classifier**: Achieved an accuracy of 75.98% on the test set.
- **Decision Tree Classifier**: Achieved an accuracy of 70.47% on the test set.
- **XGBoost Classifier**:Achieved an accuracy of 72.83% on the test set .
- **SVM**-Achieved an accuracy of 74.80% on the test set.

## Model Evaluation

We evaluated each model using metrics such as accuracy, precision, recall, and F1-score. Confusion matrices were also generated to visualize performance on test data.

## Conclusion

The Random Forest classifier performed the best among the models evaluated. Future work could involve further hyperparameter tuning and exploring additional algorithms to improve prediction accuracy.

## Files in the Repository

- **diabetes_prediction.pdf**: HTML file converted in pdf format,containing the Python code for data preprocessing, model building, and evaluation.
- **README.md**: This file, providing an overview of the project and instructions for use.


## Dependencies

- Python 3.x
- Jupyter Notebook
- Libraries: numpy, pandas, matplotlib, seaborn, scikit-learn, mlxtend, xgboost
