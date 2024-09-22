
# Autism Prediction Using Machine Learning

This project aims to develop a machine learning model to predict the likelihood of autism spectrum disorder (ASD) based on various features. It employs different machine learning algorithms, including Logistic Regression, Decision Tree Classifier, Random Forest Classifier, k-Nearest Neighbors (KNN), and Support Vector Machine (SVM). The model performance is evaluated using accuracy, precision, recall, and F1-score.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Data Preprocessing](#data-preprocessing)
- [Models Used](#models-used)
- [Model Evaluation](#model-evaluation)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Results](#results)
- [Learning Curves](#learning-curves)
- [Conclusion](#conclusion)
- [References](#references)

## Introduction

Autism Spectrum Disorder (ASD) is a developmental disorder that affects communication and behavior. Early prediction and diagnosis of ASD can significantly help in the treatment and management of the condition. This project explores different machine learning models to predict autism based on given features.

## Dataset

The dataset used in this project contains features such as age, gender, communication skills, and other behavioral aspects that help predict the possibility of autism in an individual. The dataset can be sourced from open repositories or medical datasets related to autism.

You can also explore the implementation on Kaggle: [Autism Prediction on Kaggle](https://www.kaggle.com/code/sagarakhade/autismpredictioninchildren)

## Requirements

To run this project, you need to install the following Python libraries:

- `scikit-learn`
- `numpy`
- `pandas`
- `matplotlib`

You can install these libraries using:
```bash
pip install scikit-learn numpy pandas matplotlib
```

## Data Preprocessing

The data preprocessing steps include:
- Handling missing values
- Encoding categorical features
- Splitting the data into training and testing sets
- Standardizing the feature values using `StandardScaler`

## Models Used

The following machine learning models were implemented for autism prediction:
1. **Logistic Regression**
2. **Decision Tree Classifier**
3. **Random Forest Classifier**
4. **k-Nearest Neighbors (KNN)**
5. **Support Vector Machine (SVM)**

## Model Evaluation

The models were evaluated using the following metrics:
- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**
- **Confusion Matrix**

These metrics provide insights into how well the models perform in predicting autism.

## Hyperparameter Tuning

GridSearchCV was used for hyperparameter tuning to improve model performance. The hyperparameters for models like Decision Tree, Random Forest, and SVM were fine-tuned using cross-validation to achieve optimal results.

## Results

| Model                   | Accuracy | Precision | Recall  | F1 Score |
|-------------------------|----------|-----------|---------|----------|
| Logistic Regression     | 0.946    | 0.974     | 0.946   | 0.960    |
| Decision Tree Classifier| 0.916    | 0.946     | 0.930   | 0.938    |
| Random Forest Classifier| 0.971    | 0.959     | 1.000   | 0.979    |
| k-Nearest Neighbors (KNN)| 0.910   | 0.953     | 0.913   | 0.932    |
| Support Vector Machine (SVM) | 0.940| 0.966    | 0.946   | 0.956    |


## Conclusion

Based on the evaluation metrics, the **Random Forest Classifier** performed the best in predicting autism with an accuracy score of 0.971, followed by Logistic Regression and SVM. Further improvements can be made by trying other advanced models or using feature engineering techniques.


