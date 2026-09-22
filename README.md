# Telecom Customer Churn Prediction

##  Project Overview

This project predicts whether a telecom customer is likely to churn based on their service usage, plans, billing information, and customer service interactions.

The project includes data analysis, preprocessing, machine learning model training, evaluation, and a Streamlit web application for making predictions.

##  Objective

The main objective is to build a machine learning model that can identify customers who are likely to leave the telecom service.

This can help businesses identify potential churn customers and take appropriate retention actions.

##  Dataset

The dataset contains **667 customer records and 20 columns**.

Important features include:

- Account length
- International plan
- Voice mail plan
- Number of voicemail messages
- Day, evening, and night usage
- International usage
- Customer service calls
- Churn

The target variable is:

- `Churn` — whether the customer left the service

The dataset contains 572 non-churned customers and 95 churned customers, so the target classes are imbalanced.

##  Exploratory Data Analysis

Some observations from the analysis:

- Customers with an international plan showed a higher churn rate than customers without one.
- Churned customers had a higher average number of customer service calls.
- Total day minutes and total day charge showed noticeable positive associations with churn.
- Customer service calls were also associated with churn.

These observations represent associations in the dataset and do not imply that the features directly cause churn.

##  Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Streamlit

##  Machine Learning Workflow

The project follows these steps:

1. Data loading
2. Exploratory Data Analysis
3. Feature selection
4. Train-test split
5. Numerical feature preprocessing
6. Categorical feature encoding
7. Logistic Regression baseline
8. Random Forest model
9. Model evaluation
10. Model saving using Joblib
11. Streamlit deployment

##  Models Used

### Logistic Regression

Logistic Regression was used as a baseline model.

### Random Forest

Random Forest was then trained to capture more complex relationships in the data.

Class weighting was used because the churn classes were imbalanced.

##  Model Performance

The Random Forest model achieved the following results on the test set:

| Metric | Score |
|---|---:|
| Accuracy | 92.5% |
| Precision | 100.0% |
| Recall | 47.4% |
| F1 Score | 64.3% |
| ROC-AUC | 91.4% |

Since the dataset is imbalanced, multiple evaluation metrics were considered instead of relying only on accuracy.

##  Important Features

The Random Forest model identified the following features among its highest-importance features:

- Total day charge
- Total day minutes
- Customer service calls
- Total evening minutes
- Total evening charge
- Total international minutes

Feature importance indicates which variables the model relied on more heavily; it does not establish causation.

##  Streamlit Application

A Streamlit application is included in the project.

Users can enter customer information such as:

- Account length
- Service plans
- Day/evening/night usage
- International usage
- Customer service calls

The application then provides:

- Churn prediction
- Churn probability

## Future Improvements

The project can be improved further by:

- Performing hyperparameter tuning to improve model performance.
- Using cross-validation for more reliable model evaluation.
- Adjusting the prediction threshold to improve churn recall.
- Creating additional features from customer usage and billing data.
- Comparing Random Forest with other machine learning algorithms such as XGBoost.
- Using a larger and more recent telecom customer dataset.
- Adding visual dashboards to the Streamlit application.
- Adding customer-level insights to explain why a customer is predicted to churn.
- Deploying the Streamlit application online for easier access.

## Author

Nandnee Kapse

