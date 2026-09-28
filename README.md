# Customer Churn Prediction — MLOps

An end-to-end customer churn prediction system built using Machine Learning, MLflow, FastAPI, and Docker.

## Project Overview

This project predicts whether a telecom customer is likely to churn.

The project follows an MLOps workflow:

- Data preprocessing
- Exploratory Data Analysis
- Model training
- Model evaluation
- MLflow experiment tracking
- MLflow Model Registry
- FastAPI REST API
- Docker containerization

## Machine Learning

The following models were experimented with:

- Logistic Regression
- Random Forest
- Gradient Boosting
- XGBoost
- HistGradientBoosting

The final model is **HistGradientBoosting**.

### Model Performance

| Metric | Score |
|---|---:|
| Accuracy | 77.08% |
| Precision | 55.03% |
| Recall | 74.60% |
| F1 Score | 63.34% |
| ROC-AUC | 84.63% |

The prediction threshold is **0.32**.


## 🚀 Live Demo

* **Live API:** https://churn-prediction-mlops-41f9.onrender.com/
* **Swagger API Docs:** https://churn-prediction-mlops-41f9.onrender.com/docs
* **GitHub Repository:** https://github.com/mukesh-ai577/churn-prediction-mlops

