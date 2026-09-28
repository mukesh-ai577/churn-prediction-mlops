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

## MLflow

MLflow is used for:

- Experiment tracking
- Model logging
- Model Registry
- Model versioning
- Model alias management

Registered model:

`churn-prediction-model`

Production alias:

`champion`

The FastAPI application loads the model from the MLflow Model Registry.

## FastAPI

The model is exposed through a REST API using FastAPI.

API endpoint:

`POST /predict`

Swagger documentation:

`http://127.0.0.1:8000/docs`

Example response:

```json
{
  "churn_probability": 0.2946,
  "prediction": 0,
  "result": "No Churn"
}











#####  How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/mukesh-ai577/churn-prediction-mlops.git
cd churn-prediction-mlops
```

### 2. Create and Activate Virtual Environment

```bash
conda create -n churn-env python=3.11 -y
conda activate churn-env
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Start MLflow Server

Start the MLflow tracking server:

```bash
mlflow server --host 0.0.0.0 --port 5000 --workers 1 --serve-artifacts --artifacts-destination ./mlartifacts --allowed-hosts "host.docker.internal:5000"
```

MLflow UI:

```text
http://127.0.0.1:5000
```

### 5. Build Docker Image

```bash
docker build -t churn-prediction-api .
```

### 6. Run the Docker Container

```bash
docker run -d -p 8000:8000 --name churn-api -e MLFLOW_TRACKING_URI=http://host.docker.internal:5000 churn-prediction-api
```

Check whether the container is running:

```bash
docker ps
```

### 7. Open FastAPI Swagger UI

Open the following URL in your browser:

```text
http://127.0.0.1:8000/docs
```

### 8. Test the Prediction API

In Swagger UI:

1. Open `POST /predict`
2. Click **Try it out**
3. Enter customer information
4. Click **Execute**
5. The API returns the churn probability and prediction.

Example response:

```json
{
  "churn_probability": 0.2946,
  "prediction": 0,
  "result": "No Churn"
}
```

### 9. Stop the Docker Container

```bash
docker stop churn-api
```

To remove the container:

```bash
docker rm churn-api
```

### 10. Stop MLflow

Press:

```text
CTRL + C
```

in the terminal where MLflow is running.
