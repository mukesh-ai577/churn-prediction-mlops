@echo off
title Churn Prediction MLOps

echo ==========================================
echo   Starting Churn Prediction MLOps Project
echo ==========================================

echo.
echo [1/2] Starting MLflow Server...
start "MLflow Server" cmd /k "conda activate churn-env && cd /d C:\Users\mukes\OneDrive\PROJECT\Churn Prediction && mlflow server --host 0.0.0.0 --port 5000 --workers 1 --serve-artifacts --artifacts-destination ./mlartifacts --allowed-hosts "host.docker.internal:5000"

timeout /t 5 /nobreak >nul

echo.
echo [2/2] Starting Docker API...

docker start churn-api >nul 2>&1

if errorlevel 1 (
    docker run -d -p 8000:8000 --name churn-api -e MLFLOW_TRACKING_URI=http://host.docker.internal:5000 churn-prediction-api
)

timeout /t 5 /nobreak >nul

echo.
echo ==========================================
echo   Project Started Successfully!
echo ==========================================
echo.
echo MLflow:  http://127.0.0.1:5000
echo FastAPI: http://127.0.0.1:8000/docs
echo.

start http://127.0.0.1:5000
start http://127.0.0.1:8000/docs

pause