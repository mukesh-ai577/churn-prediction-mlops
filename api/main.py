from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request


app = FastAPI(
    title="Customer Churn Prediction API",
    description="API for predicting customer churn",
    version="1.0.0"
)
templates = Jinja2Templates(directory="templates")


# Load trained model
model = joblib.load("model/churn_model.pkl")


# Input schema
class CustomerData(BaseModel):

    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html"
    )


@app.post("/predict")
def predict(data: CustomerData):

    customer_data = data.model_dump()

    input_data = pd.DataFrame([customer_data])

    probability = model.predict_proba(input_data)[0][1]

    threshold = 0.32

    prediction = int(probability >= threshold)

    result = "Churn" if prediction == 1 else "No Churn"

    return {
        "churn_probability": round(float(probability), 4),
        "prediction": prediction,
        "result": result
    }