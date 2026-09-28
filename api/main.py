from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import joblib
import pandas as pd


app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predict whether a customer is likely to churn.",
    version="1.0.0"
)

# Serve CSS and other static files
app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static"
)

# Load HTML templates
templates = Jinja2Templates(directory="templates")

# Load the trained ML model
model = joblib.load("model/churn_model.pkl")


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


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def predict(data: CustomerData):
    customer_data = data.model_dump()
    input_data = pd.DataFrame([customer_data])

    probability = float(model.predict_proba(input_data)[0][1])

    # Keep the same threshold used during model evaluation
    threshold = 0.32
    prediction = int(probability >= threshold)

    result = "Churn" if prediction == 1 else "No Churn"

    return {
        "churn_probability": round(probability, 4),
        "prediction": prediction,
        "result": result
    }