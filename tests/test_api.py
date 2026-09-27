from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


def test_home():

    response = client.get("/")

    assert response.status_code == 200

    assert response.json() == {
        "message": "Customer Churn Prediction API is running"
    }


def test_prediction():

    customer = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 5,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 85.5,
        "TotalCharges": 427.5
    }

    response = client.post(
        "/predict",
        json=customer
    )

    assert response.status_code == 200

    result = response.json()

    assert "churn_probability" in result
    assert "prediction" in result
    assert "result" in result

    assert 0 <= result["churn_probability"] <= 1
    assert result["prediction"] in [0, 1]
    assert result["result"] in ["Churn", "No Churn"]