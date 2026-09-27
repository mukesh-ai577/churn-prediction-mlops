import pandas as pd
import joblib


def predict_churn(customer_data):

    # Load trained model
    model = joblib.load(
        "artifacts/churn_model.pkl"
    )

    # Convert input dictionary into DataFrame
    input_data = pd.DataFrame([customer_data])

    # Predict probability
    probability = model.predict_proba(input_data)[0][1]

    # Same threshold used during evaluation
    threshold = 0.32

    # Final prediction
    prediction = int(probability >= threshold)

    if prediction == 1:
        result = "Churn"
    else:
        result = "No Churn"

    return {
        "churn_probability": round(float(probability), 4),
        "prediction": prediction,
        "result": result
    }


if __name__ == "__main__":

    # Example customer
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

    result = predict_churn(customer)

    print("\n===== CHURN PREDICTION =====")
    print(f"Churn Probability : {result['churn_probability']}")
    print(f"Prediction        : {result['prediction']}")
    print(f"Result            : {result['result']}")
