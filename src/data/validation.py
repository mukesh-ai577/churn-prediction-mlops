# Ab hum data validation banayenge. Iska kaam hoga check karna ki:

# file exist karti hai ya nahi
# required columns hain ya nahi
# missing values kitni hain
# target Churn valid hai ya nahi
# dataset empty nahi hai



import pandas as pd
import os


REQUIRED_COLUMNS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "Churn"
]


def validate_data(file_path):

    # Check file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Data file not found: {file_path}"
        )

    # Load data
    df = pd.read_csv(file_path)

    # Check dataset is not empty
    if df.empty:
        raise ValueError("Dataset is empty.")

    # Check required columns
    missing_columns = [
        column for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    # Check target values
    valid_target_values = {0, 1}

    if not set(df["Churn"].dropna().unique()).issubset(
        valid_target_values
    ):
        raise ValueError(
            "Churn column contains invalid values."
        )

    print("Data validation completed successfully.")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")
    print(f"Missing values: {df.isnull().sum().sum()}")


if __name__ == "__main__":

    file_path = "data/processed/churn_processed.csv"

    validate_data(file_path)