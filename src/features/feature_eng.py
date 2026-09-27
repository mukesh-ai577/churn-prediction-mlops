import pandas as pd
import os
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def create_preprocessor(df):

    # Separate features and target
    X = df.drop("Churn", axis=1)

    # Numerical columns
    numerical_features = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    # Categorical columns
    categorical_features = X.select_dtypes(
        include=["object"]
    ).columns.tolist()

    print("Numerical features:")
    print(numerical_features)

    print("\nCategorical features:")
    print(categorical_features)

    # Numerical preprocessing
    numerical_transformer = StandardScaler()

    # Categorical preprocessing
    categorical_transformer = OneHotEncoder(
        handle_unknown="ignore"
    )

    # Combine preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numerical_transformer,
                numerical_features
            ),
            (
                "cat",
                categorical_transformer,
                categorical_features
            )
        ]
    )

    return preprocessor


if __name__ == "__main__":

    # Load processed data
    data_path = "data/processed/churn_processed.csv"

    df = pd.read_csv(data_path)

    # Create preprocessor
    preprocessor = create_preprocessor(df)

    # Create directory
    os.makedirs("artifacts", exist_ok=True)

    # Save preprocessor
    joblib.dump(
        preprocessor,
        "artifacts/preprocessor.pkl"
    )

    print("\nFeature engineering setup completed.")
    print("Preprocessor saved at: artifacts/preprocessor.pkl")