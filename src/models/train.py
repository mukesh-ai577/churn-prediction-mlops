import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline


def train_model():

    # Load processed data
    data_path = "data/processed/churn_processed.csv"
    df = pd.read_csv(data_path)

    # Separate features and target
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Identify feature types
    numerical_features = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object"]
    ).columns.tolist()

    # Numerical preprocessing
    numerical_transformer = StandardScaler()

    # Categorical preprocessing
    categorical_transformer = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    )

    # Preprocessor
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

    # Final model
    model = HistGradientBoostingClassifier(
        max_iter=250,
        learning_rate=0.01602,
        max_leaf_nodes=23,
        max_depth=4,
        min_samples_leaf=34,
        l2_regularization=8.3614,
        random_state=42
    )

    # Complete pipeline
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    # Train
    pipeline.fit(X_train, y_train)

    # Create artifacts directory
    os.makedirs("artifacts", exist_ok=True)

    # Save complete pipeline
    joblib.dump(
        pipeline,
        "artifacts/churn_model.pkl"
    )

    print("Model training completed successfully.")
    print("Model saved at: artifacts/churn_model.pkl")


if __name__ == "__main__":
    train_model()