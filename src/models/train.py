import pandas as pd
import joblib
import os
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def train_model():

    # =========================
    # Load processed data
    # =========================
    data_path = "data/processed/churn_processed.csv"
    df = pd.read_csv(data_path)

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    # =========================
    # Train-test split
    # =========================
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # =========================
    # Feature identification
    # =========================
    numerical_features = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object"]
    ).columns.tolist()

    # =========================
    # Preprocessing
    # =========================
    numerical_transformer = StandardScaler()

    categorical_transformer = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    # =========================
    # Model
    # =========================
    model = HistGradientBoostingClassifier(
        max_iter=250,
        learning_rate=0.01602,
        max_leaf_nodes=23,
        max_depth=4,
        min_samples_leaf=34,
        l2_regularization=8.3614,
        random_state=42
    )

    # =========================
    # Pipeline
    # =========================
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    # =========================
    # MLflow experiment
    # =========================
    mlflow.set_experiment("Churn Prediction")

    with mlflow.start_run():

        # Train
        pipeline.fit(X_train, y_train)

        # Predictions
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        threshold = 0.32
        y_pred = (y_prob >= threshold).astype(int)

        # =========================
        # Metrics
        # =========================
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)

        # =========================
        # Log parameters
        # =========================
        mlflow.log_param("model", "HistGradientBoostingClassifier")
        mlflow.log_param("max_iter", 250)
        mlflow.log_param("learning_rate", 0.01602)
        mlflow.log_param("max_leaf_nodes", 23)
        mlflow.log_param("max_depth", 4)
        mlflow.log_param("min_samples_leaf", 34)
        mlflow.log_param("l2_regularization", 8.3614)
        mlflow.log_param("threshold", threshold)

        # =========================
        # Log metrics
        # =========================
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc_auc)

        # =========================
        # Log model
        # =========================
        mlflow.sklearn.log_model(pipeline, name="churn_model",skops_trusted_types=["sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor"])

        # =========================
        # Save local model
        # =========================
        os.makedirs("artifacts", exist_ok=True)

        joblib.dump(
            pipeline,
            "artifacts/churn_model.pkl"
        )

        print("\n===== MODEL TRAINING =====")
        print(f"Accuracy  : {accuracy:.4f}")
        print(f"Precision : {precision:.4f}")
        print(f"Recall    : {recall:.4f}")
        print(f"F1 Score  : {f1:.4f}")
        print(f"ROC-AUC   : {roc_auc:.4f}")

        print("\nModel saved at:")
        print("artifacts/churn_model.pkl")

        print("\nMLflow run completed successfully.")


if __name__ == "__main__":
    train_model()