import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


def evaluate_model():

    # Load processed data
    data_path = "data/processed/churn_processed.csv"
    df = pd.read_csv(data_path)

    # Separate features and target
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    # Same split as training
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Load trained pipeline
    model = joblib.load(
        "artifacts/churn_model.pkl"
    )

    # Probability prediction
    y_prob = model.predict_proba(X_test)[:, 1]

    # CV-selected threshold
    threshold = 0.32

    # Final prediction
    y_pred = (y_prob >= threshold).astype(int)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    print("\n===== MODEL EVALUATION =====")

    print(f"Threshold : {threshold}")
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"ROC-AUC   : {roc_auc:.4f}")

    print("\n===== CONFUSION MATRIX =====")
    print(confusion_matrix(y_test, y_pred))

    print("\n===== CLASSIFICATION REPORT =====")
    print(
        classification_report(
            y_test,
            y_pred
        )
    )


if __name__ == "__main__":
    evaluate_model()