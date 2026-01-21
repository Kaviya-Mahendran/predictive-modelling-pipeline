import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def train_models(X_train, y_train):
    """
    Train baseline and non-linear models.
    Returns trained models.
    """

    # -------------------------------
    # Baseline model: Logistic Regression
    # -------------------------------
    logistic_model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )
    logistic_model.fit(X_train, y_train)

    # -------------------------------
    # Non-linear model: Random Forest
    # -------------------------------
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        random_state=42,
        class_weight="balanced",
    )
    rf_model.fit(X_train, y_train)

    return logistic_model, rf_model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate a model and return metrics + ROC curve data.
    """

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }

    fpr, tpr, _ = roc_curve(y_test, y_proba)

    return metrics, fpr, tpr, y_proba


def save_models(logistic_model, rf_model):
    """
    Persist trained models to disk.
    """

    joblib.dump(logistic_model, "models/logistic_model.pkl")
    joblib.dump(rf_model, "models/random_forest_model.pkl")
