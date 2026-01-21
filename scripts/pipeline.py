from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import shap

from preprocessing import load_and_prepare_data
from modelling import train_models, evaluate_model, save_models


def run_pipeline():
    print("Starting predictive modelling pipeline...")

    # -------------------------------
    # Paths
    # -------------------------------
    DATA_PATH = "data/raw/sample_features.csv"
    OUTPUTS_PATH = Path("outputs")
    MODELS_PATH = Path("models")

    OUTPUTS_PATH.mkdir(parents=True, exist_ok=True)
    MODELS_PATH.mkdir(parents=True, exist_ok=True)

    # -------------------------------
    # Load & preprocess data
    # -------------------------------
    print("Loading and preprocessing data...")
    X_train, X_test, y_train, y_test, feature_names = load_and_prepare_data(
        DATA_PATH
    )

    # -------------------------------
    # Train models
    # -------------------------------
    print("Training models...")
    logistic_model, rf_model = train_models(X_train, y_train)

    # -------------------------------
    # Evaluate models
    # -------------------------------
    print("Evaluating models...")
    log_metrics, log_fpr, log_tpr, log_proba = evaluate_model(
        logistic_model, X_test, y_test
    )
    rf_metrics, rf_fpr, rf_tpr, rf_proba = evaluate_model(
        rf_model, X_test, y_test
    )

    # -------------------------------
    # Save metrics comparison
    # -------------------------------
    comparison_df = pd.DataFrame(
        [log_metrics, rf_metrics],
        index=["Logistic Regression", "Random Forest"],
    )
    comparison_df.to_csv(OUTPUTS_PATH / "model_comparison.csv")

    print("Model comparison saved.")

    # -------------------------------
    # ROC Curve
    # -------------------------------
    plt.figure(figsize=(6, 5))
    plt.plot(log_fpr, log_tpr, label="Logistic Regression")
    plt.plot(rf_fpr, rf_tpr, label="Random Forest")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUTS_PATH / "roc_curve.png")
    plt.close()

    print("ROC curve saved.")

    # -------------------------------
    # SHAP Interpretation (Tree model)
    # -------------------------------
    print("Generating SHAP explanations...")
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_test)

    shap.summary_plot(
        shap_values[1],
        X_test,
        feature_names=feature_names,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(OUTPUTS_PATH / "shap_summary.png")
    plt.close()

    print("SHAP summary plot saved.")

    # -------------------------------
    # Save models
    # -------------------------------
    save_models(logistic_model, rf_model)
    print("Models saved.")

    print("Predictive modelling pipeline completed successfully.")


if __name__ == "__main__":
    run_pipeline()
