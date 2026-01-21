from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import shap

from preprocessing import load_and_prepare_data
from modelling import train_models, evaluate_model, save_models
from utils import ensure_directory, save_metadata


def run_pipeline():
    print("Starting predictive modelling pipeline...")

    # -------------------------------
    # Paths
    # -------------------------------
    DATA_PATH = "data/raw/sample_features.csv"
    OUTPUTS_PATH = Path("outputs")
    MODELS_PATH = Path("models")

    ensure_directory(OUTPUTS_PATH)
    ensure_directory(MODELS_PATH)


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
    log_metrics, log_fpr, log_tpr, _ = evaluate_model(
        logistic_model, X_test, y_test
    )
    rf_metrics, rf_fpr, rf_tpr, _ = evaluate_model(
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

    # Convert to numeric numpy arrays
    X_test_shap = X_test.astype(float)

    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_test_shap)[1]  # churn = 1

    # Compute mean absolute SHAP values
    shap_feature_count = shap_values.shape[1]
    shap_feature_names = X_test_shap.columns[:shap_feature_count]

    shap_importance = (
    pd.DataFrame({
        "feature": shap_feature_names,
        "mean_abs_shap": abs(shap_values).mean(axis=0)
    })
    .sort_values("mean_abs_shap", ascending=False)
    )


    # Plot SHAP importance manually (robust)
    plt.figure(figsize=(8, 5))
    plt.barh(
    shap_importance["feature"],
    shap_importance["mean_abs_shap"]
    )
    plt.gca().invert_yaxis()
    plt.xlabel("Mean |SHAP value|")
    plt.title("Feature Importance (SHAP)")
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
