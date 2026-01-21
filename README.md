**Predictive Modelling Pipeline for Customer / Donor Behaviour**


**1. Overview**

Predictive models are often built as isolated experiments, but their real value comes from how reliably they can be trained, evaluated, interpreted, and reused as part of a wider analytics system.

This repository implements an end-to-end predictive modelling pipeline designed to estimate the likelihood of customer or donor churn using privacy-safe, engineered features. The focus is not only on model accuracy, but also on comparative evaluation, interpretability, and operational robustness.

The project demonstrates how structured features can be transformed into predictive signals, how multiple models can be compared objectively, and how interpretability techniques such as SHAP can be applied responsibly to understand model behaviour.

Although implementations vary across organisations, these principles apply broadly to most data analytics environments.

**2. Architecture Overview**

High-level flow:

Feature Dataset
      ↓
Preprocessing & Encoding
      ↓
Train / Test Split
      ↓
Model Training (Baseline + Tree-based)
      ↓
Evaluation & Comparison
      ↓
Interpretability (SHAP)
      ↓
Persisted Outputs & Models


Key design choices:

Clear separation between preprocessing, modelling, and orchestration

Reproducible artefacts saved at each stage

Interpretability treated as a first-class step, not an afterthought

**3. Pipeline Design (Step-by-Step)**
Step 1: Feature Ingestion

The pipeline ingests a privacy-aware feature set (sample_features.csv) containing behavioural and engagement signals such as recency, frequency, and aggregated activity metrics. No directly identifiable personal data is used.

Step 2: Preprocessing

In preprocessing.py, the data is:

cleaned and validated

categoricals encoded

split into training and test sets

returned in a format suitable for both linear and non-linear models

This ensures consistent feature handling across all models.

Step 3: Model Training

Two models are trained in parallel:

Logistic Regression as a transparent baseline

Random Forest to capture non-linear behaviour patterns

This deliberate comparison avoids relying on a single modelling approach.

Step 4: Evaluation

Each model is evaluated using:

accuracy

precision

recall

ROC-AUC

A side-by-side comparison is saved as a CSV artefact to support evidence-based model selection.

Step 5: Interpretability (SHAP)
To understand why predictions are made, SHAP values are computed for the tree-based model.
Due to known stability issues in SHAP plotting APIs, feature importance is manually aggregated using mean absolute SHAP values, ensuring robustness and reproducibility.

This provides a clear view of which behavioural features most strongly influence predicted outcomes.

Step 6: Persistence

The pipeline saves:

trained models

ROC curve visualisation

SHAP feature importance plot

model comparison metrics

This makes the pipeline reusable and auditable.

**4. Code Highlights**
Model Comparison
logistic_model, rf_model = train_models(X_train, y_train)

ROC Curve Generation
plt.plot(log_fpr, log_tpr, label="Logistic Regression")
plt.plot(rf_fpr, rf_tpr, label="Random Forest")

Manual SHAP Aggregation
shap_importance = pd.DataFrame({
    "feature": shap_feature_names,
    "mean_abs_shap": abs(shap_values).mean(axis=0)
}).sort_values("mean_abs_shap", ascending=False)


This approach avoids brittle dependencies on plotting internals while retaining interpretability.

**5. Outputs**

The pipeline produces the following artefacts:

outputs/model_comparison.csv
Quantitative comparison of model performance.

outputs/roc_curve.png
Visual comparison of classification performance.

outputs/shap_summary.png
Feature importance derived from SHAP values.

models/*.pkl
Persisted, reusable trained models.

These outputs are designed to support both technical review and stakeholder discussion.
## Architecture Overview

![Architecture Diagram](diagrams/architecture.png)

## Pipeline Flow

![Pipeline Overview](diagrams/pipeline_overview.png)

**6. Why This Matters**
Predictive modelling is only valuable when it is trustworthy, interpretable, and reusable.

This pipeline demonstrates:

how advanced models can be evaluated against simpler baselines

how interpretability can be preserved even in non-linear models

how analytics workflows can be structured as long-term systems rather than one-off analyses

The design is suitable for organisations that need to make data-driven decisions responsibly, particularly where explainability and governance are as important as accuracy.

**7. Limitations & Ethics**

The dataset is synthetic and intended for demonstration purposes.

Model performance depends on feature quality and label definition.

SHAP interpretations describe associations, not causation.

Care should be taken to avoid reinforcing bias when deploying predictive scores.

Privacy-safe features and explicit interpretability steps are used to reduce ethical and operational risk.

**8. Reflection & Future Enhancements**

Building this pipeline reinforced the importance of model comparison, interpretability, and robustness over raw performance metrics alone.

Future enhancements could include:

time-based validation for behavioural drift

calibration of predicted probabilities

integration with automated scheduling

extension to survival or uplift modelling

The same architectural principles would apply at larger scales.

**9. How to Reproduce**
pip install -r requirements.txt
python scripts/pipeline.py


All outputs will be generated in the outputs/ and models/ directories.

