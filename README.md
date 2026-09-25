# Predictive Modelling Pipeline

> A reproducible end-to-end classification workflow covering feature preparation, model comparison, evaluation, SHAP interpretability and model persistence.

## Problem

This project demonstrates how behavioural features can be transformed into predictive signals and evaluated as a reusable analytics workflow.

The emphasis is broader than accuracy:

**reproducibility → evaluation → interpretability → operational reuse**

## Pipeline architecture

```text
Feature Dataset
      ↓
Validation & Preprocessing
      ↓
Train / Test Split
      ↓
Baseline Model ───────┐
Tree-based Model ─────┤
                      ↓
             Comparative Evaluation
                      ↓
                SHAP Analysis
                      ↓
       Metrics / Plots / Saved Models
```

## Models

- Logistic Regression as a transparent baseline
- Random Forest for non-linear relationships

Evaluation includes accuracy, precision, recall and ROC-AUC.

## Explainability

SHAP analysis is used to investigate which features contribute most strongly to predictions. These explanations describe model behaviour and should not be interpreted as causal evidence.

## Outputs

```text
outputs/
├── model_comparison.csv
├── roc_curve.png
└── shap_summary.png

models/
└── *.pkl
```

## Reproduce

```bash
pip install -r requirements.txt
python scripts/pipeline.py
```

## Evaluation discipline

The repository does not present placeholder metrics as results. Re-run the pipeline and record the current metrics from the actual dataset and code version.

| Model | Accuracy | Precision | Recall | ROC-AUC |
|---|---:|---:|---:|---:|
| Logistic Regression | Run pipeline | Run pipeline | Run pipeline | Run pipeline |
| Random Forest | Run pipeline | Run pipeline | Run pipeline | Run pipeline |

## Limitations

- Demonstration dataset rather than production data
- Performance depends on feature and label quality
- SHAP is not causal inference
- Real deployment would require calibration, monitoring, drift detection and governance

## Future roadmap

- Time-based validation
- Probability calibration
- Cross-validation
- Automated tests
- Data drift monitoring
- Scheduled execution
- Survival or uplift modelling

**Focus:** predictive analytics · Python · scikit-learn · SHAP · reproducible ML