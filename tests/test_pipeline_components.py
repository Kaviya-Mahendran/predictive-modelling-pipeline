import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from modelling import evaluate_model, train_models


def test_model_training_and_evaluation():
    X = pd.DataFrame(
        {
            "feature_a": [0, 1, 0, 1, 0, 1, 0, 1],
            "feature_b": [1, 1, 0, 0, 1, 0, 0, 1],
        }
    )
    y = pd.Series([0, 1, 0, 1, 0, 1, 0, 1])

    logistic, forest = train_models(X, y)

    for model in (logistic, forest):
        metrics, fpr, tpr, probabilities = evaluate_model(model, X, y)
        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["precision"] <= 1
        assert 0 <= metrics["recall"] <= 1
        assert 0 <= metrics["roc_auc"] <= 1
        assert len(fpr) == len(tpr)
        assert len(probabilities) == len(X)
