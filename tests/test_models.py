import os
import joblib
import pytest
import numpy as np
import pandas as pd
from pathsnap.data.candidates import CORE_FEATS
from pathsnap.models.evaluate import compute_route_continuity, evaluate_predictions
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
DATA_ROOT = os.environ.get("PATHSNAP_DATA_ROOT", os.path.join(ROOT, "data"))


ART_MODELS = os.path.join(ROOT, "artifacts", "models")

def test_models_exist_and_predict():
    expected_models = [
        "logistic_regression.joblib",
        "gaussian_nb.joblib",
        "decision_tree.joblib",
        "random_forest.joblib",
        "hist_gradient_boosting.joblib"
    ]
    # Synthetic candidate features
    dummy_X = pd.DataFrame({
        "dist": [5.2, 12.0, 45.0],
        "hd": [0.1, 0.5, 1.2],
        "alen": [120.0, 80.0, 200.0],
        "speed": [12.5, 8.0, 0.0],
        "dt": [1.0, 1.0, 2.0]
    })

    for m_file in expected_models:
        path = os.path.join(ART_MODELS, m_file)
        assert os.path.isfile(path), f"Model file missing: {path}"
        model = joblib.load(path)
        probs = model.predict_proba(dummy_X[CORE_FEATS])
        assert probs.shape == (3, 2), f"Expected shape (3, 2), got {probs.shape}"
        assert np.all((probs >= 0.0) & (probs <= 1.0)), "Probabilities must be in [0, 1]"

def test_route_continuity_metric():
    # Disconnected arcs
    arcs = np.array([
        [0, 1], # arc 0: 0 -> 1
        [1, 2], # arc 1: 1 -> 2 (connects from 0)
        [2, 3], # arc 2: 2 -> 3 (connects from 1)
        [9, 10] # arc 3: disconnected
    ])
    # Continuous path: 0 -> 0 -> 1 -> 2
    path1 = np.array([0, 0, 1, 2])
    assert compute_route_continuity(path1, arcs) == 1.0

    # Disconnected jump: 0 -> 3
    path2 = np.array([0, 3])
    assert compute_route_continuity(path2, arcs) == 0.0

def test_evaluation_metrics():
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])
    metrics = evaluate_predictions(y_true, y_pred)
    assert metrics["top1_accuracy"] == 0.75
    assert "macro_f1" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
