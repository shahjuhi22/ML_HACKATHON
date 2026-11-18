import os
import time
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

from pathsnap.data.candidates import CORE_FEATS
from pathsnap.models.evaluate import evaluate_predictions, compute_route_continuity
from pathsnap.data.loader import load_trip
from pathsnap.data.candidates import generate_candidates, compute_candidate_miss_rate
from pathsnap.data.alignment import proxy_reference_labels

SEED = 42

def get_model_families() -> Dict[str, Tuple[str, Any]]:
    """
    Returns 5 distinct algorithm families with appropriate scalers/configurations:
    1. Linear/probabilistic: Logistic Regression
    2. Generative/probabilistic: Gaussian Naive Bayes
    3. Tree-based: Decision Tree
    4. Ensemble bagging: Random Forest
    5. Gradient Boosting: HistGradientBoosting
    """
    return {
        "logistic_regression": (
            "Linear / Probabilistic",
            Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=500, random_state=SEED))
            ])
        ),
        "gaussian_nb": (
            "Probabilistic Generative",
            Pipeline([
                ("scaler", StandardScaler()),
                ("clf", GaussianNB())
            ])
        ),
        "decision_tree": (
            "Tree-Based Non-Linear",
            DecisionTreeClassifier(max_depth=12, min_samples_leaf=50, random_state=SEED)
        ),
        "random_forest": (
            "Ensemble Bagging",
            RandomForestClassifier(n_estimators=100, max_depth=14, min_samples_leaf=20, n_jobs=-1, random_state=SEED)
        ),
        "hist_gradient_boosting": (
            "Gradient Boosting",
            HistGradientBoostingClassifier(max_iter=200, random_state=SEED)
        ),
    }

def evaluate_on_partition(
    model: Any, 
    data_folder: str, 
    trip_ids: List[str]
) -> Dict[str, Any]:
    """
    Evaluate trained candidate classifier across a full set of trips at observation level.
    For each observation, selects the candidate arc with highest predicted probability P(y=1).
    Computes Top-1 accuracy, Precision, Recall, Macro F1, Log Loss, and Route Continuity.
    """
    all_obs_true = []
    all_obs_pred = []
    all_probs = []
    all_cand_true = []
    all_cand_prob = []
    continuities = []
    miss_rates = []

    t0 = time.time()
    for tid in trip_ids:
        track, nodes, arcs, route = load_trip(data_folder, tid, with_route=True)
        cands = generate_candidates(track, nodes, arcs)
        labels = proxy_reference_labels(track, nodes, arcs, route)

        mr, _, _ = compute_candidate_miss_rate(cands, labels)
        miss_rates.append(mr)

        if len(cands) == 0:
            pred_arcs = np.full(len(track), -1)
        else:
            X = cands[CORE_FEATS].values
            probs = model.predict_proba(X)[:, 1]
            cands = cands.copy()
            cands["p"] = probs

            # Pick candidate with max probability per observation
            best_idx = cands.groupby("obs")["p"].idxmax()
            best_cands = cands.loc[best_idx]
            pred_dict = dict(zip(best_cands["obs"], best_cands["arc"]))
            pred_prob_dict = dict(zip(best_cands["obs"], best_cands["p"]))

            pred_arcs = np.array([pred_dict.get(i, -1) for i in range(len(track))])
            pred_probs = np.array([pred_prob_dict.get(i, 0.0) for i in range(len(track))])

            # For candidate-level metrics
            cands_with_labels = cands.merge(
                pd.DataFrame({"obs": np.arange(len(track)), "label": labels}),
                on="obs"
            )
            y_cand = ((cands_with_labels["arc"] == cands_with_labels["label"]) & (cands_with_labels["label"] >= 0)).astype(int)
            all_cand_true.extend(y_cand.tolist())
            all_cand_prob.extend(probs.tolist())

        # Observation-level evaluation against valid ground truth labels
        valid = labels >= 0
        if valid.any():
            all_obs_true.extend(labels[valid].tolist())
            all_obs_pred.extend(pred_arcs[valid].tolist())
            if len(cands) > 0:
                all_probs.extend(pred_probs[valid].tolist())

        # Route continuity along the predicted sequence
        cont = compute_route_continuity(pred_arcs, arcs)
        continuities.append(cont)

    inf_time = time.time() - t0

    # Observation-level top-1 accuracy
    all_obs_true = np.array(all_obs_true)
    all_obs_pred = np.array(all_obs_pred)
    top1_acc = float((all_obs_true == all_obs_pred).mean()) if len(all_obs_true) > 0 else 0.0

    # Candidate-level classification metrics
    all_cand_true = np.array(all_cand_true)
    all_cand_prob = np.array(all_cand_prob)
    all_cand_pred = (all_cand_prob >= 0.5).astype(int)
    clf_metrics = evaluate_predictions(all_cand_true, all_cand_pred, all_cand_prob)

    return {
        "top1_accuracy": top1_acc,
        "macro_f1": clf_metrics["macro_f1"],
        "precision": clf_metrics["precision"],
        "recall": clf_metrics["recall"],
        "log_loss": clf_metrics.get("log_loss", None),
        "route_continuity": float(np.mean(continuities)),
        "candidate_miss_rate": float(np.mean(miss_rates)),
        "inference_seconds": round(inf_time, 3),
    }
