import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from sklearn.metrics import f1_score, precision_score, recall_score, log_loss

def compute_route_continuity(predicted_arcs: np.ndarray, arcs: np.ndarray) -> float:
    """
    Calculate Route Continuity:
    Fraction of consecutive non-missing prediction steps where the transition
    is geometrically/topologically valid (either the same arc or connected: arcs[t-1, 1] == arcs[t, 0]).
    """
    valid_mask = (predicted_arcs >= 0) & (predicted_arcs < len(arcs))
    if valid_mask.sum() < 2:
        return 0.0

    valid_idx = np.where(valid_mask)[0]
    # consecutive valid transitions
    consecutive = np.where(np.diff(valid_idx) == 1)[0]
    if len(consecutive) == 0:
        return 0.0

    valid_transitions = 0
    total_transitions = len(consecutive)

    for i in consecutive:
        a_prev = predicted_arcs[valid_idx[i]]
        a_curr = predicted_arcs[valid_idx[i + 1]]
        if a_prev == a_curr:
            valid_transitions += 1
        elif arcs[a_prev, 1] == arcs[a_curr, 0]:
            valid_transitions += 1

    return float(valid_transitions) / float(total_transitions)

def evaluate_predictions(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    y_prob: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Calculate classification metrics: Top-1 Accuracy, Macro F1, Precision, Recall, and Log Loss.
    """
    acc = float((y_true == y_pred).mean())
    f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))

    loss_val = None
    if y_prob is not None:
        try:
            # Handle binary probability format
            if y_prob.ndim == 1 or (y_prob.ndim == 2 and y_prob.shape[1] == 1):
                p = np.clip(y_prob.ravel(), 1e-7, 1 - 1e-7)
                loss_val = float(log_loss(y_true, p))
            elif y_prob.ndim == 2 and y_prob.shape[1] == 2:
                loss_val = float(log_loss(y_true, y_prob))
        except Exception:
            loss_val = None

    metrics = {
        "top1_accuracy": acc,
        "macro_f1": f1,
        "precision": prec,
        "recall": rec,
    }
    if loss_val is not None:
        metrics["log_loss"] = loss_val
    return metrics
