import os
import sys
sys.path.insert(0, os.path.abspath("."))
import time
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, log_loss

from pathsnap.data.loader import load_trip, list_trips
from pathsnap.data.candidates import generate_candidates, compute_candidate_miss_rate, CORE_FEATS
from pathsnap.data.alignment import proxy_reference_labels
from pathsnap.models.evaluate import compute_route_continuity, evaluate_predictions
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
DATA_ROOT = os.environ.get("PATHSNAP_DATA_ROOT", os.path.join(ROOT, "data"))


SEED = 42

def build_dataset(data_dir: str, trip_ids: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Extract candidate features and observation metadata for all specified trips."""
    all_cands = []
    all_obs = []

    print(f"Extracting features from {len(trip_ids)} trips in {data_dir}...")
    for tid in trip_ids:
        track, nodes, arcs, route = load_trip(data_dir, tid, with_route=True)
        cands = generate_candidates(track, nodes, arcs)
        labels = proxy_reference_labels(track, nodes, arcs, route)

        cands["trip"] = tid
        obs_df = pd.DataFrame({
            "trip": tid,
            "obs": np.arange(len(track)),
            "lon": track[:, 0],
            "lat": track[:, 1],
            "t": track[:, 2],
            "label": labels
        })

        if len(cands) > 0:
            cands = cands.merge(obs_df[["obs", "label"]], on="obs")
            cands["y"] = ((cands["arc"] == cands["label"]) & (cands["label"] >= 0)).astype(int)
        else:
            cands["y"] = 0

        all_cands.append(cands)
        all_obs.append(obs_df)

    cands_df = pd.concat(all_cands, ignore_index=True)
    obs_df = pd.concat(all_obs, ignore_index=True)
    return cands_df, obs_df

def run_experiment():
    base_data = os.path.join(DATA_ROOT)
    train_dir = os.path.join(base_data, "train")
    test_dir = os.path.join(base_data, "student_test")

    out_art = os.path.join(ROOT, "artifacts")
    os.makedirs(os.path.join(out_art, "models"), exist_ok=True)
    os.makedirs(os.path.join(out_art, "configs"), exist_ok=True)

    out_res = os.path.join(ROOT, "results", "tables")
    os.makedirs(out_res, exist_ok=True)

    train_trips = list_trips(train_dir)
    test_trips = list_trips(test_dir)

    train_cands, train_obs = build_dataset(train_dir, train_trips)
    test_cands, test_obs = build_dataset(test_dir, test_trips)

    print(f"Train dataset: {len(train_cands)} candidate rows across {len(train_trips)} trips")
    print(f"Test dataset:  {len(test_cands)} candidate rows across {len(test_trips)} trips")

    # Define 5 model families
    models = {
        "logistic_regression": {
            "family": "Linear / Probabilistic",
            "config": "StandardScaler + LogisticRegression(C=1.0, max_iter=500)",
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(C=1.0, max_iter=500, random_state=SEED))
            ])
        },
        "gaussian_nb": {
            "family": "Probabilistic Generative",
            "config": "StandardScaler + GaussianNB(var_smoothing=1e-9)",
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", GaussianNB())
            ])
        },
        "decision_tree": {
            "family": "Tree-Based Non-Linear",
            "config": "DecisionTreeClassifier(max_depth=12, min_samples_leaf=50)",
            "pipeline": DecisionTreeClassifier(max_depth=12, min_samples_leaf=50, random_state=SEED)
        },
        "random_forest": {
            "family": "Ensemble Bagging",
            "config": "RandomForestClassifier(n_estimators=100, max_depth=14, min_samples_leaf=20)",
            "pipeline": RandomForestClassifier(n_estimators=100, max_depth=14, min_samples_leaf=20, n_jobs=-1, random_state=SEED)
        },
        "hist_gradient_boosting": {
            "family": "Gradient Boosting",
            "config": "HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=31)",
            "pipeline": HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=31, random_state=SEED)
        }
    }

    # GroupKFold (5 folds) on train
    gkf = GroupKFold(n_splits=5)
    groups = train_cands["trip"].values

    results = []

    for name, info in models.items():
        print(f"\n--- Training {name} ({info['family']}) ---")
        pipe = info["pipeline"]

        # Cross-validation
        cv_top1s = []
        cv_f1s = []

        # Sample for quick and stable CV
        for fold, (tr_idx, val_idx) in enumerate(gkf.split(train_cands, train_cands["y"], groups=groups)):
            tr_df = train_cands.iloc[tr_idx]
            val_df = train_cands.iloc[val_idx]

            # Fit on training fold
            pipe.fit(tr_df[CORE_FEATS], tr_df["y"])

            # Evaluate top-1 on validation fold
            val_preds_prob = pipe.predict_proba(val_df[CORE_FEATS])[:, 1]
            val_df_c = val_df.copy()
            val_df_c["p"] = val_preds_prob

            # Best candidate per observation
            best_idx = val_df_c.groupby(["trip", "obs"])["p"].idxmax()
            best_cands = val_df_c.loc[best_idx]
            correct = (best_cands["arc"] == best_cands["label"]) & (best_cands["label"] >= 0)
            valid_total = (best_cands["label"] >= 0).sum()
            fold_acc = correct.sum() / max(1, valid_total)
            cv_top1s.append(fold_acc)

            fold_f1 = f1_score(val_df["y"], (val_preds_prob >= 0.5).astype(int), average="macro", zero_division=0)
            cv_f1s.append(fold_f1)

        mean_cv_top1 = float(np.mean(cv_top1s))
        mean_cv_f1 = float(np.mean(cv_f1s))
        print(f"5-Fold CV: Top-1={mean_cv_top1*100:.2f}%, Macro F1={mean_cv_f1:.4f}")

        # Train on full training set
        t0 = time.time()
        pipe.fit(train_cands[CORE_FEATS], train_cands["y"])
        train_time = round(time.time() - t0, 3)

        # Inference on student_test set
        t0 = time.time()
        test_probs = pipe.predict_proba(test_cands[CORE_FEATS])[:, 1]
        inf_time = round(time.time() - t0, 3)

        # Candidate-level classification metrics
        test_preds = (test_probs >= 0.5).astype(int)
        macro_f1 = float(f1_score(test_cands["y"], test_preds, average="macro", zero_division=0))
        prec = float(precision_score(test_cands["y"], test_preds, average="macro", zero_division=0))
        rec = float(recall_score(test_cands["y"], test_preds, average="macro", zero_division=0))
        logloss = float(log_loss(test_cands["y"], np.clip(test_probs, 1e-7, 1 - 1e-7)))

        # Observation-level Top-1 Accuracy and Route Continuity on test trips
        test_cands_copy = test_cands.copy()
        test_cands_copy["p"] = test_probs
        best_test_idx = test_cands_copy.groupby(["trip", "obs"])["p"].idxmax()
        best_test = test_cands_copy.loc[best_test_idx]

        test_correct = (best_test["arc"] == best_test["label"]) & (best_test["label"] >= 0)
        test_valid = (best_test["label"] >= 0).sum()
        test_top1 = float(test_correct.sum() / max(1, test_valid))

        # Continuity per test trip
        trip_conts = []
        trip_misses = []
        for tid in test_trips:
            track, nodes, arcs, route = load_trip(test_dir, tid, with_route=True)
            labels = proxy_reference_labels(track, nodes, arcs, route)
            cands_t = test_cands_copy[test_cands_copy["trip"] == tid]
            mr, _, _ = compute_candidate_miss_rate(cands_t, labels)
            trip_misses.append(mr)

            pred_dict = dict(zip(cands_t[cands_t.index.isin(best_test_idx)]["obs"], cands_t[cands_t.index.isin(best_test_idx)]["arc"]))
            pred_arr = np.array([pred_dict.get(i, -1) for i in range(len(track))])
            cont = compute_route_continuity(pred_arr, arcs)
            trip_conts.append(cont)

        mean_continuity = float(np.mean(trip_conts))
        mean_miss_rate = float(np.mean(trip_misses))

        print(f"Test Evaluation: Top-1={test_top1*100:.2f}%, F1={macro_f1:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, Cont={mean_continuity*100:.2f}%, Train={train_time}s, Inf={inf_time}s")

        # Save model
        joblib.dump(pipe, os.path.join(out_art, "models", f"{name}.joblib"))

        results.append({
            "model": name,
            "algorithm_family": info["family"],
            "key_configuration": info["config"],
            "cv_top1_accuracy": round(mean_cv_top1, 4),
            "cv_macro_f1": round(mean_cv_f1, 4),
            "test_top1_accuracy": round(test_top1, 4),
            "test_macro_f1": round(macro_f1, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "log_loss": round(logloss, 4),
            "route_continuity": round(mean_continuity, 4),
            "candidate_miss_rate": round(mean_miss_rate, 4),
            "train_seconds": train_time,
            "inference_seconds": inf_time
        })

    res_df = pd.DataFrame(results)
    csv_path = os.path.join(out_res, "model_comparison.csv")
    res_df.to_csv(csv_path, index=False)
    print(f"\nSaved comprehensive model comparison to {csv_path}:")
    print(res_df.to_string(index=False))

    # Also save to gagana/results/tables/model_comparison.csv
    gagana_csv = os.path.join(ROOT, "results", "tables", "model_comparison.csv")
    res_df.to_csv(gagana_csv, index=False)

if __name__ == "__main__":
    from typing import Tuple
    run_experiment()
