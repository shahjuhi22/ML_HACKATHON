import os
import sys
sys.path.insert(0, os.path.abspath("."))
import time
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, log_loss

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    optuna = None

from pathsnap.data.loader import load_trip, list_trips
from pathsnap.data.candidates import generate_candidates, CORE_FEATS
from pathsnap.data.alignment import proxy_reference_labels
from pathsnap.models.evaluate import compute_route_continuity
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
DATA_ROOT = os.environ.get("PATHSNAP_DATA_ROOT", os.path.join(ROOT, "data"))


SEED = 42

def load_data(data_dir: str, trip_ids: List[str]) -> pd.DataFrame:
    cands_list = []
    for tid in trip_ids:
        track, nodes, arcs, route = load_trip(data_dir, tid, with_route=True)
        c = generate_candidates(track, nodes, arcs)
        lab = proxy_reference_labels(track, nodes, arcs, route)
        c["trip"] = tid
        obs_df = pd.DataFrame({"obs": np.arange(len(track)), "label": lab})
        if len(c) > 0:
            c = c.merge(obs_df, on="obs")
            c["y"] = ((c["arc"] == c["label"]) & (c["label"] >= 0)).astype(int)
        else:
            c["y"] = 0
        cands_list.append(c)
    return pd.concat(cands_list, ignore_index=True)

def run_optuna_tuning(n_trials: int = 20):
    if optuna is None:
        print("Optuna is not installed.")
        return

    base_data = os.path.join(DATA_ROOT)
    train_dir = os.path.join(base_data, "train")
    test_dir = os.path.join(base_data, "student_test")

    out_art = os.path.join(ROOT, "artifacts")
    out_res = os.path.join(ROOT, "results", "tables")
    os.makedirs(os.path.join(out_art, "models"), exist_ok=True)
    os.makedirs(os.path.join(out_art, "configs"), exist_ok=True)
    os.makedirs(out_res, exist_ok=True)

    train_trips = list_trips(train_dir)
    test_trips = list_trips(test_dir)

    print("Loading candidate features for tuning...")
    # Load 40 trips for tuning to keep runtime fast and rigorous
    train_df = load_data(train_dir, train_trips[:40])
    test_df = load_data(test_dir, test_trips)

    gkf = GroupKFold(n_splits=3)
    folds = list(gkf.split(train_df, train_df["y"], groups=train_df["trip"].values))

    def evaluate_cv(params: Dict[str, Any]) -> float:
        scores = []
        for tr_idx, val_idx in folds:
            tr = train_df.iloc[tr_idx]
            if len(tr) > 100000:
                tr = tr.sample(n=100000, random_state=SEED)
            val = train_df.iloc[val_idx]
            clf = HistGradientBoostingClassifier(random_state=SEED, **params)
            clf.fit(tr[CORE_FEATS], tr["y"])
            probs = clf.predict_proba(val[CORE_FEATS])[:, 1]
            val_c = val.copy()
            val_c["p"] = probs
            best_idx = val_c.groupby(["trip", "obs"])["p"].idxmax()
            best = val_c.loc[best_idx]
            acc = ((best["arc"] == best["label"]) & (best["label"] >= 0)).sum() / max(1, (best["label"] >= 0).sum())
            scores.append(acc)
        return float(np.mean(scores))

    # Evaluate untuned baseline
    untuned_params = {"max_iter": 200, "learning_rate": 0.05, "max_leaf_nodes": 31}
    untuned_cv = evaluate_cv(untuned_params)
    print(f"Untuned HGB 3-Fold CV Top-1: {untuned_cv*100:.2f}%")

    def objective(trial):
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 63),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 20, 200),
            "l2_regularization": trial.suggest_float("l2_regularization", 0.01, 10.0, log=True),
            "max_iter": trial.suggest_int("max_iter", 100, 300),
        }
        return evaluate_cv(params)

    sampler = optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    print(f"Running Optuna Bayesian Optimization ({n_trials} trials)...")
    study.optimize(objective, n_trials=n_trials)

    print(f"\nBest Optuna Trial #{study.best_trial.number}:")
    print(f"Best CV Top-1: {study.best_value*100:.2f}%")
    print("Best Parameters:", study.best_params)

    # Save trials dataframe
    trials_df = study.trials_dataframe()
    trials_csv = os.path.join(out_res, "optuna_trials.csv")
    trials_df.to_csv(trials_csv, index=False)

    # Fit both untuned and tuned on full train_df and evaluate on student_test
    clf_untuned = HistGradientBoostingClassifier(random_state=SEED, **untuned_params).fit(train_df[CORE_FEATS], train_df["y"])
    t0 = time.time()
    clf_tuned = HistGradientBoostingClassifier(random_state=SEED, **study.best_params).fit(train_df[CORE_FEATS], train_df["y"])
    tuned_fit_time = round(time.time() - t0, 3)

    def eval_test(clf):
        t0 = time.time()
        probs = clf.predict_proba(test_df[CORE_FEATS])[:, 1]
        inf_time = round(time.time() - t0, 3)
        preds = (probs >= 0.5).astype(int)
        macro_f1 = float(f1_score(test_df["y"], preds, average="macro", zero_division=0))
        ll = float(log_loss(test_df["y"], np.clip(probs, 1e-7, 1 - 1e-7)))

        test_c = test_df.copy()
        test_c["p"] = probs
        best_idx = test_c.groupby(["trip", "obs"])["p"].idxmax()
        best = test_c.loc[best_idx]
        top1 = float(((best["arc"] == best["label"]) & (best["label"] >= 0)).sum() / max(1, (best["label"] >= 0).sum()))
        return top1, macro_f1, ll, inf_time

    top1_un, f1_un, ll_un, inf_un = eval_test(clf_untuned)
    top1_tu, f1_tu, ll_tu, inf_tu = eval_test(clf_tuned)

    tuning_comp = pd.DataFrame([
        {
            "variant": "Untuned HGB (Baseline)",
            "cv3_top1": round(untuned_cv, 4),
            "test_top1": round(top1_un, 4),
            "test_macro_f1": round(f1_un, 4),
            "test_log_loss": round(ll_un, 4),
            "inference_seconds": inf_un
        },
        {
            "variant": "Tuned HGB (Optuna TPESampler)",
            "cv3_top1": round(study.best_value, 4),
            "test_top1": round(top1_tu, 4),
            "test_macro_f1": round(f1_tu, 4),
            "test_log_loss": round(ll_tu, 4),
            "inference_seconds": inf_tu
        }
    ])
    comp_csv = os.path.join(out_res, "tuning_comparison.csv")
    tuning_comp.to_csv(comp_csv, index=False)
    print("\n--- Tuning Comparison Table ---")
    print(tuning_comp.to_string(index=False))

    # Serialize best model and config
    joblib.dump(clf_tuned, os.path.join(out_art, "models", "hgb_tuned.joblib"))
    best_config = {
        "best_model": "hgb_tuned",
        "algorithm_family": "Gradient Boosting (HistGradientBoosting)",
        "features": CORE_FEATS,
        "selected_by": "3-fold GroupKFold Top-1 Accuracy with Optuna TPESampler",
        "cv_top1_score": round(study.best_value, 4),
        "test_top1_score": round(top1_tu, 4),
        "best_params": study.best_params,
        "fit_time_seconds": tuned_fit_time
    }
    best_cfg_path = os.path.join(out_art, "configs", "best.json")
    with open(best_cfg_path, "w", encoding="utf-8") as f:
        json.dump(best_config, f, indent=2)
    print(f"Saved best config to {best_cfg_path}")

    # Copy to gagana's results and artifacts
    for dest in [
        os.path.join(ROOT, "results", "tables", "optuna_trials.csv"),
        os.path.join(ROOT, "results", "tables", "tuning_comparison.csv"),
    ]:
        pd.read_csv(trials_csv if "trials" in dest else comp_csv).to_csv(dest, index=False)
    joblib.dump(clf_tuned, os.path.join(ROOT, "artifacts", "models", "hgb_tuned.joblib"))
    with open(os.path.join(ROOT, "artifacts", "configs", "best.json"), "w", encoding="utf-8") as f:
        json.dump(best_config, f, indent=2)

if __name__ == "__main__":
    run_optuna_tuning()
