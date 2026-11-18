"""Extended model comparison on v2 features.

- 5-fold GroupKFold by trip on the 80 training trips (selection uses CV only).
- Student-test (10 trips) used once per model for development-stage reporting.
- Evaluator trips are never read here.

Usage: python scripts/train_extended_models.py <cache_dir> <data_root> <out_root>
"""
import os
import sys
import json
import time
sys.path.insert(0, os.path.abspath("."))
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, log_loss
import lightgbm as lgb
import xgboost as xgb

from pathsnap.data.features_v2 import V2_FEATS, CORE_FEATS
from pathsnap.data.loader import load_trip

SEED = 42
SUBSAMPLE = 400_000  # row cap per fit for the slow sklearn models (stated in the report)


class RankerWrapper:
    """LightGBM LambdaRank model that exposes per-observation softmax probabilities."""

    def __init__(self, **params):
        self.params = params
        self.model = lgb.LGBMRanker(**params)

    def fit(self, X, y, group):
        self.model.fit(X, y, group=group)
        return self

    def scores(self, X):
        return self.model.predict(X)


def groups_of(df):
    return df.groupby(["trip", "obs"], sort=False).size().values


def softmax_by_obs(df, score):
    s = pd.Series(score, index=df.index)
    key = [df["trip"].values, df["obs"].values]
    mx = s.groupby(key).transform("max")
    ex = np.exp(s - mx)
    return (ex / ex.groupby(key).transform("sum")).values


def top1(df, score, obs_df):
    """Observation-level top-1 accuracy over labelled observations.
    Labelled observations with no candidates count as wrong."""
    tmp = pd.DataFrame({"trip": df["trip"].values, "obs": df["obs"].values,
                        "arc": df["arc"].values, "s": score})
    best = tmp.sort_values("s", ascending=False).drop_duplicates(["trip", "obs"])
    lab = obs_df[obs_df["label"] >= 0][["trip", "obs", "label"]]
    m = lab.merge(best[["trip", "obs", "arc"]], on=["trip", "obs"], how="left")
    return float((m["arc"] == m["label"]).mean()), best


def continuity(best, data_dir, obs_df, gap_s=30.0):
    """Fraction of consecutive predicted pairs (no gap > gap_s) that are equal or connected."""
    good = tot = 0
    for tid, g in best.groupby("trip"):
        _, _, arcs, _ = load_trip(data_dir, tid)
        o = obs_df[obs_df["trip"] == tid]
        pred = np.full(len(o), -1)
        pred[g["obs"].values] = g["arc"].values
        t = o["t"].values
        a, b = pred[:-1], pred[1:]
        ok = (a >= 0) & (b >= 0) & (np.diff(t) <= gap_s)
        same = a == b
        conn = arcs[np.clip(a, 0, None), 1] == arcs[np.clip(b, 0, None), 0]
        good += int((ok & (same | conn)).sum())
        tot += int(ok.sum())
    return good / max(tot, 1)


def make_models():
    return {
        "logistic_regression": ("Linear", lambda: Pipeline([("sc", StandardScaler()), ("m", LogisticRegression(C=1.0, max_iter=1000))]), True),
        "gaussian_nb": ("Probabilistic generative", lambda: Pipeline([("sc", StandardScaler()), ("m", GaussianNB())]), False),
        "decision_tree": ("Tree", lambda: DecisionTreeClassifier(max_depth=14, min_samples_leaf=50, random_state=SEED), False),
        "random_forest": ("Bagging ensemble", lambda: RandomForestClassifier(n_estimators=150, max_depth=18, min_samples_leaf=10, n_jobs=-1, random_state=SEED), True),
        "extra_trees": ("Randomised trees ensemble", lambda: ExtraTreesClassifier(n_estimators=200, max_depth=20, min_samples_leaf=10, n_jobs=-1, random_state=SEED), True),
        "hist_gradient_boosting": ("Boosting", lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_leaf_nodes=63, random_state=SEED), False),
        "lightgbm": ("Boosting (LightGBM)", lambda: lgb.LGBMClassifier(n_estimators=600, learning_rate=0.05, num_leaves=63, min_child_samples=50, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=SEED, n_jobs=-1, verbose=-1), False),
        "xgboost": ("Boosting (XGBoost)", lambda: xgb.XGBClassifier(n_estimators=600, learning_rate=0.05, max_depth=8, subsample=0.8, colsample_bytree=0.8, tree_method="hist", n_jobs=-1, random_state=SEED, eval_metric="logloss"), False),
        "mlp": ("Neural network", lambda: Pipeline([("sc", StandardScaler()), ("m", MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-4, batch_size=1024, max_iter=40, early_stopping=True, random_state=SEED))]), True),
        "lightgbm_ranker": ("Learning-to-rank (LambdaRank)", lambda: RankerWrapper(n_estimators=600, learning_rate=0.05, num_leaves=63, min_child_samples=50, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=SEED, n_jobs=-1, verbose=-1), False),
    }


def fit_predict(name, factory, subsample, Xtr, ytr, dtr, Xte, dte, feats):
    m = factory()
    if subsample and len(Xtr) > SUBSAMPLE:
        rs = np.random.RandomState(SEED)
        idx = np.sort(rs.choice(len(Xtr), SUBSAMPLE, replace=False))
        Xtr, ytr = Xtr[idx], ytr[idx]
        dtr = dtr.iloc[idx]
    t0 = time.time()
    if name == "lightgbm_ranker":
        m.fit(Xtr, ytr, group=groups_of(dtr))
    else:
        m.fit(Xtr, ytr)
    t_fit = time.time() - t0
    t0 = time.time()
    if name == "lightgbm_ranker":
        sc = m.scores(Xte)
        p = softmax_by_obs(dte, sc)
    else:
        p = m.predict_proba(Xte)[:, 1]
        sc = p
    t_inf = time.time() - t0
    return m, sc, p, t_fit, t_inf


def main():
    cache, data_root, out = sys.argv[1], sys.argv[2], sys.argv[3]
    only = sys.argv[4].split(",") if len(sys.argv) > 4 else None
    tab_dir = os.path.join(out, "results", "tables")
    mod_dir = os.path.join(out, "artifacts", "models")
    os.makedirs(tab_dir, exist_ok=True)
    os.makedirs(mod_dir, exist_ok=True)
    tr = pd.read_pickle(os.path.join(cache, "train_cands_v2.pkl"))
    tro = pd.read_pickle(os.path.join(cache, "train_obs_v2.pkl"))
    te = pd.read_pickle(os.path.join(cache, "student_test_cands_v2.pkl"))
    teo = pd.read_pickle(os.path.join(cache, "student_test_obs_v2.pkl"))
    label_mode = os.environ.get("PATHSNAP_LABELS", "proxy")
    sfx = "" if label_mode == "proxy" else f"_{label_mode}"
    teo_proxy = teo.copy()
    if label_mode == "dp":
        tr["y"] = np.load(os.path.join(cache, "train_y_dp.npy"))
        tro["label"] = np.load(os.path.join(cache, "train_label_dp.npy"))
        te["y"] = np.load(os.path.join(cache, "student_test_y_dp.npy"))
        teo["label"] = np.load(os.path.join(cache, "student_test_label_dp.npy"))
    feats = V2_FEATS
    Xtr, ytr = tr[feats].values.astype(np.float32), tr["y"].values
    Xte, yte = te[feats].values.astype(np.float32), te["y"].values
    gkf = GroupKFold(n_splits=5)
    folds = list(gkf.split(Xtr, ytr, groups=tr["trip"].values))
    test_dir = os.path.join(data_root, "student_test")

    rows = []
    res_path = os.path.join(tab_dir, f"model_comparison_v2{sfx}.csv")
    if os.path.exists(res_path) and only:
        rows = [r for r in pd.read_csv(res_path).to_dict("records") if r["model"] not in only]
    for name, (family, factory, sub) in make_models().items():
        if only and name not in only:
            continue
        print(f"=== {name}", flush=True)
        oof_s = np.zeros(len(tr))
        oof_p = np.zeros(len(tr))
        fold_top1, fold_f1 = [], []
        for k, (a, b) in enumerate(folds):
            _, sc, p, _, _ = fit_predict(name, factory, sub, Xtr[a], ytr[a], tr.iloc[a], Xtr[b], tr.iloc[b], feats)
            oof_s[b], oof_p[b] = sc, p
            acc, _ = top1(tr.iloc[b], sc, tro[tro["trip"].isin(tr["trip"].iloc[b].unique())])
            fold_top1.append(acc)
            fold_f1.append(f1_score(ytr[b], p >= 0.5, average="macro"))
            print(f"  fold {k}: top1={acc:.4f}", flush=True)
        cv_top1, _ = top1(tr, oof_s, tro)
        m, sc, p, t_fit, t_inf = fit_predict(name, factory, sub, Xtr, ytr, tr, Xte, te, feats)
        te_top1, best = top1(te, sc, teo)
        te_top1_proxy, _ = top1(te, sc, teo_proxy)
        row = {
            "model": name, "algorithm_family": family, "labels": label_mode,
            "test_top1_vs_proxy_labels": round(te_top1_proxy, 4),
            "train_rows": min(len(Xtr), SUBSAMPLE) if sub else len(Xtr),
            "cv_top1_accuracy": round(cv_top1, 4),
            "cv_top1_fold_std": round(float(np.std(fold_top1)), 4),
            "cv_macro_f1": round(float(np.mean(fold_f1)), 4),
            "cv_log_loss": round(log_loss(ytr, np.clip(oof_p, 1e-7, 1 - 1e-7)), 4),
            "test_top1_accuracy": round(te_top1, 4),
            "test_macro_f1": round(f1_score(yte, p >= 0.5, average="macro"), 4),
            "test_precision": round(precision_score(yte, p >= 0.5, zero_division=0), 4),
            "test_recall": round(recall_score(yte, p >= 0.5, zero_division=0), 4),
            "test_log_loss": round(log_loss(yte, np.clip(p, 1e-7, 1 - 1e-7)), 4),
            "test_route_continuity": round(continuity(best, test_dir, teo), 4),
            "train_seconds": round(t_fit, 2), "inference_seconds": round(t_inf, 3),
        }
        print(row, flush=True)
        rows.append(row)
        joblib.dump(m, os.path.join(mod_dir, f"{name}_v2{sfx}.joblib"))
        np.save(os.path.join(cache, f"oof_{name}{sfx}.npy"), oof_p)
        np.save(os.path.join(cache, f"test_{name}{sfx}.npy"), p)
        pd.DataFrame(rows).to_csv(res_path, index=False)
    print(pd.DataFrame(rows).sort_values("cv_top1_accuracy", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
