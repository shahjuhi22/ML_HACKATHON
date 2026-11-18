"""Classifier + HMM/Viterbi evaluation on v2 features.

Decoder parameters are tuned on the 80 training trips using OUT-OF-FOLD classifier
probabilities (so the classifier never saw the trips it is scoring). The chosen
setting is then applied once to the 10 student-test trips. Evaluator trips are never read.

Usage: python scripts/run_hmm_v2.py <cache_dir> <data_root> <model_name> <out_root>
"""
import os
import sys
import json
import itertools
import time
sys.path.insert(0, os.path.abspath("."))
import numpy as np
import pandas as pd
from pathsnap.data.loader import load_trip
from pathsnap.hmm.decoder import HMMViterbiDecoder


def evaluate(cands, obs_df, probs, data_dir, params, trips=None):
    good = tot = cg = ct = 0
    df = cands.assign(p=probs)
    for tid, c in df.groupby("trip", sort=False):
        if trips is not None and tid not in trips:
            continue
        _, _, arcs, _ = load_trip(data_dir, tid)
        o = obs_df[obs_df["trip"] == tid]
        t = o["t"].values
        dec = HMMViterbiDecoder(**params)
        pred = dec.decode(c[["obs", "arc", "dist"]], arcs, t, len(o), classifier_probs=c["p"].values)
        lab = o["label"].values
        L = lab >= 0
        good += int((pred[L] == lab[L]).sum())
        tot += int(L.sum())
        a, b = pred[:-1], pred[1:]
        ok = (a >= 0) & (b >= 0) & (np.diff(t) <= params.get("gap_threshold_s", 30.0))
        conn = (a == b) | (arcs[np.clip(a, 0, None), 1] == arcs[np.clip(b, 0, None), 0])
        cg += int((ok & conn).sum())
        ct += int(ok.sum())
    return good / max(tot, 1), cg / max(ct, 1)


def main():
    cache, data_root, model, out = sys.argv[1:5]
    tr = pd.read_pickle(os.path.join(cache, "train_cands_v2.pkl"))
    tro = pd.read_pickle(os.path.join(cache, "train_obs_v2.pkl"))
    te = pd.read_pickle(os.path.join(cache, "student_test_cands_v2.pkl"))
    teo = pd.read_pickle(os.path.join(cache, "student_test_obs_v2.pkl"))
    teo_proxy = teo.copy()
    if os.environ.get("PATHSNAP_LABELS", "proxy") == "dp":
        tro["label"] = np.load(os.path.join(cache, "train_label_dp.npy"))
        teo["label"] = np.load(os.path.join(cache, "student_test_label_dp.npy"))
    oof = np.load(os.path.join(cache, f"oof_{model}.npy"))
    tp = np.load(os.path.join(cache, f"test_{model}.npy"))
    train_dir = os.path.join(data_root, "train")
    test_dir = os.path.join(data_root, "student_test")

    grid = {
        "classifier_weight": [0.0, 0.5, 1.0, 2.0],
        "sigma": [10.0, 20.0, 40.0],
        "p_other": [1e-4, 1e-6],
        "p_uturn": [None, 1e-3],
    }
    fixed = {"p_stay": 0.70, "p_conn": 0.25, "gap_threshold_s": 30.0}
    rows = []
    for cw, sg, po, pu in itertools.product(*grid.values()):
        params = dict(fixed, classifier_weight=cw, sigma=sg, p_other=po, p_uturn=pu)
        t0 = time.time()
        acc, cont = evaluate(tr, tro, oof, train_dir, params)
        rows.append({**params, "train_oof_top1": round(acc, 4), "train_oof_continuity": round(cont, 4),
                     "seconds": round(time.time() - t0, 1)})
        print(rows[-1], flush=True)
    grid_df = pd.DataFrame(rows)
    tab = os.path.join(out, "results", "tables")
    os.makedirs(tab, exist_ok=True)
    grid_df.to_csv(os.path.join(tab, f"hmm_grid_train_v2_{model}.csv"), index=False)

    best = grid_df.sort_values(["train_oof_top1", "train_oof_continuity"], ascending=False).iloc[0]
    bp = {k: float(best[k]) for k in ["classifier_weight", "sigma", "p_other", "p_stay", "p_conn", "gap_threshold_s"]}
    bp["p_uturn"] = None if pd.isna(best["p_uturn"]) else float(best["p_uturn"])
    print("selected on train:", bp, flush=True)

    # Student-test: classifier-only vs pure HMM vs classifier + HMM, same partition and labels
    clf_rows = te.assign(p=tp).sort_values("p", ascending=False).drop_duplicates(["trip", "obs"])
    lab = teo[teo["label"] >= 0]
    m = lab.merge(clf_rows[["trip", "obs", "arc"]], on=["trip", "obs"], how="left")
    clf_acc = float((m["arc"] == m["label"]).mean())
    pure = dict(bp, classifier_weight=0.0)
    pure_acc, pure_cont = evaluate(te, teo, tp, test_dir, pure)
    hyb_acc, hyb_cont = evaluate(te, teo, tp, test_dir, bp)
    hyb_proxy, _ = evaluate(te, teo_proxy, tp, test_dir, bp)
    m2 = teo_proxy[teo_proxy["label"] >= 0].merge(clf_rows[["trip", "obs", "arc"]], on=["trip", "obs"], how="left")
    clf_proxy = float((m2["arc"] == m2["label"]).mean())
    summary = {"model": model, "selected_params": bp,
               "test_classifier_only_top1": round(clf_acc, 4),
               "test_pure_hmm_top1": round(pure_acc, 4), "test_pure_hmm_continuity": round(pure_cont, 4),
               "test_classifier_hmm_top1": round(hyb_acc, 4), "test_classifier_hmm_continuity": round(hyb_cont, 4),
               "labels": os.environ.get("PATHSNAP_LABELS", "proxy"),
               "test_classifier_only_top1_vs_proxy_labels": round(clf_proxy, 4),
               "test_classifier_hmm_top1_vs_proxy_labels": round(hyb_proxy, 4)}
    print(summary, flush=True)
    with open(os.path.join(tab, f"hmm_v2_summary_{model}.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
