import os
import sys
sys.path.insert(0, os.path.abspath("."))
import time
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Any

from pathsnap.data.loader import load_trip, list_trips
from pathsnap.data.candidates import generate_candidates, compute_candidate_miss_rate, CORE_FEATS
from pathsnap.data.alignment import proxy_reference_labels, sequence_aligned_reference_labels
from pathsnap.models.evaluate import compute_route_continuity
from pathsnap.hmm.decoder import HMMViterbiDecoder
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
DATA_ROOT = os.environ.get("PATHSNAP_DATA_ROOT", os.path.join(ROOT, "data"))


def run_benchmark():
    base_data = os.path.join(DATA_ROOT)
    art_dir = os.path.join(ROOT, "artifacts")
    res_dir = os.path.join(ROOT, "results", "tables")
    os.makedirs(res_dir, exist_ok=True)

    # Load best model
    best_model_path = os.path.join(art_dir, "models", "hist_gradient_boosting.joblib")
    if not os.path.isfile(best_model_path):
        best_model_path = os.path.join(ROOT, "artifacts", "models", "hist_gradient_boosting.joblib")
    
    print(f"Loading classifier from {best_model_path}...")
    model = joblib.load(best_model_path)

    pure_hmm = HMMViterbiDecoder(sigma=10.0, classifier_weight=0.0)
    hybrid_hmm = HMMViterbiDecoder(sigma=10.0, classifier_weight=1.0)

    partitions = [
        ("train", os.path.join(base_data, "train"), list_trips(os.path.join(base_data, "train"))[:20]), # sample 20 for speed
        ("student_test", os.path.join(base_data, "student_test"), list_trips(os.path.join(base_data, "student_test"))),
    ]

    summary_rows = []

    for part_name, folder, tids in partitions:
        print(f"\nEvaluating partition: {part_name} ({len(tids)} trips)...")
        
        counts = {
            "obs_total": 0,
            "baseline_corr": 0,
            "clf_corr": 0,
            "pure_hmm_corr": 0,
            "hybrid_hmm_corr": 0,
        }
        conts = {
            "baseline": [],
            "clf": [],
            "pure_hmm": [],
            "hybrid_hmm": [],
        }
        miss_rates = []

        for tid in tids:
            track, nodes, arcs, route = load_trip(folder, tid, with_route=True)
            N = len(track)
            cands = generate_candidates(track, nodes, arcs)
            labels = proxy_reference_labels(track, nodes, arcs, route)

            mr, _, _ = compute_candidate_miss_rate(cands, labels)
            miss_rates.append(mr)

            if len(cands) == 0:
                pred_base = np.full(N, -1)
                pred_clf = np.full(N, -1)
                pred_pure = np.full(N, -1)
                pred_hybrid = np.full(N, -1)
            else:
                # 1. Baseline: nearest candidate by distance
                best_base_idx = cands.groupby("obs")["dist"].idxmin()
                best_base = cands.loc[best_base_idx]
                d_base = dict(zip(best_base["obs"].astype(int), best_base["arc"].astype(int)))
                pred_base = np.array([d_base.get(i, -1) for i in range(N)])

                # 2. Classifier-only Top-1
                probs = model.predict_proba(cands[CORE_FEATS].values)[:, 1]
                cands["p"] = probs
                best_clf_idx = cands.groupby("obs")["p"].idxmax()
                best_clf = cands.loc[best_clf_idx]
                d_clf = dict(zip(best_clf["obs"].astype(int), best_clf["arc"].astype(int)))
                pred_clf = np.array([d_clf.get(i, -1) for i in range(N)])

                # 3. Pure Distance HMM
                pred_pure = pure_hmm.decode(cands, arcs, track[:, 2], N)

                # 4. Hybrid HMM (Classifier Posterior + HMM)
                pred_hybrid = hybrid_hmm.decode(cands, arcs, track[:, 2], N, classifier_probs=probs)

            # Record accuracy
            valid = labels >= 0
            n_valid = int(valid.sum())
            if n_valid > 0:
                counts["obs_total"] += n_valid
                counts["baseline_corr"] += int((pred_base[valid] == labels[valid]).sum())
                counts["clf_corr"] += int((pred_clf[valid] == labels[valid]).sum())
                counts["pure_hmm_corr"] += int((pred_pure[valid] == labels[valid]).sum())
                counts["hybrid_hmm_corr"] += int((pred_hybrid[valid] == labels[valid]).sum())

            # Record continuity
            conts["baseline"].append(compute_route_continuity(pred_base, arcs))
            conts["clf"].append(compute_route_continuity(pred_clf, arcs))
            conts["pure_hmm"].append(compute_route_continuity(pred_pure, arcs))
            conts["hybrid_hmm"].append(compute_route_continuity(pred_hybrid, arcs))

        obs_tot = max(1, counts["obs_total"])
        summary_rows.append({
            "partition": part_name,
            "labelled_observations": obs_tot,
            "candidate_miss_rate": round(float(np.mean(miss_rates)), 4),
            "baseline_accuracy": round(counts["baseline_corr"] / obs_tot, 4),
            "classifier_only_accuracy": round(counts["clf_corr"] / obs_tot, 4),
            "pure_hmm_accuracy": round(counts["pure_hmm_corr"] / obs_tot, 4),
            "hybrid_hmm_accuracy": round(counts["hybrid_hmm_corr"] / obs_tot, 4),
            "baseline_continuity": round(float(np.mean(conts["baseline"])), 4),
            "classifier_only_continuity": round(float(np.mean(conts["clf"])), 4),
            "pure_hmm_continuity": round(float(np.mean(conts["pure_hmm"])), 4),
            "hybrid_hmm_continuity": round(float(np.mean(conts["hybrid_hmm"])), 4),
        })

    summary_df = pd.DataFrame(summary_rows)
    out_csv = os.path.join(res_dir, "hmm_vs_baseline.csv")
    summary_df.to_csv(out_csv, index=False)
    print(f"\n--- HMM vs Baseline Comparison Table ({out_csv}) ---")
    print(summary_df.to_string(index=False))

    # Also update himesh/results/tables/hmm_vs_baseline.csv
    himesh_csv = os.path.join(ROOT, "results", "tables", "hmm_vs_baseline.csv")
    summary_df.to_csv(himesh_csv, index=False)

if __name__ == "__main__":
    run_benchmark()
