"""Build and cache v2 candidate feature tables for train and student_test.

Usage: python scripts/build_features_v2.py <data_root containing train/ and student_test/> <out_dir>
Labels (proxy: nearest reference-route arc within 50 m) are attached for training
and scoring only; the features themselves never use the route.
"""
import os
import sys
import time
sys.path.insert(0, os.path.abspath("."))
import numpy as np
import pandas as pd
from pathsnap.data.loader import load_trip, list_trips
from pathsnap.data.alignment import proxy_reference_labels
from pathsnap.data.features_v2 import generate_candidates_v2


def build(part_dir, tag):
    rows, obs_rows = [], []
    for tid in list_trips(part_dir):
        track, nodes, arcs, route = load_trip(part_dir, tid, with_route=True)
        c = generate_candidates_v2(track, nodes, arcs)
        lab = proxy_reference_labels(track, nodes, arcs, route)
        c["trip"] = tid
        c["y"] = (c["arc"].values == lab[c["obs"].values]).astype(np.int8)
        rows.append(c)
        obs_rows.append(pd.DataFrame({"trip": tid, "obs": np.arange(len(track)),
                                      "t": track[:, 2], "label": lab}))
        print(f"{tag} {tid}: obs={len(track)} cand_rows={len(c)}", flush=True)
    return pd.concat(rows, ignore_index=True), pd.concat(obs_rows, ignore_index=True)


if __name__ == "__main__":
    root, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    for part in ["train", "student_test"]:
        t0 = time.time()
        cands, obs = build(os.path.join(root, part), part)
        cands.to_pickle(os.path.join(out, f"{part}_cands_v2.pkl"))
        obs.to_pickle(os.path.join(out, f"{part}_obs_v2.pkl"))
        print(f"{part}: {len(cands)} rows, {len(obs)} obs, {time.time()-t0:.1f}s", flush=True)
