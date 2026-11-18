"""Attach chronological (DP-aligned) labels next to the proxy labels in the v2 cache.
Row order is identical to the cached candidate tables. Used for training and scoring only."""
import os, sys, time
sys.path.insert(0, os.path.abspath("."))
import numpy as np
import pandas as pd
from pathsnap.data.loader import load_trip
from pathsnap.data.alignment import monotonic_route_labels

cache, root = sys.argv[1], sys.argv[2]
for part in ["train", "student_test"]:
    t0 = time.time()
    c = pd.read_pickle(os.path.join(cache, f"{part}_cands_v2.pkl"))[["trip", "obs", "arc"]]
    o = pd.read_pickle(os.path.join(cache, f"{part}_obs_v2.pkl"))[["trip", "obs"]]
    lab_obs = np.full(len(o), -1)
    y = np.zeros(len(c), dtype=np.int8)
    for tid in o["trip"].unique():
        track, nodes, arcs, route = load_trip(os.path.join(root, part), tid, with_route=True)
        L = monotonic_route_labels(track, nodes, arcs, route)
        oi = np.where(o["trip"].values == tid)[0]
        lab_obs[oi] = L
        ci = np.where(c["trip"].values == tid)[0]
        y[ci] = (c["arc"].values[ci] == L[c["obs"].values[ci]]).astype(np.int8)
    np.save(os.path.join(cache, f"{part}_y_dp.npy"), y)
    np.save(os.path.join(cache, f"{part}_label_dp.npy"), lab_obs)
    print(part, "labelled obs", int((lab_obs >= 0).sum()), "of", len(o), f"{time.time()-t0:.1f}s", flush=True)
