import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from typing import Tuple, List, Optional
from pathsnap.data.projection import project, seg_dist

R_CAND = 50.0  # Metres: default candidate radius
K_CAND = 40    # Number of nearest arc midpoints queried per GPS observation

# Standard core features used across all models
CORE_FEATS = ["dist", "hd", "alen", "speed", "dt"]

def generate_candidates(
    track: np.ndarray, 
    nodes: np.ndarray, 
    arcs: np.ndarray,
    r_cand: float = R_CAND,
    k_cand: int = K_CAND
) -> pd.DataFrame:
    """
    Generate road arc candidates for each GPS point in track.
    Strictly unsupervised: NEVER accesses the reference route.
    
    Returns:
        pd.DataFrame with columns:
        ['obs', 'arc', 'dist', 'hd', 'alen', 'speed', 'dt']
    """
    lat0 = float(track[:, 1].mean())
    nx, ny = project(nodes[:, 0], nodes[:, 1], lat0)
    s, e = arcs[:, 0], arcs[:, 1]
    ax, ay = nx[s], ny[s]
    bx, by = nx[e], ny[e]
    
    # Arc midpoints, lengths, and bearings
    mx, my = (ax + bx) / 2.0, (ay + by) / 2.0
    alen = np.hypot(bx - ax, by - ay)
    abear = np.arctan2(bx - ax, by - ay)

    # GPS points in metres
    px, py = project(track[:, 0], track[:, 1], lat0)
    n = len(px)

    # Kinematics: heading, delta_t, speed
    dx = np.r_[np.diff(px), 0.0]
    dy = np.r_[np.diff(py), 0.0]
    dt = np.r_[np.diff(track[:, 2]), 1.0]
    dt = np.where(dt <= 0, 1.0, dt)
    speed = np.hypot(dx, dy) / dt
    heading = np.arctan2(dx, dy)

    k = min(k_cand, len(arcs))
    kdtree = cKDTree(np.column_stack([mx, my]))
    _, idx = kdtree.query(np.column_stack([px, py]), k=k)
    idx = np.asarray(idx).reshape(n, k)

    obs = np.repeat(np.arange(n), k)
    arc = idx.ravel()

    # Perpendicular distance to arc line segment
    d = seg_dist(px[obs], py[obs], ax[arc], ay[arc], bx[arc], by[arc])
    # Heading difference wrapped into [0, pi]
    hd = np.abs(np.angle(np.exp(1j * (heading[obs] - abear[arc]))))

    keep = d <= r_cand
    return pd.DataFrame({
        "obs": obs[keep],
        "arc": arc[keep],
        "dist": d[keep],
        "hd": hd[keep],
        "alen": alen[arc[keep]],
        "speed": speed[obs[keep]],
        "dt": dt[obs[keep]],
    })

def compute_candidate_miss_rate(
    candidates_df: pd.DataFrame, 
    labels: np.ndarray
) -> Tuple[float, int, int]:
    """
    Calculate the Candidate Miss Rate:
    Fraction of valid ground-truth observations where the true arc is NOT present
    in the generated candidate set.
    """
    valid_mask = labels >= 0
    total_valid = int(valid_mask.sum())
    if total_valid == 0:
        return 0.0, 0, 0

    valid_obs = np.where(valid_mask)[0]
    true_arcs = labels[valid_obs]

    # Map obs -> set of candidate arcs
    grouped = candidates_df.groupby("obs")["arc"].apply(set).to_dict()

    misses = 0
    for o, true_arc in zip(valid_obs, true_arcs):
        cands = grouped.get(o, set())
        if true_arc not in cands:
            misses += 1

    miss_rate = misses / float(total_valid)
    return miss_rate, misses, total_valid
