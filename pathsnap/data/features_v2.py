"""
Extended candidate features (v2).

Strictly unsupervised: uses only .track, .nodes and .arcs. Never reads .route.
Adds context around each GPS point so the classifier can tell parallel and
crossing roads apart, which a single noisy point cannot do.
"""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from pathsnap.data.projection import project

R_CAND = 50.0
K_CAND = 40
WIN = 5            # neighbouring GPS points on each side used for context
WIN_MAX_DT = 15.0  # ignore neighbours more than this many seconds away

CORE_FEATS = ["dist", "hd", "alen", "speed", "dt"]
V2_FEATS = CORE_FEATS + [
    "dist_rank", "dist_gap", "dist_ratio", "n_cand",
    "hd_s", "hd_s_undir", "proj_t", "inside",
    "win_dist_mean", "win_dist_max", "win_dist_gap", "win_rank",
    "head_outdeg", "tail_indeg", "stationary",
]


def _seg_dist_t(px, py, ax, ay, bx, by):
    abx, aby = bx - ax, by - ay
    denom = abx ** 2 + aby ** 2 + 1e-12
    t_raw = ((px - ax) * abx + (py - ay) * aby) / denom
    t = np.clip(t_raw, 0.0, 1.0)
    return np.hypot(px - (ax + t * abx), py - (ay + t * aby)), t_raw


def generate_candidates_v2(track, nodes, arcs, r_cand=R_CAND, k_cand=K_CAND):
    lat0 = float(track[:, 1].mean())
    nx, ny = project(nodes[:, 0], nodes[:, 1], lat0)
    s, e = arcs[:, 0], arcs[:, 1]
    ax, ay, bx, by = nx[s], ny[s], nx[e], ny[e]
    mx, my = (ax + bx) / 2.0, (ay + by) / 2.0
    alen = np.hypot(bx - ax, by - ay)
    abear = np.arctan2(bx - ax, by - ay)

    px, py = project(track[:, 0], track[:, 1], lat0)
    ts = track[:, 2].astype(float)
    n = len(px)

    # Point kinematics (same definitions as the core features)
    dx = np.r_[np.diff(px), 0.0]
    dy = np.r_[np.diff(py), 0.0]
    dt = np.r_[np.diff(ts), 1.0]
    dt = np.where(dt <= 0, 1.0, dt)
    speed = np.hypot(dx, dy) / dt
    heading = np.arctan2(dx, dy)

    # Smoothed heading: displacement between points WIN steps before and after
    lo = np.clip(np.arange(n) - WIN, 0, n - 1)
    hi = np.clip(np.arange(n) + WIN, 0, n - 1)
    heading_s = np.arctan2(px[hi] - px[lo], py[hi] - py[lo])

    k = min(k_cand, len(arcs))
    _, idx = cKDTree(np.column_stack([mx, my])).query(np.column_stack([px, py]), k=k)
    idx = np.asarray(idx).reshape(n, k)
    obs = np.repeat(np.arange(n), k)
    arc = idx.ravel()

    d, t_raw = _seg_dist_t(px[obs], py[obs], ax[arc], ay[arc], bx[arc], by[arc])
    keep = d <= r_cand
    obs, arc, d, t_raw = obs[keep], arc[keep], d[keep], t_raw[keep]

    df = pd.DataFrame({
        "obs": obs, "arc": arc, "dist": d,
        "hd": np.abs(np.angle(np.exp(1j * (heading[obs] - abear[arc])))),
        "alen": alen[arc], "speed": speed[obs], "dt": dt[obs],
    })
    if len(df) == 0:
        for f in V2_FEATS:
            if f not in df:
                df[f] = []
        return df

    g = df.groupby("obs")["dist"]
    df["dist_rank"] = g.rank(method="first") - 1
    dmin = g.transform("min")
    df["dist_gap"] = df["dist"] - dmin
    df["dist_ratio"] = (dmin + 1.0) / (df["dist"] + 1.0)
    df["n_cand"] = g.transform("size")

    hd_s = np.abs(np.angle(np.exp(1j * (heading_s[obs] - abear[arc]))))
    df["hd_s"] = hd_s
    df["hd_s_undir"] = np.minimum(hd_s, np.pi - hd_s)
    df["proj_t"] = np.clip(t_raw, -2.0, 3.0)
    df["inside"] = ((t_raw >= 0) & (t_raw <= 1)).astype(np.int8)

    # Window distance: how close the neighbouring GPS points are to this arc
    acc = np.zeros(len(df))
    cnt = np.zeros(len(df))
    dmax = np.zeros(len(df))
    for o in range(-WIN, WIN + 1):
        if o == 0:
            continue
        j = obs + o
        ok = (j >= 0) & (j < n)
        jj = np.clip(j, 0, n - 1)
        ok &= np.abs(ts[jj] - ts[obs]) <= WIN_MAX_DT
        dj, _ = _seg_dist_t(px[jj], py[jj], ax[arc], ay[arc], bx[arc], by[arc])
        acc += np.where(ok, dj, 0.0)
        cnt += ok
        dmax = np.maximum(dmax, np.where(ok, dj, 0.0))
    wmean = np.where(cnt > 0, acc / np.maximum(cnt, 1), df["dist"].values)
    df["win_dist_mean"] = wmean
    df["win_dist_max"] = np.where(cnt > 0, dmax, df["dist"].values)
    df["win_dist_gap"] = df["win_dist_mean"] - df.groupby("obs")["win_dist_mean"].transform("min")
    df["win_rank"] = df.groupby("obs")["win_dist_mean"].rank(method="first") - 1

    outdeg = np.bincount(s, minlength=len(nodes))
    indeg = np.bincount(e, minlength=len(nodes))
    df["head_outdeg"] = outdeg[e[arc]]
    df["tail_indeg"] = indeg[s[arc]]
    df["stationary"] = (df["speed"] < 1.0).astype(np.int8)
    return df
