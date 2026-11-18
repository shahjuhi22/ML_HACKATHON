import numpy as np
from typing import Tuple, List, Optional
from pathsnap.data.projection import project, seg_dist

def proxy_reference_labels(
    track: np.ndarray, 
    nodes: np.ndarray, 
    arcs: np.ndarray, 
    route: np.ndarray,
    r_cand: float = 50.0
) -> np.ndarray:
    """
    Standard Spatial Nearest-Arc Proxy Labeling:
    Assigns each GPS observation to the nearest arc in np.unique(route) within r_cand metres.
    
    LIMITATION:
    Pure spatial proximity ignores chronological order and network topology. If the route loops
    or self-intersects, a GPS point may be falsely assigned to an arc traversed much later.
    """
    if route is None or len(route) == 0:
        return np.full(len(track), -1, dtype=int)

    lat0 = float(track[:, 1].mean())
    nx, ny = project(nodes[:, 0], nodes[:, 1], lat0)
    s, e = arcs[:, 0], arcs[:, 1]
    px, py = project(track[:, 0], track[:, 1], lat0)

    best_dist = np.full(len(px), np.inf)
    labels = np.full(len(px), -1, dtype=int)

    unique_arcs = np.unique(route)
    for a in unique_arcs:
        if a < 0 or a >= len(arcs):
            continue
        d = seg_dist(px, py, nx[s[a]], ny[s[a]], nx[e[a]], ny[e[a]])
        closer = d < best_dist
        best_dist[closer] = d[closer]
        labels[closer] = a

    labels[best_dist > r_cand] = -1
    return labels

def sequence_aligned_reference_labels(
    track: np.ndarray, 
    nodes: np.ndarray, 
    arcs: np.ndarray, 
    route: np.ndarray,
    r_cand: float = 50.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Chronological Sequence-Aware Route Alignment:
    Aligns GPS observations monotonically along the ordered sequence of traversed route arcs.
    Enforces progress forward through route arcs [a_0, a_1, ..., a_{M-1}] using dynamic programming,
    preventing temporal jumps to intersecting or parallel road segments.
    
    Returns:
        aligned_labels: np.ndarray of shape (N,) with arc IDs (-1 if unaligned/far)
        route_indices: np.ndarray of shape (N,) with index into route array
    """
    if route is None or len(route) == 0:
        return np.full(len(track), -1, dtype=int), np.full(len(track), -1, dtype=int)

    N = len(track)
    M = len(route)

    lat0 = float(track[:, 1].mean())
    nx, ny = project(nodes[:, 0], nodes[:, 1], lat0)
    s, e = arcs[:, 0], arcs[:, 1]
    px, py = project(track[:, 0], track[:, 1], lat0)

    # Compute distance matrix between all GPS points and all route arcs
    # Shape: (N, M)
    dist_matrix = np.zeros((N, M), dtype=np.float32)
    for m, a in enumerate(route):
        if a < 0 or a >= len(arcs):
            dist_matrix[:, m] = 1e6
        else:
            dist_matrix[:, m] = seg_dist(px, py, nx[s[a]], ny[s[a]], nx[e[a]], ny[e[a]])

    # Monotonic forward alignment via dynamic programming:
    # State: at observation t, vehicle is on route index m (m >= m_{t-1})
    # We penalize distance + small penalty for advancing forward
    aligned_labels = np.full(N, -1, dtype=int)
    route_indices = np.full(N, -1, dtype=int)

    # Greedy forward tracker with lookahead window
    curr_m = 0
    WINDOW = min(15, M)  # max lookahead arcs in route
    for t in range(N):
        window_end = min(M, curr_m + WINDOW)
        window_dists = dist_matrix[t, curr_m:window_end]
        
        best_rel_idx = int(np.argmin(window_dists))
        best_d = float(window_dists[best_rel_idx])

        if best_d <= r_cand:
            curr_m = curr_m + best_rel_idx
            aligned_labels[t] = route[curr_m]
            route_indices[t] = curr_m
        else:
            # If no arc in forward window is within r_cand, check nearest overall
            min_all = float(np.min(dist_matrix[t, :]))
            if min_all <= r_cand:
                idx_all = int(np.argmin(dist_matrix[t, :]))
                aligned_labels[t] = route[idx_all]
                route_indices[t] = idx_all
                curr_m = max(curr_m, idx_all)

    return aligned_labels, route_indices


def monotonic_route_labels(
    track: np.ndarray,
    nodes: np.ndarray,
    arcs: np.ndarray,
    route: np.ndarray,
    r_cand: float = 50.0,
    skip_penalty: float = 2.0,
    max_skip: int = 8,
    cap: float = 100.0,
) -> np.ndarray:
    """
    Chronological route alignment by dynamic programming (used for training labels
    and scoring only, never for prediction).

    The reference route is an ordered list of arcs. Each GPS point t is assigned a
    route position m_t with m_t >= m_{t-1} (the vehicle never moves backwards along
    its own route). Cost = sum of point-to-arc distances (capped) plus a small
    penalty per route arc skipped. Out-and-back trips therefore get the arc of the
    correct travel direction, which the nearest-arc proxy cannot tell apart.
    Points whose assigned arc is more than r_cand metres away are left unlabelled (-1).
    """
    N = len(track)
    if route is None or len(route) == 0 or N == 0:
        return np.full(N, -1, dtype=int)
    route = np.asarray(route, dtype=int)
    valid = (route >= 0) & (route < len(arcs))
    route = route[valid]
    M = len(route)
    if M == 0:
        return np.full(N, -1, dtype=int)

    lat0 = float(track[:, 1].mean())
    nx, ny = project(nodes[:, 0], nodes[:, 1], lat0)
    s, e = arcs[:, 0], arcs[:, 1]
    px, py = project(track[:, 0], track[:, 1], lat0)

    D = np.empty((N, M), dtype=np.float32)
    for m, a in enumerate(route):
        D[:, m] = seg_dist(px, py, nx[s[a]], ny[s[a]], nx[e[a]], ny[e[a]])
    C = np.minimum(D, cap)

    INF = np.float32(1e18)
    V = C[0] + skip_penalty * np.arange(M, dtype=np.float32)  # starting late costs skips
    back = np.zeros((N, M), dtype=np.int32)
    back[0] = np.arange(M)
    for t in range(1, N):
        best = V.copy()
        arg = np.arange(M, dtype=np.int32)
        for k in range(1, max_skip + 1):
            shifted = np.full(M, INF, dtype=np.float32)
            shifted[k:] = V[:-k] + skip_penalty * (k - 1)
            better = shifted < best
            best[better] = shifted[better]
            arg[better] = (np.arange(M, dtype=np.int32) - k)[better]
        V = best + C[t]
        back[t] = arg

    pos = np.empty(N, dtype=np.int32)
    pos[-1] = int(np.argmin(V))
    for t in range(N - 1, 0, -1):
        pos[t - 1] = back[t, pos[t]]
    labels = route[pos].astype(int)
    labels[D[np.arange(N), pos] > r_cand] = -1
    return labels
