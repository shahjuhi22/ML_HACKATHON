import os
import pytest
import numpy as np
from pathsnap.data.loader import load_trip, list_trips, find_data_root
from pathsnap.data.projection import project, inverse_project, seg_dist, haversine_distance
from pathsnap.data.graph import RoadNetworkGraph
from pathsnap.data.candidates import generate_candidates, compute_candidate_miss_rate
from pathsnap.data.alignment import proxy_reference_labels, sequence_aligned_reference_labels
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
DATA_ROOT = os.environ.get("PATHSNAP_DATA_ROOT", os.path.join(ROOT, "data"))


DATA_DIR = os.path.join(DATA_ROOT, "student_test")

def test_load_trip():
    trips = list_trips(DATA_DIR)
    assert len(trips) == 10, f"Expected 10 student_test trips, got {len(trips)}"
    tid = trips[0]
    track, nodes, arcs, route = load_trip(DATA_DIR, tid, with_route=True)
    assert track.ndim == 2 and track.shape[1] == 3
    assert nodes.ndim == 2 and nodes.shape[1] == 2
    assert arcs.ndim == 2 and arcs.shape[1] == 2
    assert route is not None and len(route) > 0

def test_projection_invertibility():
    lon, lat = 8.6821, 50.1109
    x, y = project(lon, lat, lat0=lat)
    lon_rec, lat_rec = inverse_project(x, y, lat0=lat)
    assert np.isclose(lon, lon_rec, atol=1e-6)
    assert np.isclose(lat, lat_rec, atol=1e-6)

def test_seg_dist_accuracy():
    # Horizontal segment from (0, 0) to (10, 0)
    px, py = 5.0, 5.0
    ax, ay, bx, by = 0.0, 0.0, 10.0, 0.0
    d = seg_dist(np.array([px]), np.array([py]), np.array([ax]), np.array([ay]), np.array([bx]), np.array([by]))
    assert np.isclose(d[0], 5.0)

def test_road_network_graph():
    trips = list_trips(DATA_DIR)
    track, nodes, arcs, route = load_trip(DATA_DIR, trips[0], with_route=True)
    graph = RoadNetworkGraph(nodes, arcs)
    ncomp, _ = graph.get_connected_components()
    assert ncomp >= 1
    # Check consecutive route connection test
    if len(route) >= 2:
        is_conn = graph.can_transition(route[0], route[1])
        assert isinstance(is_conn, bool)

def test_candidates_and_miss_rate():
    trips = list_trips(DATA_DIR)
    track, nodes, arcs, route = load_trip(DATA_DIR, trips[0], with_route=True)
    cands = generate_candidates(track, nodes, arcs)
    assert len(cands) > 0
    assert set(["obs", "arc", "dist", "hd", "alen", "speed", "dt"]).issubset(cands.columns)

    labels = proxy_reference_labels(track, nodes, arcs, route)
    mr, misses, total = compute_candidate_miss_rate(cands, labels)
    assert 0.0 <= mr <= 0.20, f"Miss rate {mr} unexpectedly high"
