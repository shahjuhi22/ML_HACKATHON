import os
import glob
from typing import Tuple, Optional, List
import numpy as np

def find_data_root() -> str:
    """
    Search standard locations for the data root directory containing
    either 'train' and 'student_test' or 'Student folder'.
    """
    env_root = os.environ.get("PATHSNAP_DATA")
    if env_root and os.path.isdir(env_root):
        return env_root

    candidates = [
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "Student folder"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "evaluator_test"),
    ]
    for c in candidates:
        if os.path.isdir(c) and (
            os.path.isdir(os.path.join(c, "train")) or 
            os.path.isdir(os.path.join(c, "student_test")) or
            any(f.endswith(".track") for root, _, files in os.walk(c) for f in files[:2])
        ):
            return c

    return "data"

def list_trips(folder: str) -> List[str]:
    """Return sorted list of valid trip IDs in the given folder."""
    if not os.path.isdir(folder):
        return []
    return sorted(d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d)))

def load_trip(folder: str, trip_id: str, with_route: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Load .track, .nodes, .arcs, and optionally .route for a given trip.
    
    Returns:
        track: np.ndarray of shape (N, 3) [lon, lat, timestamp]
        nodes: np.ndarray of shape (V, 2) [lon, lat]
        arcs:  np.ndarray of shape (E, 2) [from_node, to_node]
        route: Optional[np.ndarray] of shape (M,) [chronological arc_ids]
    """
    trip_dir = os.path.join(folder, trip_id)
    if not os.path.isdir(trip_dir):
        # Fallback if folder directly contains the trip files
        trip_dir = folder

    track_path = os.path.join(trip_dir, f"{trip_id}.track")
    nodes_path = os.path.join(trip_dir, f"{trip_id}.nodes")
    arcs_path = os.path.join(trip_dir, f"{trip_id}.arcs")
    route_path = os.path.join(trip_dir, f"{trip_id}.route")

    if not (os.path.isfile(track_path) and os.path.isfile(nodes_path) and os.path.isfile(arcs_path)):
        raise FileNotFoundError(f"Missing core trip files for trip {trip_id} in {trip_dir}")

    track = np.loadtxt(track_path, ndmin=2)
    nodes = np.loadtxt(nodes_path, ndmin=2)
    arcs = np.loadtxt(arcs_path, dtype=int, ndmin=2)

    route = None
    if with_route and os.path.isfile(route_path):
        route = np.loadtxt(route_path, dtype=int, ndmin=1)

    return track, nodes, arcs, route
