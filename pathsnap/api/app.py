import os
import json
import time
import logging
from typing import Optional, List, Dict, Any
import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pathsnap.data.loader import load_trip, list_trips, find_data_root
from pathsnap.data.candidates import generate_candidates, CORE_FEATS
from pathsnap.data.alignment import proxy_reference_labels, sequence_aligned_reference_labels
from pathsnap.models.evaluate import compute_route_continuity
from pathsnap.hmm.decoder import HMMViterbiDecoder

# Paths from environment or sensible defaults
DATA_DIR = os.environ.get("PATHSNAP_DATA", "")
if not DATA_DIR or not os.path.isdir(DATA_DIR):
    DATA_DIR = os.path.join(find_data_root(), "student_test")
    if not os.path.isdir(DATA_DIR):
        DATA_DIR = find_data_root()

ART_DIR = os.environ.get("PATHSNAP_ARTIFACTS", "artifacts")
if not os.path.isdir(ART_DIR):
    # Fallback to gagana/artifacts or hitashree/pathsnap_api/artifacts
    for cand in ["gagana/artifacts", "hitashree/pathsnap_api/artifacts"]:
        if os.path.isdir(cand):
            ART_DIR = cand
            break

LOG_FILE = os.environ.get("PATHSNAP_LOG", "api_requests.log")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
    ]
)
logger = logging.getLogger("pathsnap_api")

# Load model configuration and trained model
best_cfg_path = os.path.join(ART_DIR, "configs", "best.json")
if os.path.isfile(best_cfg_path):
    with open(best_cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
else:
    cfg = {"best_model": "hist_gradient_boosting", "features": CORE_FEATS}

model_name = cfg.get("best_model", "hist_gradient_boosting")
model_path = os.path.join(ART_DIR, "models", f"{model_name}.joblib")
if not os.path.isfile(model_path):
    # Check alternate model files
    for alt in ["hgb_tuned.joblib", "hist_gradient_boosting.joblib", "logistic_regression.joblib", "random_forest.joblib"]:
        alt_p = os.path.join(ART_DIR, "models", alt)
        if os.path.isfile(alt_p):
            model_path = alt_p
            model_name = alt.replace(".joblib", "")
            break

try:
    model = joblib.load(model_path)
    logger.info(f"Loaded model '{model_name}' from {model_path}")
except Exception as e:
    logger.warning(f"Could not load model from {model_path}: {e}")
    model = None

# Initialize default HMM decoder
hmm_decoder = HMMViterbiDecoder(sigma=10.0, p_stay=0.70, p_conn=0.25, p_other=1e-4)

# Create FastAPI application
app = FastAPI(
    title="PathSnap Map-Matching & Route Reconstruction API",
    description="Backend API serving candidate classification and HMM Viterbi route reconstruction",
    version="1.0.0"
)

# Enable CORS for external access (Laptop 1 -> Laptop 2)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictRequest(BaseModel):
    trip_id: str
    mode: Optional[str] = "hybrid"  # "hybrid", "classifier_only", "hmm_only"
    data_partition: Optional[str] = None  # optionally override data folder

@app.get("/health")
def health():
    trips = list_trips(DATA_DIR)
    return {
        "status": "ok",
        "service": "PathSnap API",
        "model": model_name,
        "model_loaded": model is not None,
        "data_directory": DATA_DIR,
        "available_trips_count": len(trips)
    }

@app.get("/trips")
def get_trips(partition: Optional[str] = None):
    target_dir = DATA_DIR
    if partition:
        cand_p = os.path.join(os.path.dirname(DATA_DIR), partition)
        if os.path.isdir(cand_p):
            target_dir = cand_p
    trips = list_trips(target_dir)
    return {"trips": trips, "folder": target_dir}

@app.post("/predict")
def predict(req: PredictRequest, request: Request):
    target_dir = DATA_DIR
    if req.data_partition:
        cand_p = os.path.join(os.path.dirname(DATA_DIR), req.data_partition)
        if os.path.isdir(cand_p):
            target_dir = cand_p

    available = list_trips(target_dir)
    if req.trip_id not in available:
        logger.warning(f"Rejected unknown trip '{req.trip_id}' from {request.client.host if request.client else 'unknown'}")
        raise HTTPException(status_code=404, detail=f"Trip '{req.trip_id}' not found in {target_dir}")

    t0 = time.time()

    # 1. Prediction Pipeline: strictly unsupervised, loads ONLY track, nodes, arcs
    track, nodes, arcs, _ = load_trip(target_dir, req.trip_id, with_route=False)
    N = len(track)
    cands = generate_candidates(track, nodes, arcs)

    # 2. Compute probabilities if model loaded
    probs = None
    if len(cands) > 0 and model is not None:
        X = cands[CORE_FEATS].values
        probs = model.predict_proba(X)[:, 1]
        cands["p"] = probs
    else:
        cands["p"] = 0.5

    # 3. Route Reconstruction Decoding
    if req.mode == "classifier_only" or len(cands) == 0:
        if len(cands) > 0:
            best_idx = cands.groupby("obs")["p"].idxmax()
            best_cands = cands.loc[best_idx]
            pred_dict = dict(zip(best_cands["obs"].astype(int), best_cands["arc"].astype(int)))
            pred_arcs = np.array([pred_dict.get(i, -1) for i in range(N)])
        else:
            pred_arcs = np.full(N, -1, dtype=int)
    elif req.mode == "hmm_only":
        # Pure distance-based HMM
        pure_hmm = HMMViterbiDecoder(sigma=10.0, classifier_weight=0.0)
        pred_arcs = pure_hmm.decode(cands, arcs, track[:, 2], N)
    else:
        # Default: Hybrid (Classifier Posterior + HMM Road Network Connectivity)
        pred_arcs = hmm_decoder.decode(cands, arcs, track[:, 2], N, classifier_probs=probs)

    inference_seconds = round(time.time() - t0, 3)

    # 4. Server-Side Evaluation (Never exposed to model; ground truth route stays on server)
    top1_acc = None
    continuity = float(compute_route_continuity(pred_arcs, arcs))
    try:
        _, _, _, route = load_trip(target_dir, req.trip_id, with_route=True)
        if route is not None:
            labels = proxy_reference_labels(track, nodes, arcs, route)
            valid = labels >= 0
            if valid.any():
                top1_acc = float((pred_arcs[valid] == labels[valid]).mean())
    except Exception as e:
        logger.debug(f"Ground truth scoring not available for trip {req.trip_id}: {e}")

    client_ip = request.client.host if request.client else "unknown"
    logger.info(
        f"PREDICT trip={req.trip_id} mode={req.mode} obs={N} time={inference_seconds}s "
        f"top1_acc={top1_acc} continuity={continuity:.4f} client={client_ip}"
    )

    # Prepare response payload
    points_out = []
    for i in range(N):
        points_out.append({
            "obs": i,
            "lat": float(track[i, 1]),
            "lon": float(track[i, 0]),
            "timestamp": float(track[i, 2]),
            "matched_arc": int(pred_arcs[i]),
        })

    return {
        "trip_id": req.trip_id,
        "mode": req.mode,
        "model": model_name,
        "num_observations": N,
        "inference_seconds": inference_seconds,
        "route_continuity": continuity,
        "server_side_accuracy": top1_acc,
        "points": points_out
    }

@app.get("/trip_details/{trip_id}")
def get_trip_details(trip_id: str):
    """
    Returns road network geometry and GPS observations for visualization in the UI.
    """
    available = list_trips(DATA_DIR)
    if trip_id not in available:
        raise HTTPException(status_code=404, detail="Trip not found")

    # Ground truth stays on the server: the reference route is never sent to the client.
    track, nodes, arcs, _ = load_trip(DATA_DIR, trip_id, with_route=False)

    gps_coords = [[float(lat), float(lon)] for lon, lat in track[:, :2]]
    min_lat, max_lat = float(track[:, 1].min()), float(track[:, 1].max())
    min_lon, max_lon = float(track[:, 0].min()), float(track[:, 0].max())
    route_lines = []

    return {
        "trip_id": trip_id,
        "center": [(min_lat + max_lat) / 2.0, (min_lon + max_lon) / 2.0],
        "bounds": [[min_lat, min_lon], [max_lat, max_lon]],
        "gps_coords": gps_coords,
        "route_lines": route_lines
    }
