import pytest
from fastapi.testclient import TestClient
from pathsnap.api.app import app

client = TestClient(app)

def test_api_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "model" in data
    assert "available_trips_count" in data

def test_api_trips_list():
    response = client.get("/trips")
    assert response.status_code == 200
    data = response.json()
    assert "trips" in data
    assert isinstance(data["trips"], list)
    assert len(data["trips"]) > 0

def test_api_predict():
    # Fetch first available trip
    trips_res = client.get("/trips")
    trips = trips_res.json()["trips"]
    assert len(trips) > 0
    test_trip = trips[0]

    # Test predict endpoint with hybrid mode
    payload = {
        "trip_id": test_trip,
        "mode": "hybrid"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    res_data = response.json()
    assert res_data["trip_id"] == test_trip
    assert "route_continuity" in res_data
    assert "inference_seconds" in res_data
    assert "points" in res_data
    assert len(res_data["points"]) > 0
    assert "matched_arc" in res_data["points"][0]

def test_api_unknown_trip_404():
    response = client.post("/predict", json={"trip_id": "nonexistent_trip_99999"})
    assert response.status_code == 404
