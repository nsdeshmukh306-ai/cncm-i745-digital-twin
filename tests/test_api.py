"""
Integration tests for the FastAPI backend (v4.0.0).

Exercises the public surface in-process with Starlette's TestClient — no live
service required. Auth is disabled for the test run so POST endpoints are
reachable without an API key.

Run inside the venv:  pytest tests/test_api.py -v
"""
import os
import warnings

import pytest

warnings.filterwarnings("ignore")
os.environ.pop("DT_API_KEY", None)   # keep POST auth open for the test client

from fastapi.testclient import TestClient  # noqa: E402

from api.main import app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health_reports_v4(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "healthy"
    assert body["version"] == "4.0.0"
    assert "uptime_seconds" in body
    assert "memory_used_mb" in body


def test_fba_simulate_returns_positive_growth(client):
    r = client.post("/fba/simulate", json={
        "glucose": -1.65, "oxygen": -2.0, "nh4": -1.0, "pi": -0.5,
    })
    assert r.status_code == 200
    body = r.json()
    assert body["feasible"] is True
    assert body["growth_rate"] > 0.0


def test_fba_simulate_rejects_out_of_range(client):
    """A10 range validation: positive glucose uptake is rejected (422)."""
    r = client.post("/fba/simulate", json={"glucose": 5.0})
    assert r.status_code == 422


def test_surrogate_predict_non_negative(client):
    r = client.post("/surrogate/predict", json={
        "glucose": -1.65, "oxygen": -2.0, "nh4": -1.0, "pi": -0.5,
    })
    assert r.status_code == 200
    body = r.json()
    assert body["predicted_growth_rate"] >= 0.0
    assert "inference_time_s" in body
