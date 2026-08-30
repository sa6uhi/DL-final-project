"""Integration tests for the FastAPI scoring microservice."""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.serving.api import create_app
from src.utils.config import Config
from src.utils.seed import seed_everything


@pytest.fixture()
def client(tmp_path) -> TestClient:
    """Test client backed by a reference-model app with tmp config root."""
    seed_everything(42)
    cfg = Config(
        {
            "autoencoder": {
                "input_dim": 20,
                "anomaly_score": {"l1_gamma": 0.4},
            },
            "serving": {
                "model_path": str(tmp_path / "missing.pt"),
                "anomaly_const": 2.0,
            },
            "scoring": {
                "approve_threshold": 0.15,
                "block_threshold": 0.85,
                "escalate_threshold": 0.50,
            },
        },
        base_dir=tmp_path,
    )
    app = create_app(cfg)
    return TestClient(app)


@pytest.fixture()
def features() -> list[float]:
    """A 20-dimensional valid feature vector."""
    return np.random.default_rng(0).standard_normal(20).astype(float).tolist()


def test_health_ok(client: TestClient) -> None:
    """GET /health reports ok with reference model (not serialized)."""
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is False
    assert body["version"] != ""


def test_predict_valid_shape(client: TestClient, features: list[float]) -> None:
    """POST /predict scores a well-formed request end-to-end."""
    response = client.post("/predict", json={"features": features, "transaction_id": "tx-1"})
    assert response.status_code == 200
    body = response.json()
    assert body["transaction_id"] == "tx-1"
    assert 0.0 <= body["fraud_probability"] < 1.0
    assert body["anomaly_score"] >= 0.0
    assert body["decision"] in {"auto_approve", "auto_block", "escalate"}
    assert body["latency_ms"] >= 0.0


def test_predict_wrong_feature_count_is_422(client: TestClient) -> None:
    """Wrong feature dimensionality returns 422 with a clear message."""
    response = client.post("/predict", json={"features": [1.0, 2.0, 3.0]})
    assert response.status_code == 422
    assert any("features" in str(err).lower() for err in response.json()["detail"])


def test_predict_invalid_payload_type_is_422(client: TestClient) -> None:
    """Non-numeric payloads are rejected by the schema."""
    response = client.post("/predict", json={"features": ["a", "b"]})
    assert response.status_code == 422


def test_predict_empty_batch_rejected(client: TestClient) -> None:
    """Stream with no transactions is rejected."""
    response = client.post("/stream", json={"transactions": []})
    assert response.status_code == 422


def test_stream_returns_all_results(client: TestClient, features: list[float]) -> None:
    """POST /stream returns one result per transaction in order."""
    payload = {"transactions": [{"features": features} for _ in range(5)]}
    response = client.post("/stream", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 5
    assert len(body["results"]) == 5


def test_metrics_tracks_requests_and_latency(client: TestClient, features: list[float]) -> None:
    """GET /metrics reflects served requests with latency percentiles."""
    client.post("/predict", json={"features": features})
    client.post("/predict", json={"features": features})
    response = client.get("/metrics")
    assert response.status_code == 200
    body = response.json()
    assert body["requests_total"] == 2
    assert body["avg_latency_ms"] >= 0.0
    assert body["p99_latency_ms"] >= body["avg_latency_ms"]
    assert body["errors_total"] == 0


def test_route_validation_error_increments_errors_total(client: TestClient) -> None:
    """A scoring ValueError is surfaced as 422 and counted in /metrics."""
    response = client.post("/predict", json={"features": [0.1] * 8})
    assert response.status_code == 422
    metrics = client.get("/metrics").json()
    assert metrics["errors_total"] >= 1
    response = client.post("/stream", json={"transactions": [{"features": [0.1] * 8}]})
    assert response.status_code == 422
    metrics = client.get("/metrics").json()
    assert metrics["errors_total"] >= 2


def test_scorer_monotonic_probability(client: TestClient) -> None:
    """Larger residual scores imply larger fraud probabilities."""
    baseline = client.post("/predict", json={"features": [0.0] * 20}).json()
    extreme = client.post("/predict", json={"features": [1000.0] * 20}).json()
    assert extreme["anomaly_score"] > baseline["anomaly_score"]
    assert extreme["fraud_probability"] > baseline["fraud_probability"]
    assert {baseline["decision"], extreme["decision"]} <= {
        "auto_approve",
        "auto_block",
        "escalate",
    }


def test_value_error_handler_returns_400(client: TestClient) -> None:
    """The registered ValueError handler converts exceptions to HTTP 400."""
    import asyncio

    from starlette.requests import Request as StarletteRequest

    app = client.app
    handler = app.exception_handlers[ValueError]
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/predict",
        "headers": [],
        "query_string": b"",
    }
    response = asyncio.run(handler(StarletteRequest(scope), ValueError("boom")))
    assert response.status_code == 400
    assert b'"boom"' in response.body
