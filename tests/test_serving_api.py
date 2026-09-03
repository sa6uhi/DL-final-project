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


def test_predict_non_finite_features_rejected(client: TestClient) -> None:
    """NaN or infinite features are rejected by schema validation."""
    from pydantic import ValidationError
    from src.serving.schemas import ExplainRequest, PredictionRequest

    with pytest.raises(ValidationError, match="finite"):
        PredictionRequest(features=[float("nan")] * 20)

    with pytest.raises(ValidationError, match="finite"):
        PredictionRequest(features=[float("inf")] * 20)

    with pytest.raises(ValidationError, match="finite"):
        ExplainRequest(features=[float("nan")] * 20)

    raw_payload = '{"features": [' + ", ".join(["NaN"] * 20) + "]}"
    response = client.post(
        "/predict", content=raw_payload, headers={"content-type": "application/json"}
    )
    assert response.status_code in {400, 422}


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


@pytest.fixture()
def gated_client(tmp_path) -> TestClient:
    """Test client with a learned hybrid gate checkpoint on disk."""
    import torch

    from src.models.hybrid_gating import LearnedHybridGate, PercentileNormalizer
    from src.training.train_hybrid_gating import save_checkpoint

    seed_everything(42)
    gate = LearnedHybridGate(input_dim=2, hidden_dims=[16, 8], dropout=0.1)
    gate.eval()
    normalizer = PercentileNormalizer(percentile=99.0).fit(
        torch.tensor([0.1, 0.5, 1.0, 2.0], dtype=torch.float32)
    )
    gate_path = tmp_path / "hybrid_gating.pt"
    save_checkpoint(gate, normalizer, gate_path)
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
            "hybrid_gating": {"learned": {"checkpoint_path": str(gate_path)}},
        },
        base_dir=tmp_path,
    )
    return TestClient(create_app(cfg))


def test_predict_without_gate_ignores_ft_probability(
    client: TestClient, features: list[float]
) -> None:
    """DAE-only /predict accepts ft_probability but does not fuse it."""
    body = client.post("/predict", json={"features": features, "ft_probability": 0.9}).json()
    assert body["gate_used"] is False
    assert client.get("/health").json()["gate_loaded"] is False


def test_decide_maps_probability_to_triage_band() -> None:
    """_decide routes low/mid/high probabilities to the right decisions."""
    from src.serving.api import _decide

    assert _decide(0.01, 0.15, 0.85, 0.50) == ("auto_approve", False)
    assert _decide(0.99, 0.15, 0.85, 0.50) == ("auto_block", True)
    assert _decide(0.60, 0.15, 0.85, 0.50) == ("escalate", True)
    assert _decide(0.30, 0.15, 0.85, 0.50) == ("escalate", False)


def test_score_batch_posterior_size_mismatch_raises(client: TestClient) -> None:
    """Mismatched ft_probabilities batch size raises ValueError."""
    scorer = client.app.state.scorer
    with pytest.raises(ValueError, match="match the batch size"):
        scorer.score_batch([[0.0] * 20, [0.0] * 20], [0.5])


def test_health_reports_gate_loaded(gated_client: TestClient) -> None:
    """GET /health reflects the loaded hybrid gate."""
    body = gated_client.get("/health").json()
    assert body["gate_loaded"] is True


def test_predict_with_gate_fuses_ft_probability(
    gated_client: TestClient, features: list[float]
) -> None:
    """Gate-loaded /predict fuses the FT posterior into the probability."""
    low = gated_client.post("/predict", json={"features": features, "ft_probability": 0.01}).json()
    high = gated_client.post("/predict", json={"features": features, "ft_probability": 0.99}).json()
    assert low["gate_used"] is True
    assert high["gate_used"] is True
    assert 0.0 <= low["fraud_probability"] <= 1.0
    assert high["fraud_probability"] > low["fraud_probability"]


def test_predict_with_gate_missing_ft_probability_is_422(
    gated_client: TestClient, features: list[float]
) -> None:
    """Gate-loaded /predict without ft_probability is rejected."""
    response = gated_client.post("/predict", json={"features": features})
    assert response.status_code == 422


def test_stream_with_gate_fuses_batch(gated_client: TestClient, features: list[float]) -> None:
    """Gate-loaded /stream fuses per-transaction FT posteriors in order."""
    payload = {
        "transactions": [
            {"features": features, "ft_probability": 0.1},
            {"features": features, "ft_probability": 0.9},
        ]
    }
    response = gated_client.post("/stream", json=payload)
    assert response.status_code == 200
    results = response.json()["results"]
    assert all(r["gate_used"] is True for r in results)
    assert results[1]["fraud_probability"] > results[0]["fraud_probability"]


def test_stream_with_gate_partial_ft_probability_is_422(
    gated_client: TestClient, features: list[float]
) -> None:
    """Gate-loaded /stream with mixed ft_probability presence is rejected."""
    payload = {
        "transactions": [
            {"features": features, "ft_probability": 0.1},
            {"features": features},
        ]
    }
    response = gated_client.post("/stream", json=payload)
    assert response.status_code == 422


def test_explain_valid_features(client: TestClient, features: list[float]) -> None:
    """POST /explain returns ordered risk drivers for a transaction."""
    response = client.post(
        "/explain", json={"features": features, "transaction_id": "tx-exp", "top_k": 3}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["transaction_id"] == "tx-exp"
    assert len(body["top_drivers"]) == 3
    assert body["method"] == "dae_reconstruction_residual"
    assert body["latency_ms"] >= 0.0
    assert body["top_drivers"][0]["attribution"] >= body["top_drivers"][1]["attribution"]


def test_explain_wrong_feature_count_is_422(client: TestClient) -> None:
    """POST /explain with mismatched feature count returns 422."""
    response = client.post("/explain", json={"features": [1.0] * 8})
    assert response.status_code == 422


def test_explain_delegates_to_shap_when_available(
    client: TestClient, features: list[float], monkeypatch: pytest.MonkeyPatch
) -> None:
    """POST /explain uses shap_explainer when the module is present."""
    import sys
    import types

    fake_mod = types.ModuleType("src.explainability.shap_explainer")
    fake_mod.explain_transaction = lambda feats, top_k=5, ft_probability=None: [  # type: ignore
        {"feature_name": "Amount", "attribution": 0.42, "value": 150.0}
    ]
    monkeypatch.setitem(sys.modules, "src.explainability.shap_explainer", fake_mod)

    response = client.post("/explain", json={"features": features, "top_k": 1})
    assert response.status_code == 200
    body = response.json()
    assert body["method"] == "shap"
    assert body["top_drivers"][0]["feature_name"] == "Amount"
    assert body["top_drivers"][0]["attribution"] == 0.42
