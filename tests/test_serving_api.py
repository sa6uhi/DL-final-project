"""Integration tests for the FastAPI scoring microservice."""

from __future__ import annotations

from pathlib import Path
from typing import Any

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
    assert body["p90_latency_ms"] >= 0.0
    assert body["p99_latency_ms"] >= body["p90_latency_ms"]
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
    gate = LearnedHybridGate(input_dim=4, hidden_dims=[16, 8], dropout=0.1)
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
    """Gate-loaded /predict fuses FT and historical context."""
    low = gated_client.post(
        "/predict",
        json={
            "features": features,
            "ft_probability": 0.01,
            "history_density": 0.4,
            "history_amount_intensity": 2.0,
        },
    ).json()

    high = gated_client.post(
        "/predict",
        json={
            "features": features,
            "ft_probability": 0.99,
            "history_density": 0.4,
            "history_amount_intensity": 2.0,
        },
    ).json()

    assert low["gate_used"] is True
    assert high["gate_used"] is True
    assert 0.0 <= low["fraud_probability"] <= 1.0
    assert 0.0 <= high["fraud_probability"] <= 1.0


def test_predict_with_gate_missing_ft_probability_is_422(
    gated_client: TestClient, features: list[float]
) -> None:
    """Gate-loaded /predict without ft_probability is rejected."""
    response = gated_client.post(
        "/predict",
        json={
            "features": features,
            "history_density": 0.4,
            "history_amount_intensity": 2.0,
        },
    )

    assert response.status_code == 422


def test_predict_with_four_input_gate_missing_velocity_is_422(
    gated_client: TestClient, features: list[float]
) -> None:
    """A 4-input learned gate requires historical velocity context."""
    response = gated_client.post(
        "/predict",
        json={
            "features": features,
            "ft_probability": 0.5,
        },
    )

    assert response.status_code == 422
    assert "history_density" in str(response.json()["detail"])


def test_predict_with_partial_velocity_is_422(
    gated_client: TestClient, features: list[float]
) -> None:
    """History density and amount intensity must be supplied together."""
    response = gated_client.post(
        "/predict",
        json={
            "features": features,
            "ft_probability": 0.5,
            "history_density": 0.4,
        },
    )

    assert response.status_code == 422


def test_stream_with_gate_fuses_batch(gated_client: TestClient, features: list[float]) -> None:
    """4-input gate accepts aligned FT and velocity context for a batch."""
    payload = {
        "transactions": [
            {
                "features": features,
                "ft_probability": 0.1,
                "history_density": 0.2,
                "history_amount_intensity": 1.0,
            },
            {
                "features": features,
                "ft_probability": 0.9,
                "history_density": 0.8,
                "history_amount_intensity": 3.0,
            },
        ]
    }

    response = gated_client.post("/stream", json=payload)

    assert response.status_code == 200

    results = response.json()["results"]

    assert len(results) == 2
    assert all(result["gate_used"] is True for result in results)
    assert all(0.0 <= result["fraud_probability"] <= 1.0 for result in results)


def test_stream_with_gate_partial_velocity_is_422(
    gated_client: TestClient, features: list[float]
) -> None:
    """Batch velocity context must be complete for every transaction."""
    payload = {
        "transactions": [
            {
                "features": features,
                "ft_probability": 0.1,
                "history_density": 0.2,
                "history_amount_intensity": 1.0,
            },
            {
                "features": features,
                "ft_probability": 0.9,
            },
        ]
    }

    response = gated_client.post("/stream", json=payload)

    assert response.status_code == 422


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
    fake_mod.explain_transaction = (
        lambda feats, model=None, background=None, top_k=5, ft_probability=None, l1_gamma=0.4: [
            {"feature_name": "Amount", "attribution": 0.42, "value": 150.0}
        ]
    )
    monkeypatch.setitem(sys.modules, "src.explainability.shap_explainer", fake_mod)

    response = client.post("/explain", json={"features": features, "top_k": 1})
    assert response.status_code == 200
    body = response.json()
    assert body["method"] == "shap"
    assert body["top_drivers"][0]["feature_name"] == "Amount"
    assert body["top_drivers"][0]["attribution"] == 0.42


def test_explain_shap_exception_falls_back_to_dae(
    client: TestClient, features: list[float], monkeypatch: pytest.MonkeyPatch
) -> None:
    """If shap_explainer raises an exception, /explain falls back gracefully."""
    import sys
    import types

    fake_mod = types.ModuleType("src.explainability.shap_explainer")

    def faulty_explainer(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        raise TypeError("Unexpected signature")

    fake_mod.explain_transaction = faulty_explainer  # type: ignore
    monkeypatch.setitem(sys.modules, "src.explainability.shap_explainer", fake_mod)

    response = client.post("/explain", json={"features": features, "top_k": 2})
    assert response.status_code == 200
    body = response.json()
    assert body["method"] == "dae_reconstruction_residual"
    assert len(body["top_drivers"]) == 2


def test_scorer_score_rejects_nan_directly(client: TestClient) -> None:
    """Scorer.score raises ValueError on NaN features."""
    scorer = client.app.state.scorer
    with pytest.raises(ValueError, match="finite"):
        scorer.score([float("nan")] * scorer.input_dim)


def test_scorer_score_batch_rejects_empty_directly(client: TestClient) -> None:
    """Scorer.score_batch raises ValueError on empty list."""
    scorer = client.app.state.scorer
    with pytest.raises(ValueError, match="empty"):
        scorer.score_batch([])


def test_scorer_score_batch_rejects_nan_directly(client: TestClient) -> None:
    """Scorer.score_batch raises ValueError on NaN features."""
    scorer = client.app.state.scorer
    with pytest.raises(ValueError, match="finite"):
        scorer.score_batch([[float("nan")] * scorer.input_dim])


def test_serving_require_checkpoint_raises_when_missing(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """build_scorer raises FileNotFoundError when SERVING_REQUIRE_CHECKPOINT is set."""
    from src.serving.api import build_scorer

    cfg = Config(
        {
            "autoencoder": {"input_dim": 20, "anomaly_score": {"l1_gamma": 0.4}},
            "serving": {"model_path": str(tmp_path / "absent.pt"), "anomaly_const": 2.0},
            "scoring": {
                "approve_threshold": 0.1,
                "block_threshold": 0.9,
                "escalate_threshold": 0.5,
            },
        }
    )
    monkeypatch.setenv("SERVING_REQUIRE_CHECKPOINT", "1")
    with pytest.raises(FileNotFoundError, match="SERVING_REQUIRE_CHECKPOINT"):
        build_scorer(cfg)


def test_predict_uses_matching_conformal_artifact(
    tmp_path: Path,
    features: list[float],
) -> None:
    """Matching configured alpha enables conformal serving decisions."""
    import json

    artifact_path = tmp_path / "serving_threshold.json"
    artifact_path.write_text(
        json.dumps(
            {
                "method": "split_conformal",
                "alpha": 0.01,
                "threshold": 0.975174069404602,
            }
        ),
        encoding="utf-8",
    )

    cfg = Config(
        {
            "autoencoder": {
                "input_dim": 20,
                "anomaly_score": {"l1_gamma": 0.4},
            },
            "serving": {
                "model_path": str(tmp_path / "missing.pt"),
                "anomaly_const": 2.0,
                "conformal_path": str(artifact_path),
            },
            "scoring": {
                "approve_threshold": 0.15,
                "block_threshold": 0.85,
                "escalate_threshold": 0.50,
            },
            "evaluation": {
                "conformal": {
                    "alpha": 0.01,
                }
            },
        },
        base_dir=tmp_path,
    )

    test_client = TestClient(create_app(cfg))

    health = test_client.get("/health")
    assert health.status_code == 200
    assert health.json()["conformal_loaded"] is True

    response = test_client.post(
        "/predict",
        json={
            "features": features,
            "transaction_id": "conformal-test",
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["transaction_id"] == "conformal-test"
    assert body["conformal_used"] is True
    assert body["conformal_alpha"] == pytest.approx(0.01)
    assert body["conformal_threshold"] == pytest.approx(0.975174069404602)
    assert body["conformal_set"] is not None
    assert body["decision"] in {
        "auto_approve",
        "auto_block",
        "human_review",
    }


def test_serving_require_ft_model_raises_when_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Configured missing FT checkpoint can be made a startup failure."""
    from src.serving.api import build_scorer

    cfg = Config(
        {
            "autoencoder": {
                "input_dim": 20,
                "anomaly_score": {"l1_gamma": 0.4},
            },
            "serving": {
                "model_path": str(tmp_path / "missing_autoencoder.pt"),
                "transformer_path": str(tmp_path / "missing_transformer.pt"),
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

    monkeypatch.setenv("SERVING_REQUIRE_FT_MODEL", "1")

    with pytest.raises(FileNotFoundError, match="SERVING_REQUIRE_FT_MODEL"):
        build_scorer(cfg)


@pytest.mark.parametrize(
    ("legacy_path", "versioned_path"),
    [
        ("/health", "/api/v1/health"),
        ("/metrics", "/api/v1/metrics"),
    ],
)
def test_versioned_get_routes_match_legacy_routes(
    client: TestClient,
    legacy_path: str,
    versioned_path: str,
) -> None:
    """Versioned GET endpoints preserve the legacy API contract."""
    legacy = client.get(legacy_path)
    versioned = client.get(versioned_path)

    assert versioned.status_code == legacy.status_code

    legacy_body = legacy.json()
    versioned_body = versioned.json()

    # uptime_s is computed live from time.perf_counter() on every call, so
    # the legacy and versioned calls (made microseconds apart) will not
    # match exactly -- everything else in the payload must still match.
    legacy_body.pop("uptime_s", None)
    versioned_body.pop("uptime_s", None)

    assert versioned_body == legacy_body


def test_versioned_predict_matches_legacy_predict(
    client: TestClient,
    features: list[float],
) -> None:
    """Versioned prediction endpoint exposes the same inference contract."""
    payload = {
        "transaction_id": "version-test",
        "features": features,
    }

    legacy = client.post("/predict", json=payload)
    versioned = client.post("/api/v1/predict", json=payload)

    assert legacy.status_code == 200
    assert versioned.status_code == 200

    legacy_body = legacy.json()
    versioned_body = versioned.json()

    for field in (
        "transaction_id",
        "is_fraud",
        "fraud_probability",
        "anomaly_score",
        "decision",
        "gate_used",
        "ft_model_used",
        "conformal_used",
        "conformal_set",
        "conformal_alpha",
        "conformal_threshold",
    ):
        if isinstance(legacy_body[field], float):
            assert versioned_body[field] == pytest.approx(legacy_body[field])
        else:
            assert versioned_body[field] == legacy_body[field]


def test_metrics_p99_matches_numpy(client: TestClient) -> None:
    """Metrics P90 and P99 use numpy.percentile calculation."""
    client.app.state.latencies = [1.0, 2.0, 3.0, 10.0, 100.0]
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    expected_p90 = float(np.percentile([1.0, 2.0, 3.0, 10.0, 100.0], 90))
    expected_p99 = float(np.percentile([1.0, 2.0, 3.0, 10.0, 100.0], 99))
    assert data["p90_latency_ms"] == pytest.approx(expected_p90)
    assert data["p99_latency_ms"] == pytest.approx(expected_p99)


def test_build_scorer_uses_calibrated_const(
    config: dict, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """build_scorer extracts calibrated_const from model when available."""
    from src.serving.api import build_reference_model, build_scorer

    model = build_reference_model(input_dim=int(config.autoencoder.input_dim))
    setattr(model, "calibrated_const", 33.3)
    ckpt = tmp_path / "dummy_ae.pt"
    ckpt.touch()
    monkeypatch.setattr("src.serving.api.load_checkpoint", lambda _: model)
    monkeypatch.setattr(config.serving, "model_path", str(ckpt))
    scorer = build_scorer(config)
    assert scorer.anomaly_const == 33.3


def test_predict_rejects_extra_fields(client: TestClient, features: list[float]) -> None:
    """Extra or misspelled payload fields are rejected with 422."""
    payload = {"features": features, "ft_probabilty": 0.5}  # deliberate typo
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


@pytest.fixture()
def ft_client(tmp_path) -> TestClient:
    """Test client with a server-side FT-CAT checkpoint on disk."""
    from src.models.ft_transformer import TabularMLP
    from src.training.trainer_utils import save_checkpoint as save_ft_checkpoint

    seed_everything(42)
    ft_model = TabularMLP(
        n_continuous=3,
        categorical_cardinalities=[2, 2],
        seq_len=2,
        seq_dim=2,
    )
    ft_model.eval()
    ft_path = tmp_path / "ft_transformer.pt"
    save_ft_checkpoint(ft_model, ft_path)

    cfg = Config(
        {
            "autoencoder": {
                "input_dim": 20,
                "anomaly_score": {"l1_gamma": 0.4},
            },
            "serving": {
                "model_path": str(tmp_path / "missing.pt"),
                "anomaly_const": 2.0,
                "transformer_path": str(ft_path),
            },
            "scoring": {
                "approve_threshold": 0.15,
                "block_threshold": 0.85,
                "escalate_threshold": 0.50,
            },
        },
        base_dir=tmp_path,
    )
    return TestClient(create_app(cfg))


def test_predict_with_automatic_ft_cat_inputs_uses_server_side_model(
    ft_client: TestClient, features: list[float]
) -> None:
    """Providing ft_continuous/ft_categorical/ft_sequence runs server-side FT-CAT."""
    response = ft_client.post(
        "/predict",
        json={
            "features": features,
            "ft_continuous": [0.1, -0.2, 0.3],
            "ft_categorical": [0, 1],
            "ft_sequence": [[0.1, 0.2], [0.3, 0.4]],
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["ft_model_used"] is True
    assert 0.0 <= body["ft_probability"] <= 1.0
    assert body["gate_used"] is False


def test_predict_with_automatic_ft_cat_inputs_rejects_manual_ft_probability(
    ft_client: TestClient, features: list[float]
) -> None:
    """ft_probability must not be supplied alongside server-side FT-CAT inputs."""
    response = ft_client.post(
        "/predict",
        json={
            "features": features,
            "ft_probability": 0.5,
            "ft_continuous": [0.1, -0.2, 0.3],
            "ft_categorical": [0, 1],
            "ft_sequence": [[0.1, 0.2], [0.3, 0.4]],
        },
    )
    assert response.status_code == 422


def test_predict_with_automatic_ft_cat_inputs_rejects_partial_fields(
    ft_client: TestClient, features: list[float]
) -> None:
    """ft_continuous/ft_categorical/ft_sequence must be provided together."""
    response = ft_client.post(
        "/predict",
        json={
            "features": features,
            "ft_continuous": [0.1, -0.2, 0.3],
            "ft_categorical": [0, 1],
        },
    )
    assert response.status_code == 422


def test_stream_with_automatic_ft_cat_inputs_scores_batch(
    ft_client: TestClient, features: list[float]
) -> None:
    """/stream runs server-side FT-CAT for every transaction when supplied."""
    transaction = {
        "features": features,
        "ft_continuous": [0.1, -0.2, 0.3],
        "ft_categorical": [0, 1],
        "ft_sequence": [[0.1, 0.2], [0.3, 0.4]],
    }
    response = ft_client.post(
        "/stream",
        json={"transactions": [transaction, transaction]},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 2
    for result in body["results"]:
        assert result["ft_model_used"] is True
        assert 0.0 <= result["ft_probability"] <= 1.0


def test_predict_clamps_out_of_vocabulary_categorical_code_to_unk(
    ft_client: TestClient, features: list[float]
) -> None:
    """An out-of-range categorical code is clamped to <UNK> (0), mirroring
    the training-time contract in train_transformer._to_code_matrix() /
    FraudPreprocessor, instead of being rejected outright. Held-out data can
    legitimately contain category codes that were never seen at fit time
    (e.g. a browser/device string only present after the training cutoff).
    """
    base_payload = {
        "features": features,
        "ft_continuous": [0.1, -0.2, 0.3],
        "ft_sequence": [[0.1, 0.2], [0.3, 0.4]],
    }

    out_of_vocab_response = ft_client.post(
        "/predict",
        json={**base_payload, "ft_categorical": [5, 1]},
    )
    clamped_response = ft_client.post(
        "/predict",
        json={**base_payload, "ft_categorical": [0, 1]},
    )

    assert out_of_vocab_response.status_code == 200
    assert clamped_response.status_code == 200

    out_of_vocab_body = out_of_vocab_response.json()
    clamped_body = clamped_response.json()

    assert out_of_vocab_body["ft_model_used"] is True
    assert out_of_vocab_body["ft_probability"] == pytest.approx(clamped_body["ft_probability"])
