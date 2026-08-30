"""FastAPI microservice for real-time fraud scoring.

Exposes ``/predict`` (single transaction, sub-15ms P99 target), ``/stream``
(batch scoring), ``/health`` and ``/metrics``. Models are loaded lazily at
startup: a real autoencoder checkpoint when present, otherwise a reference
model so the stack is exercisable before training artifacts exist.
"""

from __future__ import annotations

import time
from collections import deque
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from src.serving.model_serializer import build_reference_model
from src.serving.schemas import (
    HealthResponse,
    MetricsResponse,
    PredictionRequest,
    PredictionResponse,
    StreamRequest,
    StreamResponse,
)
from src.training.train_autoencoder import load_checkpoint
from src.utils.config import Config, load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

VERSION = "0.1.0"
MAX_LATENCY_SAMPLES = 10000
_ScoreFn = Callable[[np.ndarray], np.ndarray]


def _score_to_probability(score: np.ndarray, const: float) -> np.ndarray:
    """Map residual scores monotonically into the probability interval.

    Uses the saturating transform ``p = s / (s + const)`` to clamp scores
    to ``[0, 1)`` without needing a calibration dataset.

    Args:
        score: Per-sample anomaly residual scores.
        const: Saturating constant from the central configuration.

    Returns:
        Probability-likeness in ``[0, 1)``.
    """
    return score / (score + const)


def _decide(
    probability: float,
    approve_threshold: float,
    block_threshold: float,
    escalate_threshold: float,
) -> tuple[str, bool]:
    """Map a probability to a triage decision and fraud flag.

    Args:
        probability: Calibrated fraud likelihood in ``[0, 1)``.
        approve_threshold: Max probability granting autonomous approval.
        block_threshold: Min probability granting autonomous blocking.
        escalate_threshold: Fraud flag cut-off inside the escalation band.

    Returns:
        ``(decision, is_fraud)`` tuple.
    """
    if probability >= block_threshold:
        return "auto_block", True
    if probability <= approve_threshold:
        return "auto_approve", False
    return "escalate", probability >= escalate_threshold


def build_scorer(config: Config) -> "Scorer":
    """Instantiate the scorer used by the application.

    Args:
        config: Central application configuration.

    Returns:
        A ready-to-serve :class:`Scorer` instance.
    """
    model_path = Path(config.serving.model_path)
    if model_path.is_file():
        model = load_checkpoint(model_path)
        logger.info("Loaded autoencoder checkpoint from %s", model_path)
        loaded_serialized = True
    else:
        model = build_reference_model(input_dim=int(config.autoencoder.input_dim))
        logger.warning("Checkpoint %s missing; using reference scoring model", model_path)
        loaded_serialized = False
    return Scorer(
        model=model,
        input_dim=int(config.autoencoder.input_dim),
        anomaly_const=float(config.serving.anomaly_const),
        approve_threshold=float(config.scoring.approve_threshold),
        block_threshold=float(config.scoring.block_threshold),
        escalate_threshold=float(config.scoring.escalate_threshold),
        model_loaded=loaded_serialized,
        l1_gamma=float(config.autoencoder.anomaly_score.l1_gamma),
    )


class Scorer:
    """Stateful scorer holding the model and all scoring thresholds.

    Args:
        model: PyTorch scoring module in evaluation mode.
        input_dim: Expected feature vector dimensionality.
        anomaly_const: Saturating constant for score normalization.
        approve_threshold: Probability below which transactions are
            autonomously approved.
        block_threshold: Probability above which transactions are
            autonomously blocked.
        escalate_threshold: Probability above which humans review
            borderline cases.
        model_loaded: Whether a serialized artifact (vs reference model)
            is in use.
        l1_gamma: L1 weighting of the DAE residual metric.
    """

    def __init__(
        self,
        model: Any,
        input_dim: int,
        anomaly_const: float,
        approve_threshold: float,
        block_threshold: float,
        escalate_threshold: float,
        model_loaded: bool,
        l1_gamma: float,
    ) -> None:
        """Initialize the scorer."""
        self.model = model
        self.model.eval()
        self.input_dim = input_dim
        self.anomaly_const = anomaly_const
        self.approve_threshold = approve_threshold
        self.block_threshold = block_threshold
        self.escalate_threshold = escalate_threshold
        self.model_loaded = model_loaded
        self.l1_gamma = l1_gamma

    def score(self, features: list[float]) -> dict[str, Any]:
        """Score a single transaction.

        Args:
            features: Feature vector of the transaction.

        Returns:
            Dict with ``is_fraud``, ``fraud_probability``, ``anomaly_score``
            and ``decision`` keys.

        Raises:
            ValueError: If feature count does not match the model input.
        """
        if len(features) != self.input_dim:
            raise ValueError(f"Expected {self.input_dim} features, got {len(features)}")
        x = torch.as_tensor(np.asarray([features], dtype=np.float32))
        with torch.no_grad():
            score = self.model.anomaly_score(x, l1_gamma=self.l1_gamma).item()
        probability = float(_score_to_probability(np.asarray([score]), self.anomaly_const)[0])
        decision, is_fraud = _decide(
            probability, self.approve_threshold, self.block_threshold, self.escalate_threshold
        )
        return {
            "is_fraud": is_fraud,
            "fraud_probability": probability,
            "anomaly_score": score,
            "decision": decision,
        }

    def score_batch(self, features_batch: list[list[float]]) -> list[dict[str, Any]]:
        """Score multiple transactions in one forward pass.

        Args:
            features_batch: Batch of feature vectors.

        Returns:
            Per-transaction score dicts in request order.
        """
        if any(len(f) != self.input_dim for f in features_batch):
            raise ValueError(f"All rows must have exactly {self.input_dim} features")
        x = torch.as_tensor(np.asarray(features_batch, dtype=np.float32))
        with torch.no_grad():
            scores = self.model.anomaly_score(x, l1_gamma=self.l1_gamma).cpu().numpy()
        probabilities = _score_to_probability(scores, self.anomaly_const)
        results: list[dict[str, Any]] = []
        for score, probability in zip(scores, probabilities):
            decision, is_fraud = _decide(
                float(probability),
                self.approve_threshold,
                self.block_threshold,
                self.escalate_threshold,
            )
            results.append(
                {
                    "is_fraud": is_fraud,
                    "fraud_probability": float(probability),
                    "anomaly_score": float(score),
                    "decision": decision,
                }
            )
        return results


def create_app(config: Config | None = None) -> FastAPI:
    """Application factory wiring routes, models and self-monitoring.

    Args:
        config: Project configuration; defaults to ``config/config.yaml``.

    Returns:
        The configured FastAPI instance.
    """
    cfg = config if config is not None else load_config()
    app = FastAPI(
        title="Fraud Detection Engine",
        version=VERSION,
        description="Real-time fraud scoring with semi-supervised DAE",
    )

    app.state.config = cfg
    app.state.scorer = build_scorer(cfg)
    app.state.start_time = time.perf_counter()
    app.state.latencies: deque[float] = deque(maxlen=MAX_LATENCY_SAMPLES)
    app.state.requests_total = 0
    app.state.errors_total = 0

    @app.post("/predict", response_model=PredictionResponse)
    def predict(request: PredictionRequest) -> PredictionResponse:
        """Score a single transaction end-to-end."""
        start = time.perf_counter()
        try:
            result = app.state.scorer.score(request.features)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        latency_ms = (time.perf_counter() - start) * 1000.0
        app.state.latencies.append(latency_ms)
        app.state.requests_total += 1
        return PredictionResponse(
            transaction_id=request.transaction_id,
            is_fraud=result["is_fraud"],
            fraud_probability=result["fraud_probability"],
            anomaly_score=result["anomaly_score"],
            decision=result["decision"],
            latency_ms=latency_ms,
        )

    @app.post("/stream", response_model=StreamResponse)
    def stream(request: StreamRequest) -> StreamResponse:
        """Score a batch of transactions in a single forward pass."""
        start = time.perf_counter()
        try:
            results_raw = app.state.scorer.score_batch([t.features for t in request.transactions])
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        latency_ms = (time.perf_counter() - start) * 1000.0
        app.state.latencies.append(latency_ms)
        app.state.requests_total += len(request.transactions)
        results = [
            PredictionResponse(
                transaction_id=payload.transaction_id,
                **raw,
                latency_ms=latency_ms,
            )
            for payload, raw in zip(request.transactions, results_raw)
        ]
        return StreamResponse(results=results, count=len(results), total_latency_ms=latency_ms)

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        """Liveness and readiness summary."""
        return HealthResponse(
            status="ok" if app.state.scorer is not None else "degraded",
            version=VERSION,
            model_loaded=app.state.scorer.model_loaded,
            uptime_s=time.perf_counter() - app.state.start_time,
        )

    @app.get("/metrics", response_model=MetricsResponse)
    def metrics() -> MetricsResponse:
        """Self-monitoring counters: volume, errors, latency percentiles."""
        latencies = sorted(app.state.latencies)
        avg = sum(latencies) / len(latencies) if latencies else 0.0
        p99 = latencies[min(len(latencies) - 1, int(0.99 * len(latencies)))] if latencies else 0.0
        return MetricsResponse(
            requests_total=app.state.requests_total,
            errors_total=app.state.errors_total,
            avg_latency_ms=avg,
            p99_latency_ms=p99,
        )

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
        """Convert semantic scoring errors into 400 responses."""
        app.state.errors_total += 1
        logger.warning("Bad request on %s: %s", request.url.path, exc)
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    return app


app = create_app()
