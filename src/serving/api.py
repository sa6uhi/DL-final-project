"""FastAPI microservice for real-time fraud scoring.

Exposes ``/predict`` (single transaction, sub-15ms P99 target), ``/stream``
(batch scoring), ``/health`` and ``/metrics``. Models are loaded lazily at
startup: a real autoencoder checkpoint when present, otherwise a reference
model so the stack is exercisable before training artifacts exist. When a
learned hybrid gate checkpoint is present, the DAE residual is fused with the
caller-supplied FT-Transformer posterior (``ft_probability``).
"""

from __future__ import annotations

import math
import os
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
    ExplainRequest,
    ExplainResponse,
    HealthResponse,
    MetricsResponse,
    PredictionRequest,
    PredictionResponse,
    StreamRequest,
    StreamResponse,
)
from src.models.hybrid_gating import LearnedHybridGate, PercentileNormalizer
from src.training.train_autoencoder import load_checkpoint
from src.training.train_hybrid_gating import load_checkpoint as load_gate_checkpoint
from src.utils.config import Config, load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

VERSION = "0.1.0"
MAX_LATENCY_SAMPLES = 10000
_ScoreFn = Callable[[np.ndarray], np.ndarray]


def _score_to_probability(score: np.ndarray, const: float) -> np.ndarray:
    """Map residual scores monotonically into the pseudo-probability interval.

    Uses the soft-saturating transform ``p = s / (s + const)`` to project
    unbounded reconstruction residuals onto ``[0, 1)``. Note: This mapping
    is an uncalibrated monotonic proxy for standalone DAE scoring; formal
    coverage guarantees are governed by downstream conformal prediction.

    Args:
        score: Per-sample anomaly residual scores.
        const: Positive scaling constant governing the half-saturation point.

    Returns:
        Pseudo-probability array in ``[0, 1)``.
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


def _load_gate(config: Config) -> tuple[LearnedHybridGate | None, PercentileNormalizer | None]:
    """Load the learned hybrid gate when its checkpoint exists.

    Args:
        config: Central application configuration.

    Returns:
        ``(gate, normalizer)`` when the checkpoint configured under
        ``hybrid_gating.learned.checkpoint_path`` exists, else ``(None, None)``
        so scoring falls back to the DAE-only path.
    """
    gate_path = config.nested_get("hybrid_gating.learned.checkpoint_path", None)
    if not gate_path:
        return None, None
    if not Path(str(gate_path)).is_file():
        logger.warning("Gate checkpoint %s missing; using DAE-only scoring", gate_path)
        return None, None
    gate, normalizer = load_gate_checkpoint(gate_path)
    logger.info("Loaded hybrid gate checkpoint from %s", gate_path)
    return gate, normalizer


def build_scorer(config: Config) -> "Scorer":
    """Instantiate the scorer used by the application.

    Args:
        config: Central application configuration.

    Returns:
        A ready-to-serve :class:`Scorer` instance.
    """
    model_path = Path(config.serving.model_path)
    require_ckpt = os.environ.get("SERVING_REQUIRE_CHECKPOINT", "0").lower() in (
        "1",
        "true",
        "yes",
    )
    if model_path.is_file():
        model = load_checkpoint(model_path)
        logger.info("Loaded autoencoder checkpoint from %s", model_path)
        loaded_serialized = True
    elif require_ckpt:
        raise FileNotFoundError(
            f"SERVING_REQUIRE_CHECKPOINT is enabled but checkpoint not found at {model_path}"
        )
    else:
        model = build_reference_model(input_dim=int(config.autoencoder.input_dim))
        logger.warning("Checkpoint %s missing; using reference scoring model", model_path)
        loaded_serialized = False
    gate, normalizer = _load_gate(config)
    return Scorer(
        model=model,
        input_dim=int(config.autoencoder.input_dim),
        anomaly_const=float(config.serving.anomaly_const),
        approve_threshold=float(config.scoring.approve_threshold),
        block_threshold=float(config.scoring.block_threshold),
        escalate_threshold=float(config.scoring.escalate_threshold),
        model_loaded=loaded_serialized,
        l1_gamma=float(config.autoencoder.anomaly_score.l1_gamma),
        gate=gate,
        normalizer=normalizer,
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
        gate: Optional learned hybrid gate fusing the normalized DAE
            residual with the FT-Transformer posterior.
        normalizer: Fitted percentile normalizer paired with ``gate``.
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
        gate: LearnedHybridGate | None = None,
        normalizer: PercentileNormalizer | None = None,
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
        self.gate = gate
        self.normalizer = normalizer
        if gate is not None:
            gate.eval()

    def _fuse(
        self, anomaly_scores: np.ndarray, ft_probabilities: list[float] | None
    ) -> tuple[np.ndarray, bool]:
        """Fuse DAE residuals with FT posteriors when the gate is loaded.

        Args:
            anomaly_scores: Per-sample raw DAE residual scores.
            ft_probabilities: Per-sample FT-Transformer posteriors, or None.

        Returns:
            ``(probabilities, gate_used)`` tuple.

        Raises:
            ValueError: If the gate is loaded but posteriors are missing.
        """
        if self.gate is None or self.normalizer is None:
            return _score_to_probability(anomaly_scores, self.anomaly_const), False
        if ft_probabilities is None:
            raise ValueError("ft_probability is required when the hybrid gate is loaded")
        scores_t = torch.as_tensor(np.asarray(anomaly_scores, dtype=np.float32))
        normalized = self.normalizer.transform(scores_t)
        probs_t = torch.as_tensor(np.asarray(ft_probabilities, dtype=np.float32))
        gate_in = torch.stack([normalized.reshape(-1), probs_t.reshape(-1)], dim=1)
        with torch.no_grad():
            fused = self.gate(gate_in).cpu().numpy()
        return fused, True

    def score(self, features: list[float], ft_probability: float | None = None) -> dict[str, Any]:
        """Score a single transaction.

        Args:
            features: Feature vector of the transaction.
            ft_probability: Optional FT-Transformer posterior in ``[0, 1]``.

        Returns:
            Dict with ``is_fraud``, ``fraud_probability``, ``anomaly_score``,
            ``decision`` and ``gate_used`` keys.

        Raises:
            ValueError: If feature count does not match the model input.
        """
        if len(features) != self.input_dim:
            raise ValueError(f"Expected {self.input_dim} features, got {len(features)}")
        if any(not math.isfinite(x_val) for x_val in features):
            raise ValueError("All features must be finite numbers (no NaN or Inf)")
        x = torch.as_tensor(np.asarray([features], dtype=np.float32))
        with torch.no_grad():
            score = self.model.anomaly_score(x, l1_gamma=self.l1_gamma).item()
        probabilities, gate_used = self._fuse(
            np.asarray([score]), [ft_probability] if ft_probability is not None else None
        )
        probability = float(probabilities[0])
        decision, is_fraud = _decide(
            probability, self.approve_threshold, self.block_threshold, self.escalate_threshold
        )
        return {
            "is_fraud": is_fraud,
            "fraud_probability": probability,
            "anomaly_score": score,
            "decision": decision,
            "gate_used": gate_used,
        }

    def score_batch(
        self, features_batch: list[list[float]], ft_probabilities: list[float] | None = None
    ) -> list[dict[str, Any]]:
        """Score multiple transactions in one forward pass.

        Args:
            features_batch: Batch of feature vectors.
            ft_probabilities: Optional batch of FT-Transformer posteriors.

        Returns:
            Per-transaction score dicts in request order.
        """
        if not features_batch:
            raise ValueError("Batch cannot be empty")
        if any(len(f) != self.input_dim for f in features_batch):
            raise ValueError(f"All rows must have exactly {self.input_dim} features")
        for row in features_batch:
            if any(not math.isfinite(x_val) for x_val in row):
                raise ValueError("All features must be finite numbers (no NaN or Inf)")
        if ft_probabilities is not None and len(ft_probabilities) != len(features_batch):
            raise ValueError("ft_probabilities must match the batch size")
        x = torch.as_tensor(np.asarray(features_batch, dtype=np.float32))
        with torch.no_grad():
            scores = self.model.anomaly_score(x, l1_gamma=self.l1_gamma).cpu().numpy()
        probabilities, gate_used = self._fuse(scores, ft_probabilities)
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
                    "gate_used": gate_used,
                }
            )
        return results

    def explain(
        self, features: list[float], top_k: int = 5, ft_probability: float | None = None
    ) -> dict[str, Any]:
        """Explain feature risk drivers for a single transaction.

        If the fast-path SHAP explainer module (``src.explainability.shap_explainer``)
        is importable, delegates to it. Otherwise, computes fast-path
        reconstruction residual attributions from the DAE.

        Args:
            features: Feature vector of length ``input_dim``.
            top_k: Number of highest-risk features to return.
            ft_probability: Optional supervised fraud probability.

        Returns:
            Dict containing ``top_drivers`` and ``method``.

        Raises:
            ValueError: If feature vector length does not match ``input_dim``.
        """
        if len(features) != self.input_dim:
            raise ValueError(f"Expected {self.input_dim} features, got {len(features)}")
        has_shap = False
        try:
            from src.explainability.shap_explainer import explain_transaction  # type: ignore

            has_shap = True
        except (ImportError, AttributeError):
            pass

        if has_shap:
            try:
                drivers = explain_transaction(features, top_k=top_k, ft_probability=ft_probability)
                return {"top_drivers": drivers, "method": "shap"}
            except Exception as exc:
                logger.warning("SHAP explanation failed, falling back to DAE residuals: %s", exc)

        x = torch.as_tensor(np.asarray([features], dtype=np.float32))
        with torch.no_grad():
            if hasattr(self.model, "decode") and hasattr(self.model, "encode"):
                x_hat = self.model(x)
                residuals = (x - x_hat).abs().squeeze(0).cpu().numpy()
            else:
                residuals = np.abs(np.asarray(features))
        top_indices = np.argsort(residuals)[::-1][:top_k]
        drivers = [
            {
                "feature_name": f"feature_{idx}",
                "attribution": float(residuals[idx]),
                "value": float(features[idx]),
            }
            for idx in top_indices
        ]
        return {"top_drivers": drivers, "method": "dae_reconstruction_residual"}


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
    app.state.latencies = deque(maxlen=MAX_LATENCY_SAMPLES)
    app.state.requests_total = 0
    app.state.errors_total = 0

    @app.post("/predict", response_model=PredictionResponse)
    def predict(request: PredictionRequest) -> PredictionResponse:
        """Score a single transaction end-to-end."""
        start = time.perf_counter()
        try:
            result = app.state.scorer.score(request.features, request.ft_probability)
        except ValueError as exc:
            app.state.errors_total += 1
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
            gate_used=result["gate_used"],
        )

    @app.post("/stream", response_model=StreamResponse)
    def stream(request: StreamRequest) -> StreamResponse:
        """Score a batch of transactions in a single forward pass."""
        start = time.perf_counter()
        try:
            ft_probs = [t.ft_probability for t in request.transactions]
            if all(p is None for p in ft_probs):
                ft_probs_arg: list[float] | None = None
            elif any(p is None for p in ft_probs):
                raise ValueError("ft_probability must be set for all or none of the batch")
            else:
                ft_probs_arg = [float(p) for p in ft_probs]
            results_raw = app.state.scorer.score_batch(
                [t.features for t in request.transactions], ft_probs_arg
            )
        except ValueError as exc:
            app.state.errors_total += 1
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

    @app.post("/explain", response_model=ExplainResponse)
    def explain(request: ExplainRequest) -> ExplainResponse:
        """Explain the primary risk drivers for a transaction."""
        start = time.perf_counter()
        try:
            result = app.state.scorer.explain(
                request.features, top_k=request.top_k, ft_probability=request.ft_probability
            )
        except ValueError as exc:
            app.state.errors_total += 1
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        latency_ms = (time.perf_counter() - start) * 1000.0
        app.state.latencies.append(latency_ms)
        app.state.requests_total += 1
        return ExplainResponse(
            transaction_id=request.transaction_id,
            top_drivers=result["top_drivers"],
            method=result["method"],
            latency_ms=latency_ms,
        )

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        """Liveness and readiness summary."""
        return HealthResponse(
            status="ok" if app.state.scorer is not None else "degraded",
            version=VERSION,
            model_loaded=app.state.scorer.model_loaded,
            uptime_s=time.perf_counter() - app.state.start_time,
            gate_loaded=app.state.scorer.gate is not None,
        )

    @app.get("/metrics", response_model=MetricsResponse)
    async def metrics() -> MetricsResponse:
        """Self-monitoring counters: volume, errors, latency percentiles."""
        latencies = list(app.state.latencies)
        avg = float(np.mean(latencies)) if latencies else 0.0
        p99 = float(np.percentile(latencies, 99)) if latencies else 0.0
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
