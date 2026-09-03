"""Pydantic request/response schemas for the serving microservice.

All API payloads are strictly validated: lengths, bounds, and enum values
are checked at the request boundary so downstream modules never see
malformed data.
"""

from __future__ import annotations

import math
from typing import Literal
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

Decision = Literal["auto_approve", "auto_block", "escalate"]

N_FEATURES_MIN = 8
N_FEATURES_MAX = 2500
MAX_BATCH_SIZE = 256


class PredictionRequest(BaseModel):
    """A single transaction feature vector to score.

    Attributes:
        features: Raw numeric feature vector of a transaction. Exact
            dimensionality is validated at the model level at runtime.
        transaction_id: Optional external transaction identifier.
        card_id: Optional masked cardholder identifier for triage context.
        ft_probability: Optional supervised FT-Transformer fraud posterior in
            ``[0, 1]``. When the learned hybrid gate is loaded, this is fused
            with the DAE residual; otherwise it is ignored.
    """

    model_config = ConfigDict(extra="ignore")

    features: list[float] = Field(..., min_length=N_FEATURES_MIN, max_length=N_FEATURES_MAX)
    transaction_id: Optional[str] = Field(default=None, max_length=64)
    card_id: Optional[str] = Field(default=None, max_length=64)
    ft_probability: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    @field_validator("features")
    @classmethod
    def validate_finite_features(cls, features: list[float]) -> list[float]:
        """Ensure all feature values are finite (no NaN or Inf).

        Args:
            features: List of float values to validate.

        Returns:
            The validated list of features.

        Raises:
            ValueError: If any element is NaN or infinite.
        """
        if any(not math.isfinite(x) for x in features):
            raise ValueError("All features must be finite numbers (no NaN or Inf)")
        return features


class StreamRequest(BaseModel):
    """A batch of transactions submitted through the streaming endpoint.

    Attributes:
        transactions: Payload list, capped at ``MAX_BATCH_SIZE`` entries.
    """

    model_config = ConfigDict(extra="ignore")

    transactions: list[PredictionRequest] = Field(..., min_length=1, max_length=MAX_BATCH_SIZE)


class PredictionResponse(BaseModel):
    """Scoring result for a single transaction.

    Attributes:
        transaction_id: Echoed transaction identifier (if provided).
        is_fraud: Discrete fraud prediction flag.
        fraud_probability: Fraud likelihood in ``[0, 1]``.
        anomaly_score: Raw DAE reconstruction residual score.
        decision: Triage policy decision.
        latency_ms: End-to-end request latency in milliseconds.
        gate_used: Whether the learned hybrid gate fused the DAE residual
            with the FT-Transformer posterior for this prediction.
    """

    model_config = ConfigDict(extra="ignore")

    transaction_id: Optional[str] = None
    is_fraud: bool
    fraud_probability: float = Field(..., ge=0.0, le=1.0)
    anomaly_score: float = Field(..., ge=0.0)
    decision: Decision
    latency_ms: float = Field(..., ge=0.0)
    gate_used: bool = False


class StreamResponse(BaseModel):
    """Scoring results for a batch of transactions.

    Attributes:
        results: Per-transaction predictions in request order.
        count: Number of scored transactions.
        total_latency_ms: Cumulative batch processing time.
    """

    model_config = ConfigDict(extra="ignore")

    results: list[PredictionResponse]
    count: int = Field(..., ge=1)
    total_latency_ms: float = Field(..., ge=0.0)


class HealthResponse(BaseModel):
    """Service health summary.

    Attributes:
        status: ``"ok"`` when the service can score transactions.
        version: Package version of the running service.
        model_loaded: Whether a real checkpoint (vs reference model) is up.
        uptime_s: Seconds since the process started.
        gate_loaded: Whether the learned hybrid gate checkpoint is up.
    """

    model_config = ConfigDict(extra="ignore")

    status: Literal["ok", "degraded"]
    version: str
    model_loaded: bool
    uptime_s: float = Field(..., ge=0.0)
    gate_loaded: bool = False


class MetricsResponse(BaseModel):
    """In-flight self-monitoring counters.

    Attributes:
        requests_total: Number of transactions served since startup.
        errors_total: Number of failed requests.
        avg_latency_ms: Mean latency across served transactions.
        p99_latency_ms: 99th percentile latency across served transactions.
    """

    model_config = ConfigDict(extra="ignore")

    requests_total: int = Field(..., ge=0)
    errors_total: int = Field(..., ge=0)
    avg_latency_ms: float = Field(..., ge=0.0)
    p99_latency_ms: float = Field(..., ge=0.0)


class RiskDriver(BaseModel):
    """Individual feature contribution to the fraud/anomaly assessment.

    Attributes:
        feature_name: Name or index of the contributing feature.
        attribution: Attribution score (positive increases risk, negative decreases).
        value: Original numeric feature value.
    """

    model_config = ConfigDict(extra="ignore")

    feature_name: str
    attribution: float
    value: Optional[float] = None


class ExplainRequest(BaseModel):
    """Payload for transaction feature attribution explanation.

    Attributes:
        features: Numeric feature vector matching the model input.
        transaction_id: Optional identifier for logging/tracking.
        top_k: Number of top contributing risk drivers to return (default 5).
        ft_probability: Optional FT-Transformer probability for fused attribution.
    """

    model_config = ConfigDict(extra="ignore")

    features: list[float] = Field(..., min_length=N_FEATURES_MIN, max_length=N_FEATURES_MAX)
    transaction_id: Optional[str] = Field(default=None, max_length=64)
    top_k: int = Field(default=5, ge=1, le=50)
    ft_probability: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    @field_validator("features")
    @classmethod
    def validate_finite_features(cls, features: list[float]) -> list[float]:
        """Ensure all feature values are finite (no NaN or Inf).

        Args:
            features: List of float values to validate.

        Returns:
            The validated list of features.

        Raises:
            ValueError: If any element is NaN or infinite.
        """
        if any(not math.isfinite(x) for x in features):
            raise ValueError("All features must be finite numbers (no NaN or Inf)")
        return features


class ExplainResponse(BaseModel):
    """Attribution breakdown explaining risk drivers for a transaction.

    Attributes:
        transaction_id: Echoed identifier (if provided).
        top_drivers: Ordered list of top risk-increasing features.
        method: Attribution method used (e.g. 'shap' or 'dae_reconstruction_residual').
        latency_ms: Processing duration in milliseconds.
    """

    model_config = ConfigDict(extra="ignore")

    transaction_id: Optional[str] = None
    top_drivers: list[RiskDriver]
    method: str
    latency_ms: float = Field(..., ge=0.0)
