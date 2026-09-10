"""FastAPI microservice for real-time fraud scoring.

Exposes ``/predict`` (single transaction, sub-15ms P99 target), ``/stream``
(batch scoring), ``/health`` and ``/metrics``. Models are loaded lazily at
startup: a real autoencoder checkpoint when present, otherwise a reference
model so the stack is exercisable before training artifacts exist. When a
learned hybrid gate checkpoint is present, the DAE residual is fused with the
caller-supplied FT-Transformer posterior (``ft_probability``).
"""

from __future__ import annotations

import json
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
from src.uncertainty.conformal_predictor import prediction_set, triage_decision
from src.training.train_autoencoder import _load_legit_features, load_checkpoint
from src.training.train_hybrid_gating import load_checkpoint as load_gate_checkpoint
from src.training.gate_velocity import extract_velocity_features
from src.training.train_transformer import load_ft_transformer
from src.training.feature_selection import UNK_INDEX
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


def _load_ft_model(
    config: Config,
) -> tuple[Any | None, dict[str, Any] | None]:
    """Load the trained FT-CAT model and checkpoint payload when available."""
    transformer_path = config.nested_get("serving.transformer_path", None)

    if not transformer_path:
        logger.warning("No serving transformer path configured; server-side FT-CAT disabled")
        return None, None

    model_path = Path(str(transformer_path))
    require_ft_model = os.environ.get("SERVING_REQUIRE_FT_MODEL", "0").lower() in (
        "1",
        "true",
        "yes",
    )

    if not model_path.is_file():
        if require_ft_model:
            raise FileNotFoundError(
                f"SERVING_REQUIRE_FT_MODEL is enabled but checkpoint not found at {model_path}"
            )
        logger.warning(
            "FT-CAT checkpoint %s missing; server-side FT-CAT disabled",
            model_path,
        )
        return None, None

    model, payload = load_ft_transformer(model_path, device="cpu")
    model.eval()

    logger.info("Loaded FT-CAT checkpoint from %s", model_path)

    return model, payload


def _load_conformal_threshold(
    config: Config,
) -> tuple[float | None, float | None]:
    """Load the calibrated conformal operating point used by serving."""
    artifact_path = Path(
        config.nested_get(
            "serving.conformal_path",
            "results/conformal/serving_threshold.json",
        )
    )

    if not artifact_path.is_file():
        logger.warning(
            "Conformal serving artifact %s missing; using legacy threshold triage",
            artifact_path,
        )
        return None, None

    try:
        with artifact_path.open("r", encoding="utf-8") as artifact_file:
            artifact = json.load(artifact_file)

        method = artifact.get("method")
        alpha = float(artifact["alpha"])
        threshold = float(artifact["threshold"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        logger.warning(
            "Unable to load conformal serving artifact %s; " "using legacy threshold triage: %s",
            artifact_path,
            exc,
        )
        return None, None

    if method != "split_conformal":
        logger.warning(
            "Unsupported conformal method %r in %s; using legacy threshold triage",
            method,
            artifact_path,
        )
        return None, None

    if not 0.0 < alpha < 1.0:
        logger.warning(
            "Invalid conformal alpha %.6f in %s; using legacy threshold triage",
            alpha,
            artifact_path,
        )
        return None, None

    if not 0.0 <= threshold <= 1.0:
        logger.warning(
            "Invalid conformal threshold %.6f in %s; using legacy threshold triage",
            threshold,
            artifact_path,
        )
        return None, None

    configured_alpha_value = config.nested_get("evaluation.conformal.alpha")

    if configured_alpha_value is None:
        logger.warning("Conformal alpha is not configured; using legacy threshold triage")
        return None, None

    try:
        configured_alpha = float(configured_alpha_value)
    except (TypeError, ValueError):
        logger.warning(
            "Configured conformal alpha %r is invalid; using legacy threshold triage",
            configured_alpha_value,
        )
        return None, None

    if not math.isclose(alpha, configured_alpha, rel_tol=0.0, abs_tol=1e-12):
        logger.warning(
            "Conformal artifact alpha %.6f does not match configured alpha %.6f; "
            "using legacy threshold triage",
            alpha,
            configured_alpha,
        )
        return None, None

    logger.info(
        "Loaded split-conformal serving threshold %.6f at alpha %.4f from %s",
        threshold,
        alpha,
        artifact_path,
    )

    return alpha, threshold


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
    ft_model, ft_payload = _load_ft_model(config)
    conformal_alpha, conformal_threshold = _load_conformal_threshold(config)
    calibrated_const = getattr(model, "calibrated_const", None)
    if calibrated_const is not None:
        anomaly_const = float(calibrated_const)
        logger.info("Using calibrated anomaly_const %.2f from checkpoint", anomaly_const)
    else:
        anomaly_const = float(config.serving.anomaly_const)

    shap_background = None

    try:
        train_path = config.get_path("data.train_data_path")
        non_feature_cols = list(config.data.non_feature_cols)

        legit_features = _load_legit_features(
            train_path,
            non_feature_cols,
        )

        background_size = min(
            int(config.explainability.shap_background_size),
            len(legit_features),
        )

        shap_background = torch.as_tensor(
            legit_features[:background_size],
            dtype=torch.float32,
        )

        logger.info(
            "Loaded %d legitimate transactions for SHAP background",
            background_size,
        )
    except (FileNotFoundError, ValueError, AttributeError, KeyError) as exc:
        logger.warning(
            "Unable to load SHAP background; explanations will use fallback residuals: %s",
            exc,
        )

    return Scorer(
        model=model,
        input_dim=int(config.autoencoder.input_dim),
        anomaly_const=anomaly_const,
        approve_threshold=float(config.scoring.approve_threshold),
        block_threshold=float(config.scoring.block_threshold),
        escalate_threshold=float(config.scoring.escalate_threshold),
        model_loaded=loaded_serialized,
        l1_gamma=float(config.autoencoder.anomaly_score.l1_gamma),
        gate=gate,
        normalizer=normalizer,
        shap_background=shap_background,
        conformal_alpha=conformal_alpha,
        conformal_threshold=conformal_threshold,
        ft_model=ft_model,
        ft_payload=ft_payload,
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
        shap_background: torch.Tensor | None = None,
        conformal_alpha: float | None = None,
        conformal_threshold: float | None = None,
        ft_model: Any | None = None,
        ft_payload: dict[str, Any] | None = None,
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
        self.shap_background = shap_background
        self.conformal_alpha = conformal_alpha
        self.conformal_threshold = conformal_threshold
        self.ft_model = ft_model
        self.ft_payload = ft_payload

        if gate is not None:
            gate.eval()

        if self.ft_model is not None:
            self.ft_model.eval()

    def _fuse(
        self,
        anomaly_scores: np.ndarray,
        ft_probabilities: list[float] | None,
        velocity_features: np.ndarray | None = None,
    ) -> tuple[np.ndarray, bool]:
        """Fuse DAE residuals with supervised and historical gate context.

        Args:
            anomaly_scores: Per-sample raw DAE residual scores.
            ft_probabilities: Per-sample FT-Transformer posteriors, or None.
            velocity_features: Optional array of shape ``(n_samples, 2)``
                containing history density and amount intensity.

        Returns:
            ``(probabilities, gate_used)`` tuple.

        Raises:
            ValueError: If required learned-gate inputs are missing or malformed.
        """
        if self.gate is None or self.normalizer is None:
            return _score_to_probability(anomaly_scores, self.anomaly_const), False

        if ft_probabilities is None:
            raise ValueError("ft_probability is required when the hybrid gate is loaded")

        scores_t = torch.as_tensor(np.asarray(anomaly_scores, dtype=np.float32))
        normalized = self.normalizer.transform(scores_t).reshape(-1)

        probs_t = torch.as_tensor(np.asarray(ft_probabilities, dtype=np.float32)).reshape(-1)

        if probs_t.numel() != normalized.numel():
            raise ValueError("ft_probabilities must match the anomaly-score batch size")

        input_dim = int(self.gate.input_dim)

        if input_dim == 2:
            gate_in = torch.stack(
                [normalized, probs_t],
                dim=1,
            )
        elif input_dim == 4:
            if velocity_features is None:
                raise ValueError(
                    "history_density and history_amount_intensity are required "
                    "when the 4-input hybrid gate is loaded"
                )

            velocity_array = np.asarray(
                velocity_features,
                dtype=np.float32,
            )

            if velocity_array.ndim != 2 or velocity_array.shape[1] != 2:
                raise ValueError("velocity_features must have shape (n_samples, 2)")

            if velocity_array.shape[0] != normalized.numel():
                raise ValueError("velocity_features must match the anomaly-score batch size")

            if not np.isfinite(velocity_array).all():
                raise ValueError("velocity_features must contain only finite values")

            velocity_t = torch.as_tensor(velocity_array)

            gate_in = torch.cat(
                (
                    torch.stack([normalized, probs_t], dim=1),
                    velocity_t,
                ),
                dim=1,
            )
        else:
            raise ValueError(f"Unsupported learned gate input dimension: {input_dim}")

        with torch.no_grad():
            fused = self.gate(gate_in).cpu().numpy()

        return fused, True

    def _predict_ft(
        self,
        ft_continuous: list[list[float]],
        ft_categorical: list[list[int]],
        ft_sequence: list[list[list[float]]],
    ) -> np.ndarray:
        """Run server-side FT-CAT inference for one or more transactions."""
        if self.ft_model is None or self.ft_payload is None:
            raise ValueError(
                "Server-side FT-CAT inference requested but the FT-CAT model is not loaded"
            )

        meta = self.ft_payload.get("meta")

        if not isinstance(meta, dict):
            raise ValueError("FT-CAT checkpoint is missing model metadata")

        n_continuous = int(meta["n_continuous"])
        categorical_cardinalities = [int(value) for value in meta["categorical_cardinalities"]]
        n_categorical = len(categorical_cardinalities)
        seq_len = int(meta["seq_len"])
        seq_dim = int(meta["seq_dim"])

        continuous = np.asarray(ft_continuous, dtype=np.float32)
        categorical = np.asarray(ft_categorical, dtype=np.int64)
        sequence = np.asarray(ft_sequence, dtype=np.float32)

        if continuous.ndim != 2 or continuous.shape[1] != n_continuous:
            raise ValueError(
                "ft_continuous must have shape "
                f"(batch, {n_continuous}), got {tuple(continuous.shape)}"
            )

        if categorical.ndim != 2 or categorical.shape[1] != n_categorical:
            raise ValueError(
                "ft_categorical must have shape "
                f"(batch, {n_categorical}), got {tuple(categorical.shape)}"
            )

        expected_sequence_shape = (
            continuous.shape[0],
            seq_len,
            seq_dim,
        )

        if sequence.shape != expected_sequence_shape:
            raise ValueError(
                "ft_sequence must have shape "
                f"{expected_sequence_shape}, got {tuple(sequence.shape)}"
            )

        if categorical.shape[0] != continuous.shape[0]:
            raise ValueError("FT-CAT categorical inputs must match the continuous-input batch size")

        if not np.isfinite(continuous).all():
            raise ValueError("ft_continuous must contain only finite values")

        if not np.isfinite(sequence).all():
            raise ValueError("ft_sequence must contain only finite values")

        for index, cardinality in enumerate(categorical_cardinalities):
            codes = categorical[:, index]
            out_of_range = (codes < 0) | (codes >= cardinality)

            if np.any(out_of_range):
                logger.warning(
                    "ft_categorical feature %d holds %d codes outside [0, %d); clamping to <UNK>",
                    index,
                    int(np.sum(out_of_range)),
                    cardinality,
                )
                codes[out_of_range] = UNK_INDEX
                categorical[:, index] = codes

        x_cont = torch.as_tensor(continuous, dtype=torch.float32)
        x_cat = torch.as_tensor(categorical, dtype=torch.long)
        seq = torch.as_tensor(sequence, dtype=torch.float32)

        with torch.no_grad():
            logits = self.ft_model(x_cont, x_cat, seq)
            probabilities = torch.sigmoid(logits).cpu().numpy()

        return np.asarray(probabilities, dtype=np.float32).reshape(-1)

    def _resolve_ft_single(
        self,
        ft_probability: float | None,
        history_density: float | None,
        history_amount_intensity: float | None,
        ft_continuous: list[float] | None,
        ft_categorical: list[int] | None,
        ft_sequence: list[list[float]] | None,
    ) -> tuple[float | None, np.ndarray | None, bool]:
        """Resolve legacy caller-supplied or server-side FT-CAT inputs."""
        automatic_values = (
            ft_continuous,
            ft_categorical,
            ft_sequence,
        )
        automatic_present = [value is not None for value in automatic_values]

        if any(automatic_present) and not all(automatic_present):
            raise ValueError(
                "ft_continuous, ft_categorical, and ft_sequence " "must be provided together"
            )

        if all(automatic_present):
            if ft_probability is not None:
                raise ValueError(
                    "ft_probability must not be provided when server-side "
                    "FT-CAT inputs are supplied"
                )

            if history_density is not None or history_amount_intensity is not None:
                raise ValueError(
                    "Manual history features must not be provided when " "ft_sequence is supplied"
                )

            assert ft_continuous is not None
            assert ft_categorical is not None
            assert ft_sequence is not None

            probability = float(
                self._predict_ft(
                    [ft_continuous],
                    [ft_categorical],
                    [ft_sequence],
                )[0]
            )

            velocity = (
                extract_velocity_features(
                    [ft_sequence],
                    transaction_amount_index=0,
                )
                .cpu()
                .numpy()
            )

            return probability, velocity, True

        velocity_features = None

        if history_density is not None or history_amount_intensity is not None:
            if history_density is None or history_amount_intensity is None:
                raise ValueError(
                    "history_density and history_amount_intensity " "must be provided together"
                )

            velocity_features = np.asarray(
                [[history_density, history_amount_intensity]],
                dtype=np.float32,
            )

        return ft_probability, velocity_features, False

    def _triage(self, probability: float) -> dict[str, Any]:
        """Apply conformal triage when calibrated, otherwise use legacy thresholds."""
        if self.conformal_threshold is not None and self.conformal_alpha is not None:
            conformal_prediction = prediction_set(
                fraud_probability=probability,
                threshold=self.conformal_threshold,
            )
            decision = triage_decision(conformal_prediction)

            return {
                "decision": decision,
                "is_fraud": probability >= self.escalate_threshold,
                "conformal_used": True,
                "conformal_set": sorted(conformal_prediction),
                "conformal_alpha": self.conformal_alpha,
                "conformal_threshold": self.conformal_threshold,
            }

        decision, is_fraud = _decide(
            probability,
            self.approve_threshold,
            self.block_threshold,
            self.escalate_threshold,
        )

        return {
            "decision": decision,
            "is_fraud": is_fraud,
            "conformal_used": False,
            "conformal_set": None,
            "conformal_alpha": None,
            "conformal_threshold": None,
        }

    def score(
        self,
        features: list[float],
        ft_probability: float | None = None,
        history_density: float | None = None,
        history_amount_intensity: float | None = None,
        ft_continuous: list[float] | None = None,
        ft_categorical: list[int] | None = None,
        ft_sequence: list[list[float]] | None = None,
    ) -> dict[str, Any]:
        """Score a single transaction.

        Args:
            features: Feature vector of the transaction.
            ft_probability: Optional FT-Transformer posterior in ``[0, 1]``.
            history_density: Optional historical activity density for the
                4-input learned gate.
            history_amount_intensity: Optional historical amount intensity
                for the 4-input learned gate.

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

        resolved_ft_probability, velocity_features, ft_model_used = self._resolve_ft_single(
            ft_probability=ft_probability,
            history_density=history_density,
            history_amount_intensity=history_amount_intensity,
            ft_continuous=ft_continuous,
            ft_categorical=ft_categorical,
            ft_sequence=ft_sequence,
        )

        probabilities, gate_used = self._fuse(
            np.asarray([score]),
            ([resolved_ft_probability] if resolved_ft_probability is not None else None),
            velocity_features,
        )
        probability = float(probabilities[0])
        triage = self._triage(probability)

        return {
            "fraud_probability": probability,
            "anomaly_score": score,
            "gate_used": gate_used,
            "ft_probability": resolved_ft_probability,
            "ft_model_used": ft_model_used,
            **triage,
        }

    def score_batch(
        self,
        features_batch: list[list[float]],
        ft_probabilities: list[float] | None = None,
        velocity_features: np.ndarray | None = None,
    ) -> list[dict[str, Any]]:
        """Score multiple transactions in one forward pass.

        Args:
            features_batch: Batch of feature vectors.
            ft_probabilities: Optional batch of FT-Transformer posteriors.
            velocity_features: Optional array of shape ``(n_samples, 2)``
                containing history density and amount intensity.

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
        probabilities, gate_used = self._fuse(
            scores,
            ft_probabilities,
            velocity_features,
        )
        results: list[dict[str, Any]] = []
        for score, probability in zip(scores, probabilities):
            triage = self._triage(float(probability))

            results.append(
                {
                    "fraud_probability": float(probability),
                    "anomaly_score": float(score),
                    "gate_used": gate_used,
                    **triage,
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
        except ImportError:
            pass
        except AttributeError:
            pass

        if has_shap:
            try:
                drivers = explain_transaction(
                    features,
                    model=self.model,
                    background=self.shap_background,
                    top_k=top_k,
                    ft_probability=ft_probability,
                    l1_gamma=self.l1_gamma,
                )
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

    @app.post("/api/v1/predict", response_model=PredictionResponse)
    @app.post("/predict", response_model=PredictionResponse)
    def predict(request: PredictionRequest) -> PredictionResponse:
        """Score a single transaction end-to-end."""
        start = time.perf_counter()
        try:
            result = app.state.scorer.score(
                request.features,
                request.ft_probability,
                request.history_density,
                request.history_amount_intensity,
                request.ft_continuous,
                request.ft_categorical,
                request.ft_sequence,
            )
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
            conformal_used=result["conformal_used"],
            conformal_set=result["conformal_set"],
            conformal_alpha=result["conformal_alpha"],
            conformal_threshold=result["conformal_threshold"],
            ft_probability=result["ft_probability"],
            ft_model_used=result["ft_model_used"],
        )

    @app.post("/api/v1/stream", response_model=StreamResponse)
    @app.post("/stream", response_model=StreamResponse)
    def stream(request: StreamRequest) -> StreamResponse:
        """Score a batch of transactions in a single forward pass."""
        start = time.perf_counter()
        try:
            ft_probs = [t.ft_probability for t in request.transactions]
            densities = [t.history_density for t in request.transactions]
            intensities = [t.history_amount_intensity for t in request.transactions]

            ft_continuous = [item.ft_continuous for item in request.transactions]
            ft_categorical = [item.ft_categorical for item in request.transactions]
            ft_sequences = [item.ft_sequence for item in request.transactions]

            automatic_presence = [
                (
                    item.ft_continuous is not None,
                    item.ft_categorical is not None,
                    item.ft_sequence is not None,
                )
                for item in request.transactions
            ]

            for index, presence in enumerate(automatic_presence):
                if any(presence) and not all(presence):
                    raise ValueError(
                        "ft_continuous, ft_categorical, and ft_sequence "
                        f"must be provided together for transaction {index}"
                    )

            automatic_rows = [all(presence) for presence in automatic_presence]

            if any(automatic_rows) and not all(automatic_rows):
                raise ValueError(
                    "Server-side FT-CAT inputs must be provided for every transaction "
                    "or for none of them"
                )

            if all(automatic_rows):
                if any(value is not None for value in ft_probs):
                    raise ValueError(
                        "ft_probability must not be provided when server-side "
                        "FT-CAT inputs are supplied"
                    )

                if any(value is not None for value in densities) or any(
                    value is not None for value in intensities
                ):
                    raise ValueError(
                        "Manual history features must not be provided when "
                        "ft_sequence is supplied"
                    )

            resolved_ft_probs = ft_probs
            resolved_densities = densities
            resolved_intensities = intensities
            ft_model_used_flags = [False] * len(request.transactions)

            if all(automatic_rows):
                assert all(value is not None for value in ft_continuous)
                assert all(value is not None for value in ft_categorical)
                assert all(value is not None for value in ft_sequences)

                resolved_ft_probs = (
                    app.state.scorer._predict_ft(
                        [value for value in ft_continuous if value is not None],
                        [value for value in ft_categorical if value is not None],
                        [value for value in ft_sequences if value is not None],
                    )
                    .astype(float)
                    .tolist()
                )

                velocity = (
                    extract_velocity_features(
                        [value for value in ft_sequences if value is not None],
                        transaction_amount_index=0,
                    )
                    .cpu()
                    .numpy()
                )

                resolved_densities = velocity[:, 0].astype(float).tolist()
                resolved_intensities = velocity[:, 1].astype(float).tolist()
                ft_model_used_flags = [True] * len(request.transactions)

            if all(p is None for p in resolved_ft_probs):
                ft_probs_arg: list[float] | None = None
            elif any(p is None for p in resolved_ft_probs):
                raise ValueError("ft_probability must be set for all or none of the batch")
            else:
                ft_probs_arg = [float(p) for p in resolved_ft_probs]

            velocity_missing = [
                density is None or intensity is None
                for density, intensity in zip(
                    resolved_densities,
                    resolved_intensities,
                )
            ]

            if all(velocity_missing):
                velocity_arg = None
            elif any(velocity_missing):
                raise ValueError(
                    "history_density and history_amount_intensity "
                    "must be set for all or none of the batch"
                )
            else:
                velocity_arg = np.asarray(
                    [
                        [float(density), float(intensity)]
                        for density, intensity in zip(
                            resolved_densities,
                            resolved_intensities,
                        )
                    ],
                    dtype=np.float32,
                )

            results_raw = app.state.scorer.score_batch(
                [t.features for t in request.transactions],
                ft_probs_arg,
                velocity_arg,
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
                ft_probability=(
                    resolved_ft_probs[index] if resolved_ft_probs is not None else None
                ),
                ft_model_used=ft_model_used_flags[index],
            )
            for index, (payload, raw) in enumerate(zip(request.transactions, results_raw))
        ]
        return StreamResponse(results=results, count=len(results), total_latency_ms=latency_ms)

    @app.post("/api/v1/explain", response_model=ExplainResponse)
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

    @app.get("/api/v1/health", response_model=HealthResponse)
    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        """Liveness and readiness summary."""
        return HealthResponse(
            status="ok" if app.state.scorer is not None else "degraded",
            version=VERSION,
            model_loaded=app.state.scorer.model_loaded,
            uptime_s=time.perf_counter() - app.state.start_time,
            gate_loaded=app.state.scorer.gate is not None,
            conformal_loaded=(
                app.state.scorer.conformal_alpha is not None
                and app.state.scorer.conformal_threshold is not None
            ),
            ft_model_loaded=app.state.scorer.ft_model is not None,
        )

    @app.get("/api/v1/metrics", response_model=MetricsResponse)
    @app.get("/metrics", response_model=MetricsResponse)
    async def metrics() -> MetricsResponse:
        """Self-monitoring counters: volume, errors, latency percentiles."""
        latencies = list(app.state.latencies)
        avg = float(np.mean(latencies)) if latencies else 0.0
        p90 = float(np.percentile(latencies, 90)) if latencies else 0.0
        p99 = float(np.percentile(latencies, 99)) if latencies else 0.0
        return MetricsResponse(
            requests_total=app.state.requests_total,
            errors_total=app.state.errors_total,
            avg_latency_ms=avg,
            p90_latency_ms=p90,
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
