"""Leakage-safe orchestration utilities for hybrid fraud gating and conformal triage."""

# Import necessary libraries and modules
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import torch
from torch import nn

from src.data.temporal_split import (
    split_gate_development,
    split_validation_for_gate_and_conformal,
)
from src.models.hybrid_gating import LearnedHybridGate, PercentileNormalizer
from src.training.gate_velocity import extract_velocity_features
from src.training.train_hybrid_gating import (
    GateData,
    build_gate_features,
    train_gate,
)
from src.uncertainty.conformal_predictor import SplitConformalPredictor
from src.utils.config import Config
from src.utils.logger import get_logger
from src.utils.tensor_validation import (
    validate_probability_tensor,
    validate_tensor,
)

logger = get_logger(__name__)


# Define dataclasses for hybrid data splits and pipeline artifacts
@dataclass(frozen=True)
class HybridDataSplit:
    """Chronological subsets used by the learned gate and conformal predictor."""

    gate_train: pd.DataFrame
    gate_val: pd.DataFrame
    conformal_calibration: pd.DataFrame


@dataclass(frozen=True)
class HybridPipelineArtifacts:
    """Trained components produced by the hybrid calibration pipeline."""

    gate: LearnedHybridGate
    normalizer: PercentileNormalizer
    conformal_predictor: SplitConformalPredictor


def create_hybrid_data_split(
    validation_df: pd.DataFrame,
    config: Config,
) -> HybridDataSplit:
    """Create leakage-safe chronological gate and conformal subsets.

    The original validation period is first divided into an earlier
    gate-development pool and a later conformal-calibration pool. The
    gate-development pool is then divided again into gate-training and
    gate-validation subsets for early stopping.

    The resulting temporal ordering is:

        gate train -> gate validation -> conformal calibration

    Args:
        validation_df: Original chronological validation period.
        config: Central project configuration.

    Returns:
        Chronological subsets for learned-gate training, gate validation,
        and conformal calibration.
    """
    time_col = str(config.split.time_col)
    gate_fraction = float(config.hybrid_gating.split.gate_fraction)
    gate_train_fraction = float(config.hybrid_gating.split.gate_train_fraction)

    gate_development_df, conformal_df = split_validation_for_gate_and_conformal(
        validation_df,
        time_col=time_col,
        gate_fraction=gate_fraction,
    )

    gate_train_df, gate_val_df = split_gate_development(
        gate_development_df,
        time_col=time_col,
        train_fraction=gate_train_fraction,
    )

    if gate_train_df[time_col].max() >= gate_val_df[time_col].min():
        raise ValueError("TEMPORAL LEAKAGE: gate training overlaps gate validation")

    if gate_val_df[time_col].max() >= conformal_df[time_col].min():
        raise ValueError("TEMPORAL LEAKAGE: gate validation overlaps conformal calibration")

    logger.info(
        "Hybrid temporal split - gate train: %d, gate validation: %d, " "conformal calibration: %d",
        len(gate_train_df),
        len(gate_val_df),
        len(conformal_df),
    )

    return HybridDataSplit(
        gate_train=gate_train_df,
        gate_val=gate_val_df,
        conformal_calibration=conformal_df,
    )


def transformer_probabilities(
    transformer: nn.Module,
    x_cont: torch.Tensor,
    x_cat: torch.Tensor,
    sequence: torch.Tensor,
) -> torch.Tensor:
    """Run FT-CAT inference and convert raw logits to fraud probabilities.

    Args:
        transformer: Trained FT-CAT compatible model.
        x_cont: Continuous transaction features.
        x_cat: Encoded categorical transaction features.
        sequence: Historical transaction windows.

    Returns:
        One fraud probability per transaction.

    Raises:
        ValueError: If model output is not a finite one-dimensional tensor.
        TypeError: If the model returns an unsupported output type.
    """
    was_training = transformer.training
    transformer.eval()

    try:
        with torch.no_grad():
            output = transformer(
                x_cont,
                x_cat,
                sequence,
            )
    finally:
        if was_training:
            transformer.train()

    if isinstance(output, tuple):
        logits = output[0]
    else:
        logits = output

    if not isinstance(logits, torch.Tensor):
        raise TypeError("Transformer output must contain a tensor of logits")

    validate_tensor(
        logits,
        name="Transformer logits",
        ndim=1,
        allow_empty=False,
        require_finite=True,
    )

    return torch.sigmoid(logits)


def autoencoder_anomaly_scores(
    autoencoder: nn.Module,
    features: torch.Tensor,
    l1_gamma: float,
) -> torch.Tensor:
    """Generate deterministic per-transaction DAE anomaly scores.

    Args:
        autoencoder: Trained DAE exposing ``anomaly_score``.
        features: DAE feature matrix.
        l1_gamma: L1 residual weighting from project configuration.

    Returns:
        One raw anomaly score per transaction.

    Raises:
        AttributeError: If the supplied model has no ``anomaly_score`` method.
        ValueError: If anomaly scores are invalid.
    """
    anomaly_method = getattr(autoencoder, "anomaly_score", None)

    if anomaly_method is None or not callable(anomaly_method):
        raise AttributeError("Autoencoder must expose a callable anomaly_score method")

    scores = anomaly_method(
        features,
        l1_gamma=l1_gamma,
        reduction="none",
    )

    if not isinstance(scores, torch.Tensor):
        raise TypeError("Autoencoder anomaly_score must return a tensor")

    validate_tensor(
        scores,
        name="Autoencoder anomaly scores",
        ndim=1,
        allow_empty=False,
        require_finite=True,
    )

    return scores


def velocity_features_from_frame(
    df: pd.DataFrame,
    *,
    sequence_col: str = "sequence_array",
    transaction_amount_index: int = 0,
) -> torch.Tensor:
    """Extract causal learned-gate velocity features from a processed split.

    The processed data pipeline stores each transaction's prior-only history
    in ``sequence_array``. This helper converts those histories into the
    velocity context consumed by the learned hybrid gate.

    Args:
        df: Processed transaction split containing historical sequences.
        sequence_col: Column containing per-row historical sequence arrays.
        transaction_amount_index: Position of TransactionAmt inside each
            sequence timestep.

    Returns:
        Float tensor of shape ``(n_samples, 2)``.

    Raises:
        ValueError: If the frame is empty.
        KeyError: If the historical sequence column is missing.
    """
    if df.empty:
        raise ValueError("Cannot extract velocity features from an empty frame")

    if sequence_col not in df.columns:
        raise KeyError(f"DataFrame is missing required sequence column {sequence_col!r}")

    features = extract_velocity_features(
        df[sequence_col].tolist(),
        transaction_amount_index=transaction_amount_index,
    )

    if features.shape[0] != len(df):
        raise ValueError("Velocity feature count does not match DataFrame row count")

    return features


def make_gate_data(
    anomaly_scores: torch.Tensor,
    ft_probabilities: torch.Tensor,
    velocity_features: torch.Tensor,
    labels: torch.Tensor,
) -> GateData:
    """Construct one learned-gate dataset from aligned upstream signals."""
    return GateData(
        anomaly_scores=anomaly_scores.detach().cpu().float(),
        ft_probabilities=ft_probabilities.detach().cpu().float(),
        velocity_features=velocity_features.detach().cpu().float(),
        labels=labels.detach().cpu().long(),
    )


def learned_gate_probabilities(
    gate: LearnedHybridGate,
    normalizer: PercentileNormalizer,
    anomaly_scores: torch.Tensor,
    ft_probabilities: torch.Tensor,
    velocity_features: torch.Tensor,
) -> torch.Tensor:
    """Generate fused fraud probabilities using a trained learned gate.

    The anomaly normalizer is reused without fitting so calibration or test
    observations cannot influence preprocessing learned from gate-training
    data.

    Args:
        gate: Trained learned hybrid gate.
        normalizer: Normalizer fitted on gate-training anomaly scores.
        anomaly_scores: Raw DAE anomaly scores.
        ft_probabilities: FT-CAT fraud probabilities.
        velocity_features: Causal transaction-velocity features.

    Returns:
        Learned fused fraud probabilities on CPU.
    """
    features = build_gate_features(
        anomaly_scores=anomaly_scores.detach().cpu().float(),
        transformer_probabilities=ft_probabilities.detach().cpu().float(),
        velocity_features=velocity_features.detach().cpu().float(),
        normalizer=normalizer,
        fit_normalizer=False,
    )

    try:
        gate_device = next(gate.parameters()).device
    except StopIteration:
        gate_device = torch.device("cpu")

    was_training = gate.training
    gate.eval()

    try:
        with torch.no_grad():
            probabilities = gate(features.to(gate_device))
    finally:
        if was_training:
            gate.train()

    probabilities = probabilities.detach().cpu()

    validate_probability_tensor(
        probabilities,
        name="Learned gate probabilities",
        ndim=1,
        allow_empty=False,
    )

    return probabilities


def train_hybrid_calibration_pipeline(
    gate_train_data: GateData,
    gate_val_data: GateData,
    conformal_anomaly_scores: torch.Tensor,
    conformal_ft_probabilities: torch.Tensor,
    conformal_velocity_features: torch.Tensor,
    conformal_labels: torch.Tensor,
    config: Config,
    device: str | None = None,
) -> HybridPipelineArtifacts:
    """Train the learned gate and calibrate conformal prediction.

    Gate parameters and anomaly normalization are learned exclusively from
    the gate-development period. The later conformal subset is used only
    after gate training has finished.

    Args:
        gate_train_data: Earlier learned-gate training observations.
        gate_val_data: Later gate-development observations used for early
            stopping.
        conformal_anomaly_scores: DAE scores for untouched calibration data.
        conformal_ft_probabilities: FT-CAT probabilities for untouched
            calibration data.
        conformal_velocity_features: Causal transaction-velocity features for
            untouched conformal calibration observations.
        conformal_labels: Binary labels for untouched calibration data.
        config: Central project configuration.
        device: Optional learned-gate training device.

    Returns:
        Trained gate, fitted normalizer, and calibrated conformal predictor.
    """
    gate, normalizer = train_gate(
        train_data=gate_train_data,
        val_data=gate_val_data,
        config=config,
        device=device,
    )

    conformal_probabilities = learned_gate_probabilities(
        gate=gate,
        normalizer=normalizer,
        anomaly_scores=conformal_anomaly_scores,
        ft_probabilities=conformal_ft_probabilities,
        velocity_features=conformal_velocity_features,
    )

    labels = conformal_labels.detach().cpu().long()

    if labels.ndim != 1:
        raise ValueError("Conformal labels must be one-dimensional")

    if labels.shape != conformal_probabilities.shape:
        raise ValueError("Conformal probabilities and labels must have matching shapes")

    alpha = float(
        config.nested_get(
            "evaluation.conformal.alpha",
            0.01,
        )
    )

    predictor = SplitConformalPredictor(alpha=alpha)

    predictor.fit(
        fraud_probabilities=conformal_probabilities,
        labels=labels,
    )

    logger.info(
        "Hybrid calibration pipeline complete - conformal samples: %d, " "alpha: %.4f",
        labels.numel(),
        alpha,
    )

    return HybridPipelineArtifacts(
        gate=gate,
        normalizer=normalizer,
        conformal_predictor=predictor,
    )
