"""Build real probability inputs for conformal triage evaluation."""

# Import necessary modules and libraries
from __future__ import annotations

import argparse
import gc
from pathlib import Path

import numpy as np
import torch

from experiments.hybrid_gating_seed_robustness import build_gate_data
from src.training.feature_selection import FeatureSpec
from src.training.hybrid_pipeline import (
    create_hybrid_data_split,
    learned_gate_probabilities,
)
from src.training.train_autoencoder import (
    load_checkpoint as load_autoencoder_checkpoint,
)
from src.training.train_hybrid_gating import (
    load_checkpoint as load_gate_checkpoint,
)
from src.training.train_transformer import (
    _load_split,
    load_ft_transformer,
)
from src.utils.config import Config, load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_CONFIG = Path("config/config.yaml")
DEFAULT_OUTPUT = Path("results/conformal/conformal_inputs.npz")


# Define a function to generate gate probabilities and labels for a given frame
def generate_gate_probabilities(
    frame,
    autoencoder: torch.nn.Module,
    ft_model: torch.nn.Module,
    feature_spec: FeatureSpec,
    gate: torch.nn.Module,
    normalizer,
    l1_gamma: float,
    non_feature_cols: list[str],
    dae_expected_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate fixed learned-gate probabilities and labels for one frame."""
    gate_data = build_gate_data(
        frame=frame,
        autoencoder=autoencoder,
        ft_model=ft_model,
        feature_spec=feature_spec,
        l1_gamma=l1_gamma,
        non_feature_cols=non_feature_cols,
        dae_expected_dim=dae_expected_dim,
    )

    probabilities = learned_gate_probabilities(
        gate=gate,
        normalizer=normalizer,
        anomaly_scores=gate_data.anomaly_scores,
        ft_probabilities=gate_data.ft_probabilities,
        velocity_features=gate_data.velocity_features,
    )

    probabilities_array = probabilities.detach().cpu().numpy().astype(np.float64, copy=False)
    labels_array = gate_data.labels.detach().cpu().numpy().astype(np.int64, copy=False)

    if probabilities_array.ndim != 1:
        raise ValueError("Gate probabilities must be one-dimensional")

    if labels_array.ndim != 1:
        raise ValueError("Gate labels must be one-dimensional")

    if probabilities_array.shape[0] != labels_array.shape[0]:
        raise ValueError("Gate probabilities and labels must have matching lengths")

    if probabilities_array.size == 0:
        raise ValueError("Gate probability output must not be empty")

    if not np.isfinite(probabilities_array).all():
        raise ValueError("Gate probabilities must contain only finite values")

    if np.any((probabilities_array < 0.0) | (probabilities_array > 1.0)):
        raise ValueError("Gate probabilities must lie in [0, 1]")

    if not np.isin(labels_array, [0, 1]).all():
        raise ValueError("Gate labels must be binary")

    return probabilities_array, labels_array


# Define a function to generate gate probabilities and labels in batches for large frames
def generate_gate_probabilities_batched(
    frame,
    autoencoder: torch.nn.Module,
    ft_model: torch.nn.Module,
    feature_spec: FeatureSpec,
    gate: torch.nn.Module,
    normalizer,
    l1_gamma: float,
    non_feature_cols: list[str],
    dae_expected_dim: int,
    batch_rows: int = 5000,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate gate probabilities and labels in bounded row batches."""
    if batch_rows <= 0:
        raise ValueError("batch_rows must be positive")

    probability_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []

    total_rows = len(frame)

    for start in range(0, total_rows, batch_rows):
        stop = min(start + batch_rows, total_rows)
        batch = frame.iloc[start:stop]

        logger.info(
            "Processing rows %d-%d of %d",
            start,
            stop,
            total_rows,
        )

        probabilities, labels = generate_gate_probabilities(
            frame=batch,
            autoencoder=autoencoder,
            ft_model=ft_model,
            feature_spec=feature_spec,
            gate=gate,
            normalizer=normalizer,
            l1_gamma=l1_gamma,
            non_feature_cols=non_feature_cols,
            dae_expected_dim=dae_expected_dim,
        )

        probability_parts.append(probabilities)
        label_parts.append(labels)

        del batch
        gc.collect()

    return (
        np.concatenate(probability_parts),
        np.concatenate(label_parts),
    )


# Define the main function to build the conformal archive
def build_conformal_archive(
    config: Config,
    output_path: Path,
    device: str = "cpu",
) -> Path:
    """Build calibration and final-test probability arrays sequentially."""
    validation_path = Path(config.data.val_data_path)
    test_path = Path(config.data.test_data_path)
    checkpoint_dir = Path(config.paths.checkpoints)

    autoencoder_path = checkpoint_dir / "autoencoder.pt"
    transformer_path = checkpoint_dir / "ft_transformer.pt"
    gate_path = Path(config.hybrid_gating.learned.checkpoint_path)

    autoencoder = load_autoencoder_checkpoint(
        autoencoder_path,
        device=device,
    )
    autoencoder.eval()

    ft_model, ft_payload = load_ft_transformer(
        transformer_path,
        device=device,
    )
    ft_model.eval()

    gate, normalizer = load_gate_checkpoint(
        gate_path,
        device=device,
    )
    gate.eval()

    feature_spec_payload = ft_payload.get("feature_spec")
    if not isinstance(feature_spec_payload, dict):
        raise ValueError("FT-CAT checkpoint is missing a valid feature_spec")

    feature_spec = FeatureSpec.from_dict(feature_spec_payload)

    l1_gamma = float(config.autoencoder.anomaly_score.l1_gamma)
    non_feature_cols = [str(col) for col in config.data.non_feature_cols]
    dae_expected_dim = int(config.autoencoder.input_dim)

    logger.info("Loading validation split")
    validation_df = _load_split(validation_path)

    hybrid_split = create_hybrid_data_split(validation_df, config)
    calibration_df = hybrid_split.conformal_calibration

    del validation_df
    del hybrid_split
    gc.collect()

    logger.info(
        "Generating conformal calibration probabilities from %d rows",
        len(calibration_df),
    )

    calibration_probabilities, calibration_labels = generate_gate_probabilities_batched(
        frame=calibration_df,
        autoencoder=autoencoder,
        ft_model=ft_model,
        feature_spec=feature_spec,
        gate=gate,
        normalizer=normalizer,
        l1_gamma=l1_gamma,
        non_feature_cols=non_feature_cols,
        dae_expected_dim=dae_expected_dim,
    )

    del calibration_df
    gc.collect()

    logger.info(
        "Calibration inference complete: %d rows",
        calibration_probabilities.size,
    )

    logger.info("Loading final chronological test split")
    test_df = _load_split(test_path)

    logger.info(
        "Generating final chronological test probabilities from %d rows",
        len(test_df),
    )

    eval_probabilities, eval_labels = generate_gate_probabilities_batched(
        frame=test_df,
        autoencoder=autoencoder,
        ft_model=ft_model,
        feature_spec=feature_spec,
        gate=gate,
        normalizer=normalizer,
        l1_gamma=l1_gamma,
        non_feature_cols=non_feature_cols,
        dae_expected_dim=dae_expected_dim,
    )

    del test_df
    gc.collect()

    logger.info(
        "Test inference complete: %d rows",
        eval_probabilities.size,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path,
        calibration_probabilities=calibration_probabilities,
        calibration_labels=calibration_labels,
        eval_probabilities=eval_probabilities,
        eval_labels=eval_labels,
    )

    logger.info(
        "Saved conformal archive to %s (calibration=%d, evaluation=%d)",
        output_path,
        calibration_probabilities.size,
        eval_probabilities.size,
    )

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build real learned-gate probability archive for conformal evaluation."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    build_conformal_archive(
        config=config,
        output_path=args.output,
        device=args.device,
    )


if __name__ == "__main__":
    main()
