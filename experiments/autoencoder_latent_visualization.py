"""Visualize the trained DAE bottleneck with t-SNE.

The experiment loads the processed evaluation split, materializes the exact
numeric feature contract consumed by the DAE, obtains deterministic latent
embeddings from the trained autoencoder, and projects a stratified sample into
two dimensions with t-SNE.

The resulting figure is intended for the IEEE report and analyst-facing
interpretation of fraud-vs-legitimate separation in the learned DAE manifold.
"""

# Import necessary modules and libraries
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

from src.training.dae_features import (
    materialize_dae_features,
    resolve_dae_feature_columns,
)
from src.training.train_autoencoder import load_checkpoint
from src.utils.config import Config, load_config
from src.utils.logger import get_logger, setup_logging
from src.utils.seed import seed_everything

logger = get_logger(__name__)

DEFAULT_CONFIG = Path("config/config.yaml")


def validate_labels(labels: np.ndarray) -> None:
    """Validate binary fraud labels.

    Args:
        labels: One-dimensional array containing legitimate/fraud labels.

    Raises:
        ValueError: If labels are empty, non-finite, non-binary, or not 1D.
    """
    if labels.ndim != 1:
        raise ValueError("labels must be one-dimensional")

    if labels.size == 0:
        raise ValueError("labels must not be empty")

    if not np.isfinite(labels).all():
        raise ValueError("labels must contain only finite values")

    if not np.isin(labels, [0, 1]).all():
        raise ValueError("labels must contain only 0 or 1")


def stratified_sample_indices(
    labels: np.ndarray,
    max_samples: int,
    random_state: int,
) -> np.ndarray:
    """Return deterministic class-stratified sample indices.

    Fraud is typically rare, so ordinary random sampling can leave too few
    positive examples for a useful latent-space visualization. This sampler
    attempts to preserve both classes while never duplicating observations.

    Args:
        labels: Binary labels with shape ``(n_samples,)``.
        max_samples: Maximum number of rows to retain.
        random_state: Seed controlling deterministic sampling.

    Returns:
        Sorted integer indices into the original array.

    Raises:
        ValueError: If ``max_samples`` is smaller than 2.
    """
    validate_labels(labels)

    if max_samples < 2:
        raise ValueError("max_samples must be at least 2")

    n_samples = labels.shape[0]

    if n_samples <= max_samples:
        return np.arange(n_samples, dtype=np.int64)

    rng = np.random.default_rng(random_state)

    legitimate = np.flatnonzero(labels == 0)
    fraud = np.flatnonzero(labels == 1)

    if legitimate.size == 0 or fraud.size == 0:
        chosen = rng.choice(
            n_samples,
            size=max_samples,
            replace=False,
        )
        return np.sort(chosen.astype(np.int64))

    fraud_target = min(
        fraud.size,
        max(1, max_samples // 2),
    )
    legitimate_target = max_samples - fraud_target

    if legitimate_target > legitimate.size:
        shortfall = legitimate_target - legitimate.size
        legitimate_target = legitimate.size
        fraud_target = min(fraud.size, fraud_target + shortfall)

    if fraud_target > fraud.size:
        shortfall = fraud_target - fraud.size
        fraud_target = fraud.size
        legitimate_target = min(
            legitimate.size,
            legitimate_target + shortfall,
        )

    fraud_indices = rng.choice(
        fraud,
        size=fraud_target,
        replace=False,
    )
    legitimate_indices = rng.choice(
        legitimate,
        size=legitimate_target,
        replace=False,
    )

    chosen = np.concatenate(
        (
            legitimate_indices,
            fraud_indices,
        )
    )

    return np.sort(chosen.astype(np.int64))


def encode_latent_space(
    model: torch.nn.Module,
    features: torch.Tensor,
    device: str = "cpu",
    batch_size: int = 2048,
) -> np.ndarray:
    """Encode input features into deterministic DAE bottleneck vectors.

    Args:
        model: Trained DAE exposing an ``encode`` method.
        features: Feature tensor shaped ``(n_samples, input_dim)``.
        device: Torch execution device.
        batch_size: Number of rows encoded per batch.

    Returns:
        NumPy array of latent vectors shaped ``(n_samples, latent_dim)``.

    Raises:
        ValueError: If inputs are empty, malformed, or batch size is invalid.
        AttributeError: If the model has no callable ``encode`` method.
    """
    if features.ndim != 2:
        raise ValueError("features must be two-dimensional")

    if features.shape[0] == 0:
        raise ValueError("features must not be empty")

    if not torch.isfinite(features).all():
        raise ValueError("features must contain only finite values")

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    encode_method = getattr(model, "encode", None)

    if not callable(encode_method):
        raise AttributeError("model must expose a callable encode method")

    model = model.to(device)
    model.eval()

    latent_batches: list[torch.Tensor] = []

    with torch.no_grad():
        for start in range(0, features.shape[0], batch_size):
            end = min(start + batch_size, features.shape[0])
            batch = features[start:end].to(device)

            latent = encode_method(batch)

            if not isinstance(latent, torch.Tensor):
                raise TypeError("model.encode must return a torch.Tensor")

            if latent.ndim != 2:
                raise ValueError("latent embeddings must be two-dimensional")

            if not torch.isfinite(latent).all():
                raise ValueError("latent embeddings must contain only finite values")

            latent_batches.append(latent.detach().cpu())

    encoded = torch.cat(latent_batches, dim=0).numpy()

    logger.info(
        "Encoded %d transactions into %d-dimensional latent space",
        encoded.shape[0],
        encoded.shape[1],
    )

    return encoded


def project_tsne(
    latent_vectors: np.ndarray,
    random_state: int,
    perplexity: float = 30.0,
) -> np.ndarray:
    """Project latent vectors to two dimensions using t-SNE.

    Args:
        latent_vectors: Matrix shaped ``(n_samples, latent_dim)``.
        random_state: Reproducibility seed.
        perplexity: Requested t-SNE perplexity.

    Returns:
        Two-dimensional embedding shaped ``(n_samples, 2)``.

    Raises:
        ValueError: If the latent matrix is malformed or too small.
    """
    latent_vectors = np.asarray(latent_vectors, dtype=np.float32)

    if latent_vectors.ndim != 2:
        raise ValueError("latent_vectors must be two-dimensional")

    if latent_vectors.shape[0] < 3:
        raise ValueError("At least three latent vectors are required for t-SNE")

    if latent_vectors.shape[1] == 0:
        raise ValueError("latent_vectors must contain at least one feature")

    if not np.isfinite(latent_vectors).all():
        raise ValueError("latent_vectors must contain only finite values")

    if perplexity <= 0:
        raise ValueError("perplexity must be positive")

    effective_perplexity = min(
        float(perplexity),
        float(latent_vectors.shape[0] - 1),
    )

    logger.info(
        "Running t-SNE for %d samples with perplexity %.2f",
        latent_vectors.shape[0],
        effective_perplexity,
    )

    reducer = TSNE(
        n_components=2,
        perplexity=effective_perplexity,
        random_state=random_state,
        init="pca",
        learning_rate="auto",
        max_iter=1000,
    )

    embedding = reducer.fit_transform(latent_vectors)

    if embedding.shape != (latent_vectors.shape[0], 2):
        raise ValueError("t-SNE returned an unexpected embedding shape")

    if not np.isfinite(embedding).all():
        raise ValueError("t-SNE embedding contains non-finite values")

    return embedding.astype(np.float32, copy=False)


def latent_silhouette_score(
    latent_vectors: np.ndarray,
    labels: np.ndarray,
) -> float | None:
    """Measure fraud-vs-legitimate separation in latent space.

    The score is computed in the original DAE bottleneck rather than the
    two-dimensional t-SNE projection, avoiding interpretation of t-SNE
    distances as globally meaningful geometry.

    Args:
        latent_vectors: DAE latent vectors.
        labels: Binary fraud labels.

    Returns:
        Silhouette score, or ``None`` if both classes are not represented.
    """
    validate_labels(labels)

    latent_vectors = np.asarray(latent_vectors, dtype=np.float32)

    if latent_vectors.ndim != 2:
        raise ValueError("latent_vectors must be two-dimensional")

    if latent_vectors.shape[0] != labels.shape[0]:
        raise ValueError("latent_vectors and labels must have matching sample counts")

    if np.unique(labels).size < 2:
        return None

    if latent_vectors.shape[0] <= np.unique(labels).size:
        return None

    return float(
        silhouette_score(
            latent_vectors,
            labels,
            metric="euclidean",
        )
    )


def plot_latent_space(
    embedding: np.ndarray,
    labels: np.ndarray,
    output_path: str | Path,
    silhouette: float | None = None,
    show_plot: bool = False,
) -> Path:
    """Plot and save the 2D t-SNE latent-space visualization.

    Args:
        embedding: t-SNE coordinates shaped ``(n_samples, 2)``.
        labels: Binary fraud labels.
        output_path: Figure destination.
        silhouette: Optional silhouette score from original latent space.
        show_plot: Whether to display the figure interactively.

    Returns:
        Path to the saved figure.
    """
    embedding = np.asarray(embedding, dtype=np.float32)
    labels = np.asarray(labels)

    if embedding.ndim != 2 or embedding.shape[1] != 2:
        raise ValueError("embedding must have shape (n_samples, 2)")

    validate_labels(labels)

    if embedding.shape[0] != labels.shape[0]:
        raise ValueError("embedding and labels must have matching sample counts")

    if not np.isfinite(embedding).all():
        raise ValueError("embedding must contain only finite values")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    legitimate_mask = labels == 0
    fraud_mask = labels == 1

    figure, axis = plt.subplots(figsize=(10, 7))

    axis.scatter(
        embedding[legitimate_mask, 0],
        embedding[legitimate_mask, 1],
        s=16,
        alpha=0.45,
        color="#A8C3A0",
        label="Legitimate",
    )

    axis.scatter(
        embedding[fraud_mask, 0],
        embedding[fraud_mask, 1],
        s=28,
        alpha=0.8,
        color="#1F6B45",
        label="Fraud",
    )

    axis.set_title("DAE Latent Space — t-SNE Projection")
    axis.set_xlabel("t-SNE Dimension 1")
    axis.set_ylabel("t-SNE Dimension 2")
    axis.legend()
    axis.grid(alpha=0.15)

    if silhouette is not None:
        axis.text(
            0.02,
            0.02,
            f"Latent silhouette: {silhouette:.3f}",
            transform=axis.transAxes,
            fontsize=9,
            verticalalignment="bottom",
        )

    figure.tight_layout()
    figure.savefig(
        output,
        dpi=300,
        bbox_inches="tight",
    )

    logger.info("Saved DAE latent-space figure to %s", output)

    if show_plot:
        plt.show()

    plt.close(figure)

    return output


def run_visualization(
    data: pd.DataFrame,
    config: Config,
    checkpoint_path: str | Path,
    output_path: str | Path,
    max_samples: int,
    random_state: int,
    perplexity: float,
    batch_size: int,
    device: str,
    show_plot: bool,
) -> tuple[Path, float | None]:
    """Run the complete DAE latent-space visualization pipeline.

    Args:
        data: Processed labelled evaluation DataFrame.
        config: Loaded project configuration.
        checkpoint_path: Trained DAE checkpoint.
        output_path: Figure destination.
        max_samples: Maximum number of rows used for visualization.
        random_state: Sampling and t-SNE seed.
        perplexity: t-SNE perplexity.
        batch_size: DAE encoding batch size.
        device: Torch device.
        show_plot: Whether to display the plot interactively.

    Returns:
        Saved figure path and optional latent-space silhouette score.
    """
    target_col = str(config.features.target_col)

    if target_col not in data.columns:
        raise KeyError(f"Evaluation data is missing target column: {target_col}")

    labels = data[target_col].to_numpy()
    validate_labels(labels)

    sample_indices = stratified_sample_indices(
        labels=labels,
        max_samples=max_samples,
        random_state=random_state,
    )

    sampled_df = data.iloc[sample_indices].copy()
    sampled_labels = labels[sample_indices].astype(np.int64, copy=False)

    model = load_checkpoint(
        checkpoint_path,
        device=device,
    )

    feature_columns = resolve_dae_feature_columns(
        sampled_df,
        non_feature_cols=list(config.data.non_feature_cols),
        expected_dim=int(model.input_dim),
    )

    feature_tensor = materialize_dae_features(
        sampled_df,
        feature_columns,
    )

    latent_vectors = encode_latent_space(
        model=model,
        features=feature_tensor,
        device=device,
        batch_size=batch_size,
    )

    silhouette = latent_silhouette_score(
        latent_vectors,
        sampled_labels,
    )

    if silhouette is not None:
        logger.info(
            "Fraud-vs-legitimate latent silhouette score: %.4f",
            silhouette,
        )
    else:
        logger.warning(
            "Silhouette score unavailable because the sample does not contain both classes"
        )

    embedding = project_tsne(
        latent_vectors=latent_vectors,
        random_state=random_state,
        perplexity=perplexity,
    )

    figure_path = plot_latent_space(
        embedding=embedding,
        labels=sampled_labels,
        output_path=output_path,
        silhouette=silhouette,
        show_plot=show_plot,
    )

    return figure_path, silhouette


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for DAE latent-space visualization."""
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Processed labelled parquet split; defaults to configured test data",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Trained DAE checkpoint; defaults to models/checkpoints/autoencoder.pt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output figure path",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=5000,
        help="Maximum stratified sample size used by t-SNE",
    )
    parser.add_argument(
        "--perplexity",
        type=float,
        default=30.0,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2048,
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
    )
    parser.add_argument(
        "--show-plot",
        action="store_true",
    )

    args = parser.parse_args(argv)

    config = load_config(args.config)

    setup_logging(
        level=str(config.logging.level),
        log_file=str(config.get_path("logging.log_file")),
    )

    random_state = int(config.seed)
    seed_everything(random_state)

    data_path = args.data or config.get_path("data.test_data_path")
    checkpoint_path = args.checkpoint or (config.get_path("paths.checkpoints") / "autoencoder.pt")
    output_path = args.output or (
        config.get_path("paths.figures") / "autoencoder_latent" / "tsne_latent_space.png"
    )

    logger.info("Loading processed evaluation data from %s", data_path)
    data = pd.read_parquet(data_path)

    run_visualization(
        data=data,
        config=config,
        checkpoint_path=checkpoint_path,
        output_path=output_path,
        max_samples=args.max_samples,
        random_state=random_state,
        perplexity=args.perplexity,
        batch_size=args.batch_size,
        device=args.device,
        show_plot=args.show_plot,
    )


if __name__ == "__main__":
    main()
