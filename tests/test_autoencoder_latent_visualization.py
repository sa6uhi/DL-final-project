"""Tests for the DAE latent-space t-SNE visualization experiment."""

# Import necessary modules and libraries for testing
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn


from experiments.autoencoder_latent_visualization import (
    encode_latent_space,
    latent_silhouette_score,
    plot_latent_space,
    project_tsne,
    stratified_sample_indices,
    validate_labels,
)


class DummyEncoder(nn.Module):
    """Minimal deterministic encoder used by latent-space tests."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Linear(4, 2, bias=False)

        with torch.no_grad():
            self.encoder.weight.copy_(
                torch.tensor(
                    [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                    ],
                    dtype=torch.float32,
                )
            )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return deterministic two-dimensional embeddings."""
        return self.encoder(x)


class InvalidEncoder(nn.Module):
    """Encoder-like module intentionally missing an encode method."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def test_validate_labels_accepts_binary_labels() -> None:
    labels = np.array([0, 1, 0, 1], dtype=np.int64)

    validate_labels(labels)


@pytest.mark.parametrize(
    ("labels", "message"),
    [
        (np.array([], dtype=np.int64), "labels must not be empty"),
        (
            np.array([[0, 1]], dtype=np.int64),
            "labels must be one-dimensional",
        ),
        (
            np.array([0.0, np.nan], dtype=np.float64),
            "labels must contain only finite values",
        ),
        (
            np.array([0, 2], dtype=np.int64),
            "labels must contain only 0 or 1",
        ),
    ],
)
def test_validate_labels_rejects_invalid_inputs(
    labels: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_labels(labels)


def test_stratified_sample_is_deterministic() -> None:
    labels = np.array(
        [0] * 90 + [1] * 10,
        dtype=np.int64,
    )

    first = stratified_sample_indices(
        labels=labels,
        max_samples=20,
        random_state=42,
    )

    second = stratified_sample_indices(
        labels=labels,
        max_samples=20,
        random_state=42,
    )

    np.testing.assert_array_equal(first, second)


def test_stratified_sample_preserves_both_classes() -> None:
    labels = np.array(
        [0] * 90 + [1] * 10,
        dtype=np.int64,
    )

    indices = stratified_sample_indices(
        labels=labels,
        max_samples=20,
        random_state=7,
    )

    sampled_labels = labels[indices]

    assert indices.shape == (20,)
    assert set(np.unique(sampled_labels)) == {0, 1}


def test_stratified_sample_returns_all_when_below_limit() -> None:
    labels = np.array([0, 1, 0, 1], dtype=np.int64)

    indices = stratified_sample_indices(
        labels=labels,
        max_samples=10,
        random_state=1,
    )

    np.testing.assert_array_equal(
        indices,
        np.arange(4, dtype=np.int64),
    )


def test_stratified_sample_rejects_tiny_limit() -> None:
    labels = np.array([0, 1, 0], dtype=np.int64)

    with pytest.raises(
        ValueError,
        match="max_samples must be at least 2",
    ):
        stratified_sample_indices(
            labels=labels,
            max_samples=1,
            random_state=1,
        )


def test_encode_latent_space_returns_expected_shape() -> None:
    model = DummyEncoder()

    features = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
        ],
        dtype=torch.float32,
    )

    latent = encode_latent_space(
        model=model,
        features=features,
        device="cpu",
        batch_size=2,
    )

    assert latent.shape == (3, 2)

    np.testing.assert_allclose(
        latent,
        np.array(
            [
                [1.0, 2.0],
                [5.0, 6.0],
                [9.0, 10.0],
            ],
            dtype=np.float32,
        ),
    )


def test_encode_latent_space_requires_encode_method() -> None:
    model = InvalidEncoder()

    features = torch.ones(
        (3, 4),
        dtype=torch.float32,
    )

    with pytest.raises(
        AttributeError,
        match="model must expose a callable encode method",
    ):
        encode_latent_space(
            model=model,
            features=features,
        )


def test_encode_latent_space_rejects_non_finite_features() -> None:
    model = DummyEncoder()

    features = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],
            [1.0, float("nan"), 3.0, 4.0],
        ],
        dtype=torch.float32,
    )

    with pytest.raises(
        ValueError,
        match="features must contain only finite values",
    ):
        encode_latent_space(
            model=model,
            features=features,
        )


def test_project_tsne_returns_two_dimensions() -> None:
    rng = np.random.default_rng(42)

    latent = rng.normal(
        size=(20, 4),
    ).astype(np.float32)

    embedding = project_tsne(
        latent_vectors=latent,
        random_state=42,
        perplexity=5.0,
    )

    assert embedding.shape == (20, 2)
    assert np.isfinite(embedding).all()


def test_project_tsne_is_deterministic_for_fixed_seed() -> None:
    rng = np.random.default_rng(17)

    latent = rng.normal(
        size=(20, 4),
    ).astype(np.float32)

    first = project_tsne(
        latent_vectors=latent,
        random_state=123,
        perplexity=5.0,
    )

    second = project_tsne(
        latent_vectors=latent,
        random_state=123,
        perplexity=5.0,
    )

    np.testing.assert_allclose(
        first,
        second,
        rtol=1e-5,
        atol=1e-5,
    )


def test_latent_silhouette_score_returns_float() -> None:
    latent = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [0.0, 0.1],
            [5.0, 5.0],
            [5.1, 5.0],
            [5.0, 5.1],
        ],
        dtype=np.float32,
    )

    labels = np.array(
        [0, 0, 0, 1, 1, 1],
        dtype=np.int64,
    )

    score = latent_silhouette_score(
        latent_vectors=latent,
        labels=labels,
    )

    assert score is not None
    assert score > 0.9


def test_latent_silhouette_score_returns_none_for_single_class() -> None:
    latent = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [0.0, 0.1],
        ],
        dtype=np.float32,
    )

    labels = np.zeros(
        3,
        dtype=np.int64,
    )

    score = latent_silhouette_score(
        latent_vectors=latent,
        labels=labels,
    )

    assert score is None


def test_plot_latent_space_saves_figure(
    tmp_path: Path,
) -> None:
    embedding = np.array(
        [
            [0.0, 0.0],
            [0.5, 0.25],
            [5.0, 5.0],
            [5.5, 5.25],
        ],
        dtype=np.float32,
    )

    labels = np.array(
        [0, 0, 1, 1],
        dtype=np.int64,
    )

    output = tmp_path / "autoencoder_latent" / "tsne_latent_space.png"

    result = plot_latent_space(
        embedding=embedding,
        labels=labels,
        output_path=output,
        silhouette=0.75,
        show_plot=False,
    )

    assert result == output
    assert output.is_file()
    assert output.stat().st_size > 0
