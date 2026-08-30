"""Semi-supervised Deep Denoising Autoencoder for zero-day anomaly detection.

The encoder compresses input features into a low-dimensional latent space
(``D -> 256 -> 128 -> 32``) and the decoder reconstructs the input. The model
is trained exclusively on legitimate transactions (``y=0``); novel fraud
patterns violate the learned manifold and produce high reconstruction
residuals that serve as anomaly scores.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn


class DenoisingAutoencoder(nn.Module):
    """Deep denoising autoencoder with composite MSE + BCE reconstruction loss.

    The model applies optional deterministic input corruption (Gaussian noise
    plus random feature dropout) before encoding, learns a 32D bottleneck
    manifold, and scores anomalies via per-sample reconstruction residuals
    ``S(x) = ||x - x_bar||_2^2 + gamma * ||x - x_bar||_1``.
    """

    def __init__(
        self,
        input_dim: int,
        encoder_hidden_dims: list[int] | tuple[int, ...] | None = None,
        latent_dim: int = 32,
        dropout: float = 0.2,
        noise_std: float = 0.1,
        feature_dropout_prob: float = 0.1,
        activation_slope: float = 0.2,
        bias: bool = True,
    ) -> None:
        """Initialize the denoising autoencoder.

        Args:
            input_dim: Dimensionality of the input feature vector.
            encoder_hidden_dims: Hidden widths of the encoder; defaults to
                ``(256, 128)``.
            latent_dim: Bottleneck dimensionality; defaults to 32.
            dropout: Dropout probability after the first encoder layer.
            noise_std: Standard deviation of Gaussian corruption noise.
            feature_dropout_prob: Probability of zeroing individual features.
            activation_slope: Negative slope of LeakyReLU activations.
            bias: Whether to use bias in linear layers.

        Raises:
            ValueError: If ``input_dim`/``latent_dim`` are not positive, or
                hidden dimensions do not form a strictly decreasing sequence
                ending above ``latent_dim``.
        """
        super().__init__()
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {latent_dim}")
        if not 0 <= feature_dropout_prob <= 1:
            raise ValueError(
                f"feature_dropout_prob must be in [0, 1], got {feature_dropout_prob}"
            )

        hidden_dims = list(encoder_hidden_dims) if encoder_hidden_dims else [256, 128]
        if any(left <= right for left, right in zip(hidden_dims, hidden_dims[1:])):
            raise ValueError(
                f"encoder_hidden_dims must strictly decrease: got {hidden_dims}"
            )
        if hidden_dims[-1] <= latent_dim:
            raise ValueError(
                f"Encoder must narrow to the latent dim; got hidden {hidden_dims} "
                f"and latent {latent_dim}"
            )

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoder_hidden_dims = tuple(hidden_dims)
        self.dropout = dropout
        self.noise_std = noise_std
        self.feature_dropout_prob = feature_dropout_prob
        self.activation_slope = activation_slope
        self.bias = bias

        leaky = nn.LeakyReLU(negative_slope=activation_slope)

        encoder_layers: list[nn.Module] = []
        encoder_layers.append(nn.Linear(input_dim, hidden_dims[0], bias=bias))
        encoder_layers.append(nn.BatchNorm1d(hidden_dims[0]))
        encoder_layers.append(leaky)
        encoder_layers.append(nn.Dropout(dropout))
        for left, right in zip(hidden_dims, hidden_dims[1:] + [latent_dim]):
            encoder_layers.append(nn.Linear(left, right, bias=bias))
            encoder_layers.append(leaky)
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers: list[nn.Module] = []
        for left, right in zip(
            [latent_dim, *hidden_dims], [*hidden_dims, input_dim]
        ):
            decoder_layers.append(nn.Linear(left, right, bias=bias))
            if right != input_dim:
                decoder_layers.append(leaky)
        self.decoder = nn.Sequential(*decoder_layers)

    def corrupt(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Gaussian noise plus random feature dropout corruption.

        Args:
            x: Input tensor of shape ``(batch, input_dim)``.

        Returns:
            Corrupted input of the same shape as ``x``.
        """
        x = x + torch.randn_like(x) * self.noise_std
        mask = torch.rand_like(x) > self.feature_dropout_prob
        return x * mask.float()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Map input features to the latent bottleneck representation.

        Args:
            x: Input tensor of shape ``(batch, input_dim)``.

        Returns:
            Latent embeddings of shape ``(batch, latent_dim)``.
        """
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct input features from latent embeddings.

        Args:
            z: Latent embeddings of shape ``(batch, latent_dim)``.

        Returns:
            Reconstructed features of shape ``(batch, input_dim)``.
        """
        return self.decoder(z)

    def forward(self, x: torch.Tensor, corrupt: bool = False) -> torch.Tensor:
        """Full encode-decode pass.

        Args:
            x: Input tensor of shape ``(batch, input_dim)``.
            corrupt: If True, applies corruption before encoding.

        Returns:
            Reconstruction of shape ``(batch, input_dim)``.

        Raises:
            ValueError: If the input has not exactly two dimensions or its
                feature dimension mismatches ``input_dim``.
        """
        if x.dim() != 2 or x.size(-1) != self.input_dim:
            raise ValueError(
                f"Expected input of shape (batch, {self.input_dim}), got {tuple(x.shape)}"
            )
        if corrupt and self.training:
            x = self.corrupt(x)
        return self.decode(self.encode(x))

    def anomaly_score(
        self, x: torch.Tensor, l1_gamma: float = 0.4, reduction: str = "none"
    ) -> torch.Tensor:
        """Compute per-sample anomaly residual scores.

        ``S(x) = ||x - x_bar||_2^2 + gamma * ||x - x_bar||_1`` where the
        reconstruction is produced with corruption disabled.

        Args:
            x: Input tensor of shape ``(batch, input_dim)``.
            l1_gamma: Scaling of the L1 residual term.
            reduction: ``"none"`` returns per-sample scores,
                ``"mean"`` the batch average, ``"sum"`` the batch total.

        Returns:
            Anomaly scores of shape ``(batch,)`` or a scalar.

        Raises:
            ValueError: If ``l1_gamma`` is negative or ``reduction`` is
                unsupported.
        """
        if l1_gamma < 0:
            raise ValueError(f"l1_gamma must be non-negative, got {l1_gamma}")
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError(f"Unsupported reduction: {reduction!r}")
        with torch.no_grad():
            x_hat = self.forward(x)
        residual = x - x_hat
        scores = (residual.pow(2)).sum(dim=-1) + l1_gamma * residual.abs().sum(dim=-1)
        if reduction == "mean":
            return scores.mean()
        if reduction == "sum":
            return scores.sum()
        return scores

    @staticmethod
    def loss(
        x: torch.Tensor,
        x_hat: torch.Tensor,
        mse_weight: float = 0.5,
        bce_weight: float = 0.5,
    ) -> torch.Tensor:
        """Composite reconstruction loss: weighted MSE plus BCE.

        Args:
            x: Ground-truth input of shape ``(batch, input_dim)``.
            x_hat: Reconstruction of shape ``(batch, input_dim)``.
            mse_weight: Weight assigned to the MSE term.
            bce_weight: Weight assigned to the BCE term.

        Returns:
            Scalar composite loss tensor.

        Raises:
            ValueError: If the tensors are not shape-compatible.
        """
        if x.shape != x_hat.shape:
            raise ValueError(
                f"Shape mismatch: x {tuple(x.shape)} vs x_hat {tuple(x_hat.shape)}"
            )
        mse_term = nn.functional.mse_loss(x_hat, x)
        bce_term = nn.functional.binary_cross_entropy_with_logits(
            x_hat, x.clamp(0.0, 1.0)
        )
        return mse_weight * mse_term + bce_weight * bce_term

    def init_weights(self) -> None:
        """He-initialize all linear layers for faster convergence."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(
                    module.weight, a=math.sqrt(5), mode="fan_in", nonlinearity="leaky_relu"
                )
                if module.bias is not None:
                    fan_in = module.weight.size(1)
                    bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
                    nn.init.uniform_(module.bias, -bound, bound)

    def state_meta(self) -> dict[str, Any]:
        """Serialization metadata needed to rebuild the architecture.

        Returns:
            Dictionary with every constructor argument of this model.
        """
        return {
            "input_dim": self.input_dim,
            "encoder_hidden_dims": list(self.encoder_hidden_dims),
            "latent_dim": self.latent_dim,
            "dropout": self.dropout,
            "noise_std": self.noise_std,
            "feature_dropout_prob": self.feature_dropout_prob,
            "activation_slope": self.activation_slope,
            "bias": self.bias,
        }
