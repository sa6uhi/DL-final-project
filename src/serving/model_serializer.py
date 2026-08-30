"""Model serialization: ``torch.export`` (EXIR) and ONNX with parity checks.

Serializes trained models to a portable EXIR graph via ``torch.export`` and
to a standard ONNX graph with dynamic batch axes, then asserts numerical
agreement between the original PyTorch model and each exported artifact
(``|y_torch - y_export| < 1e-4``).

Note: ``torch.jit.trace`` is deprecated and unsupported on Python 3.14+, so
TorchScript is replaced by ``torch.export`` (EXIR) which is the maintained
portable graph format.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from src.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_OPSET = 17
PARITY_TOLERANCE = 1.0e-4


class ReferenceEncoder(nn.Module):
    """Small deterministic MLP used to test serialization without checkpoints.

    The reference model mirrors the anomaly-scoring contract of the trained
    autoencoder: it embeds an input vector and returns a scalar residual
    score so the full serving path can be exercised before real training
    artifacts exist.

    Args:
        input_dim: Dimensionality of the input feature vector.
        latent_dim: Dimensionality of the internal embedding.
    """

    def __init__(self, input_dim: int, latent_dim: int = 16) -> None:
        """Initialize the reference scoring network."""
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, latent_dim),
        )
        self.score_head = nn.Sequential(
            nn.Linear(latent_dim, 1),
            nn.Softplus(beta=1.0, threshold=20.0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embed input features and return a per-sample residual score.

        Args:
            x: Input tensor of shape ``(batch, input_dim)``.

        Returns:
            Score tensor of shape ``(batch, 1)``.
        """
        return self.score_head(self.encoder(x))

    def anomaly_score(
        self, x: torch.Tensor, l1_gamma: float = 0.4, reduction: str = "none"
    ) -> torch.Tensor:
        """Match the autoencoder's anomaly scoring interface.

        The reference model is a direct score head, so ``l1_gamma`` has no
        effect here; the argument exists solely for interface parity.

        Args:
            x: Input tensor of shape ``(batch, input_dim)``.
            l1_gamma: Ignored; kept for interface compatibility.
            reduction: ``"none"``/``"mean"``/``"sum"``.

        Returns:
            Per-sample scores of shape ``(batch,)`` or a scalar.
        """
        scores = self.forward(x).squeeze(-1)
        if reduction == "mean":
            return scores.mean()
        if reduction == "sum":
            return scores.sum()
        return scores

    @staticmethod
    def build_from_state(meta: dict[str, Any]) -> "ReferenceEncoder":
        """Rebuild a reference model from its ``state_meta`` payload.

        Args:
            meta: Serialization metadata produced by :func:`state_meta`.

        Returns:
            Reconstructed reference model.
        """
        input_dim = int(meta["input_dim"])
        latent_dim = int(meta.get("latent_dim", 16))
        return ReferenceEncoder(input_dim=input_dim, latent_dim=latent_dim)

    def state_meta(self) -> dict[str, Any]:
        """Serialize constructor arguments needed to rebuild the model."""
        return {"input_dim": self.input_dim, "latent_dim": self.latent_dim}


def build_reference_model(input_dim: int, latent_dim: int = 16) -> ReferenceEncoder:
    """Construct a default reference model for early serving/testing.

    Args:
        input_dim: Dimensionality of the input feature vector.
        latent_dim: Dimensionality of the internal embedding.

    Returns:
        An unloaded ``ReferenceEncoder`` in evaluation mode.
    """
    model = ReferenceEncoder(input_dim=input_dim, latent_dim=latent_dim)
    model.eval()
    return model


def export_exir(
    model: nn.Module, path: str | Path, sample: torch.Tensor, device: str = "cpu"
) -> Path:
    """Export ``model`` to the portable EXIR graph format.

    The export is traced with the provided sample and supports dynamic batch
    sizes at inference time.

    Args:
        model: PyTorch module in evaluation mode.
        path: Destination ``.pt2`` file path.
        sample: Example input tensor of shape ``(batch, input_dim)``.
        device: Device used for export.

    Returns:
        The resolved destination path.

    Raises:
        ValueError: If the model is training mode, or export fails.
    """
    if model.training:
        raise ValueError("Model must be in eval mode before export")
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    model = model.to(device)
    sample = sample.to(device)
    with torch.no_grad():
        exported = torch.export.export(model, (sample,))
        torch.export.save(exported, out)
    logger.info("Exported EXIR model to %s", out)
    return out


def export_onnx(
    model: nn.Module,
    path: str | Path,
    sample: torch.Tensor,
    opset: int = DEFAULT_OPSET,
    input_name: str = "features",
    output_name: str = "scores",
    device: str = "cpu",
) -> Path:
    """Export ``model`` to a standard ONNX graph with dynamic batch axis.

    Args:
        model: PyTorch module in evaluation mode.
        path: Destination ``.onnx`` file path.
        sample: Example input tensor of shape ``(batch, input_dim)``.
        opset: ONNX operator set version to target.
        input_name: Name of the :math:`N \\times D` input tensor.
        output_name: Name of the :math:`N \\times 1` output tensor.
        device: Device used for export.

    Returns:
        The resolved destination path.

    Raises:
        ValueError: If the model is training mode, or export fails.
    """
    if model.training:
        raise ValueError("Model must be in eval mode before export")
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    model = model.to(device)
    sample = sample.to(device)
    batch_dim = torch.export.Dim("batch")
    dynamic_shapes = ({0: batch_dim},)
    with torch.no_grad():
        torch.onnx.export(
            model,
            sample,
            str(out),
            input_names=[input_name],
            output_names=[output_name],
            dynamic_shapes=dynamic_shapes,
            opset_version=opset,
        )
    logger.info("Exported ONNX model to %s (opset %d)", out, opset)
    return out


def load_exir(path: str | Path, device: str = "cpu") -> Any:
    """Load a serialized EXIR graph and return it as an eager callable.

    Args:
        path: EXIR artifact produced by :func:`export_exir`.
        device: Target device for the loaded module.

    Returns:
        A callable eager module running the exported graph.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    export_path = Path(path)
    if not export_path.is_file():
        raise FileNotFoundError(f"EXIR artifact not found: {export_path}")
    exported = torch.export.load(str(export_path))
    return exported.module().to(device)


def verify_parity(
    reference: nn.Module,
    artifact: Any,
    sample: torch.Tensor,
    tolerance: float = PARITY_TOLERANCE,
) -> float:
    """Measure maximum absolute output difference between two scoriing paths.

    Args:
        reference: Original PyTorch model.
        artifact: EXIR module, ONNX session output tensor, or TorchScript
            module handled through the common ``__call__`` protocol.
        sample: Input tensor shared by both models.
        tolerance: Maximum acceptable absolute difference.

    Returns:
        The observed maximum absolute difference.

    Raises:
        RuntimeError: If the difference exceeds ``tolerance``.
    """
    reference.eval()
    with torch.no_grad():
        y_ref = reference(sample)
        y_art = artifact(sample) if callable(artifact) else artifact
    max_diff = float((y_ref - y_art).abs().max().item())
    if max_diff > tolerance:
        raise RuntimeError(
            f"Serialization parity failed: max diff {max_diff:.2e} > {tolerance:.2e}"
        )
    logger.info("Parity verified: max diff %.6e (tol %.1e)", max_diff, tolerance)
    return max_diff
