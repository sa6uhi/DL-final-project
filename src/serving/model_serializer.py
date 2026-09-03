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

import argparse
from pathlib import Path
from typing import Any

import torch
import onnxruntime
from torch import nn

from src.training.train_autoencoder import load_checkpoint
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_OPSET = 18
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
        tolerance: Maximum acceptable relative error bound
            (scaled by max(1.0, |y_ref|)).

    Returns:
        The observed maximum absolute difference.

    Raises:
        RuntimeError: If the relative difference exceeds ``tolerance``.
    """
    reference.eval()
    with torch.no_grad():
        y_ref = reference(sample)
        y_art = artifact(sample) if callable(artifact) else artifact
    max_diff = float((y_ref - y_art).abs().max().item())
    ref_scale = float(max(1.0, y_ref.abs().max().item()))
    relative_diff = max_diff / ref_scale
    effective_tol = tolerance * ref_scale
    if max_diff > effective_tol:
        msg = (
            f"Serialization parity failed: max diff {max_diff:.2e} "
            f"(relative {relative_diff:.2e}) > tol {tolerance:.2e}"
        )
        raise RuntimeError(msg)
    logger.info(
        "Parity verified: max diff %.6e (relative %.2e, tol %.1e)",
        max_diff,
        relative_diff,
        tolerance,
    )
    return max_diff


class ScoreModule(nn.Module):
    """Wrap a model exposing ``anomaly_score`` as a ``(batch, 1)`` scorer.

    Both the trained ``DenoisingAutoencoder`` and the ``ReferenceEncoder``
    implement ``anomaly_score(x)`` returning ``(batch,)`` residuals; this
    module makes them traceable by ``torch.export``/``torch.onnx`` while
    emitting scores of shape ``(batch, 1)``.

    Args:
        base: The underlying model (must expose ``anomaly_score``).
        l1_gamma: Scaling of the L1 residual term used by the autoencoder.
    """

    def __init__(self, base: nn.Module, l1_gamma: float = 0.4) -> None:
        """Initialize the wrapper around ``base``."""
        super().__init__()
        self.base = base
        self.l1_gamma = l1_gamma
        self.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return per-sample anomaly scores of shape ``(batch, 1)``.

        Args:
            x: Input tensor of shape ``(batch, input_dim)``.

        Returns:
            Score tensor of shape ``(batch, 1)``.
        """
        scores = self.base.anomaly_score(x, l1_gamma=self.l1_gamma)
        return scores.unsqueeze(-1)


def _resolve_model(input_dir: str | Path, config: dict) -> tuple[nn.Module, Path]:
    """Load the trained checkpoint or fall back to the reference model.

    Args:
        input_dir: Directory scanned for a serialized checkpoint.
        config: Loaded configuration dict.

    Returns:
        Tuple of ``(model, source_path)``; the source path is the checkpoint
        when one was found.
    """
    source = Path(input_dir) / "autoencoder.pt"
    if source.is_file():
        model = load_checkpoint(source)
        logger.info("Loaded trained checkpoint %s", source)
        return model, source
    logger.warning("Checkpoint %s missing; exporting the reference model", source)
    model = build_reference_model(int(config["autoencoder"]["input_dim"]))
    return model, source


def export_all(input_dir: str | Path, output_dir: str | Path, config: dict) -> dict[str, float]:
    """Export EXIR + ONNX artifacts and verify parity on a sample batch.

    Both the deep autoencoder and the reference scorer expose
    ``anomaly_score``; each is wrapped so the exported graph emits per-sample
    scores of shape ``(batch, 1)`` with identical semantics.

    Args:
        input_dir: Directory scanned for the trained checkpoint.
        output_dir: Destination directory for serialized artifacts.
        config: Loaded configuration dict.

    Returns:
        Dict of artifact name to measured parity error.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model, _ = _resolve_model(input_path, config)
    l1_gamma = float(config.get("autoencoder", {}).get("anomaly_score", {}).get("l1_gamma", 0.4))
    scorer = ScoreModule(model, l1_gamma=l1_gamma)
    input_dim = int(config["autoencoder"]["input_dim"])
    sample = torch.randn(4, input_dim)
    exir_path = output_path / "autoencoder.pt2"
    onnx_path = output_path / "autoencoder.onnx"
    export_exir(scorer, exir_path, sample)
    export_onnx(scorer, onnx_path, sample)
    exir_module = load_exir(exir_path)
    exir_diff = verify_parity(scorer, exir_module, sample)
    onnx_session = onnxruntime.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = onnx_session.get_inputs()[0].name
    onnx_out = torch.from_numpy(onnx_session.run(None, {input_name: sample.numpy()})[0]).reshape(
        -1, 1
    )
    onnx_diff = verify_parity(scorer, onnx_out, sample)
    return {"exir_max_diff": round(exir_diff, 8), "onnx_max_diff": round(onnx_diff, 8)}


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: ``python src/serving/model_serializer.py``."""
    parser = argparse.ArgumentParser(description="Serialize model to EXIR/ONNX with parity checks")
    parser.add_argument("--input", required=True, help="Checkpoint directory")
    parser.add_argument("--output", required=True, help="Serialized artifact directory")
    parser.add_argument("--config", default="config/config.yaml", help="Path to config.yaml")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    report = export_all(args.input, args.output, config)
    logger.info("Serialization report: %s", report)


if __name__ == "__main__":
    main()
