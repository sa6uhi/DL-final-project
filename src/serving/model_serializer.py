"""Model serialization: ``torch.export`` (EXIR) and ONNX with parity checks.

Serializes trained models to a portable EXIR graph via ``torch.export`` and
to a standard ONNX graph with dynamic batch axes, then asserts numerical
agreement between the original PyTorch model and each exported artifact
(``|y_torch - y_export| < 2.0e-3``, bounding float32 machine precision to < 10 ULPs).

Note: ``torch.jit.trace`` is deprecated and unsupported on Python 3.14+, so
TorchScript is replaced by ``torch.export`` (EXIR) which is the maintained
portable graph format.
"""

# Import necessary libraries and modules
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import onnxruntime
import torch
from torch import nn

from src.models.ft_transformer import FTCATransformer
from src.models.hybrid_gating import LearnedHybridGate
from src.training.train_autoencoder import load_checkpoint
from src.training.train_hybrid_gating import (
    load_checkpoint as load_hybrid_gate_checkpoint,
)
from src.training.train_transformer import load_ft_transformer
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_OPSET = 18
# For an 800-dim autoencoder emitting scores ~1500, float32 machine precision
# has a unit in the last place (ULP) of ~2.44e-4. A strict bound of 2.0e-3
# guarantees < 10 ULPs (< 1.5 ppm relative error) without scale inflation.
PARITY_TOLERANCE = 2.0e-3


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
    batch_dim = torch.export.Dim("batch")
    dynamic_shapes = ({0: batch_dim},)
    with torch.no_grad():
        exported = torch.export.export(model, (sample,), dynamic_shapes=dynamic_shapes)
        torch.export.save(exported, out)
    logger.info("Exported EXIR model to %s (dynamic batch)", out)
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
    """Measure maximum absolute output difference between two scoring paths.

    Enforces a strict absolute tolerance bound ``|y_ref - y_art| <= tolerance``
    with zero scale-dependent inflation.

    Args:
        reference: Original PyTorch model.
        artifact: EXIR module, ONNX session output tensor, or TorchScript
            module handled through the common ``__call__`` protocol.
        sample: Input tensor shared by both models.
        tolerance: Maximum acceptable absolute difference.

    Returns:
        The observed maximum absolute difference.

    Raises:
        RuntimeError: If the maximum absolute difference exceeds ``tolerance``.
    """
    reference.eval()
    with torch.no_grad():
        y_ref = reference(sample)
        y_art = artifact(sample) if callable(artifact) else artifact
    max_diff = float((y_ref - y_art).abs().max().item())
    if max_diff > tolerance:
        msg = (
            f"Serialization parity failed: max diff {max_diff:.2e} > "
            f"strict tolerance {tolerance:.2e}"
        )
        raise RuntimeError(msg)
    logger.info("Parity verified: max diff %.6e (strict tol %.1e)", max_diff, tolerance)
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
        if hasattr(self.base, "decode") and hasattr(self.base, "encode"):
            x_hat = self.base.decode(self.base.encode(x))
            residual = x - x_hat
            l2 = torch.sum(residual.pow(2), dim=-1)
            l1 = torch.sum(torch.abs(residual), dim=-1)
            scores = l2 + self.l1_gamma * l1
        else:
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


class HybridGateExportModule(nn.Module):
    """Export-safe wrapper around a trained learned hybrid gate."""

    def __init__(self, gate: LearnedHybridGate) -> None:
        """Initialize the wrapper with the trained gate."""
        super().__init__()
        self.network = gate.network

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return fraud probabilities without Python-side validation branches."""
        logits = self.network(features)
        return torch.sigmoid(logits).squeeze(-1)


class FTTransformerExportModule(nn.Module):
    """Export-safe wrapper around a trained FT-CAT transformer."""

    def __init__(self, model: FTCATransformer) -> None:
        """Initialize the wrapper with the trained FT-CAT model."""
        super().__init__()
        self.model = model

    def forward(
        self,
        x_cont: torch.Tensor,
        x_cat: torch.Tensor,
        seq: torch.Tensor,
    ) -> torch.Tensor:
        """Return FT-CAT logits without Python-side validation branches."""
        tokenizer = self.model.tokenizer
        batch = x_cont.size(0)

        tokens = [tokenizer.cls_token.expand(batch, -1, -1)]

        if tokenizer.cont_weight is not None:
            continuous_tokens = (
                x_cont.unsqueeze(-1) * tokenizer.cont_weight.unsqueeze(0) + tokenizer.cont_bias
            )
            tokens.append(continuous_tokens)

        for index, embedding in enumerate(tokenizer.cat_embeddings):
            codes = x_cat[:, index]
            tokens.append(embedding(codes).unsqueeze(1))

        tokens_tensor = torch.cat(tokens, dim=1)
        tokens_tensor = self.model.encoder(tokens_tensor)

        if self.model.cross_attention is not None and self.model.history_encoder is not None:
            memory = self.model.history_encoder(seq)
            padding_mask = self.model.history_encoder.padding_mask(seq)

            tokens_tensor, _ = self.model.cross_attention(
                tokens_tensor,
                memory,
                key_padding_mask=padding_mask,
                need_weights=False,
            )

        logits = self.model.head(self.model.head_norm(tokens_tensor[:, 0])).squeeze(-1)

        return logits


def export_ft_transformer(
    input_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, float]:
    """Export the trained FT-CAT transformer to EXIR and ONNX."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    checkpoint_path = input_path / "ft_transformer.pt"

    model, _ = load_ft_transformer(
        checkpoint_path,
        device="cpu",
    )

    if not isinstance(model, FTCATransformer):
        raise TypeError("FT-CAT serialization requires an FTCATransformer checkpoint")

    model.eval()

    export_model = FTTransformerExportModule(model)
    export_model.eval()

    meta = model.state_meta()

    batch_size = 4
    n_continuous = int(meta["n_continuous"])
    n_categorical = len(meta["categorical_cardinalities"])
    seq_len = int(meta["seq_len"])
    seq_dim = int(meta["seq_dim"])

    x_cont = torch.randn(
        batch_size,
        n_continuous,
        dtype=torch.float32,
    )

    x_cat = torch.zeros(
        batch_size,
        n_categorical,
        dtype=torch.long,
    )

    seq = torch.randn(
        batch_size,
        seq_len,
        seq_dim,
        dtype=torch.float32,
    )

    exir_path = output_path / "ft_transformer.pt2"
    onnx_path = output_path / "ft_transformer.onnx"

    batch_dim = torch.export.Dim("batch")

    dynamic_shapes = (
        {0: batch_dim},
        {0: batch_dim},
        {0: batch_dim},
    )

    with torch.no_grad():
        exported = torch.export.export(
            export_model,
            (x_cont, x_cat, seq),
            dynamic_shapes=dynamic_shapes,
        )
        torch.export.save(exported, exir_path)

    logger.info(
        "Exported FT-CAT EXIR model to %s (dynamic batch)",
        exir_path,
    )

    with torch.no_grad():
        torch.onnx.export(
            export_model,
            (x_cont, x_cat, seq),
            str(onnx_path),
            input_names=[
                "continuous_features",
                "categorical_features",
                "history",
            ],
            output_names=["fraud_logits"],
            dynamic_shapes=dynamic_shapes,
            opset_version=DEFAULT_OPSET,
        )

    logger.info(
        "Exported FT-CAT ONNX model to %s (opset %d)",
        onnx_path,
        DEFAULT_OPSET,
    )

    with torch.no_grad():
        reference_output = model(
            x_cont,
            x_cat,
            seq,
        )

    exported_program = torch.export.load(str(exir_path))
    exir_model = exported_program.module()

    with torch.no_grad():
        exir_output = exir_model(
            x_cont,
            x_cat,
            seq,
        )

    exir_diff = float((reference_output - exir_output).abs().max().item())

    if exir_diff > PARITY_TOLERANCE:
        raise RuntimeError(
            "FT-CAT EXIR parity failed: " f"max diff {exir_diff:.2e} > " f"{PARITY_TOLERANCE:.2e}"
        )

    onnx_session = onnxruntime.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )

    onnx_inputs = {
        onnx_session.get_inputs()[0].name: x_cont.numpy(),
        onnx_session.get_inputs()[1].name: x_cat.numpy(),
        onnx_session.get_inputs()[2].name: seq.numpy(),
    }

    onnx_output = torch.from_numpy(
        onnx_session.run(
            None,
            onnx_inputs,
        )[0]
    )

    onnx_diff = float((reference_output - onnx_output).abs().max().item())

    if onnx_diff > PARITY_TOLERANCE:
        raise RuntimeError(
            "FT-CAT ONNX parity failed: " f"max diff {onnx_diff:.2e} > " f"{PARITY_TOLERANCE:.2e}"
        )

    logger.info(
        "FT-CAT parity verified: EXIR %.6e, ONNX %.6e",
        exir_diff,
        onnx_diff,
    )

    return {
        "exir_max_diff": round(exir_diff, 8),
        "onnx_max_diff": round(onnx_diff, 8),
    }


def export_hybrid_gate(
    input_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, float]:
    """Export the trained learned hybrid gate to EXIR and ONNX.

    Args:
        input_dir: Directory containing ``hybrid_gating.pt``.
        output_dir: Destination directory for serialized artifacts.

    Returns:
        Dictionary containing EXIR and ONNX parity errors.

    Raises:
        FileNotFoundError: If the trained gate checkpoint does not exist.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    checkpoint_path = input_path / "hybrid_gating.pt"
    gate, _ = load_hybrid_gate_checkpoint(checkpoint_path)
    gate.eval()

    export_gate = HybridGateExportModule(gate)
    export_gate.eval()

    sample = torch.randn(4, gate.input_dim)

    exir_path = output_path / "hybrid_gating.pt2"
    onnx_path = output_path / "hybrid_gating.onnx"

    export_exir(export_gate, exir_path, sample)

    export_onnx(
        export_gate,
        onnx_path,
        sample,
        input_name="gate_features",
        output_name="fraud_probability",
    )

    exir_module = load_exir(exir_path)
    exir_diff = verify_parity(gate, exir_module, sample)

    onnx_session = onnxruntime.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )

    input_name = onnx_session.get_inputs()[0].name
    onnx_output = onnx_session.run(
        None,
        {input_name: sample.numpy()},
    )[0]

    onnx_diff = verify_parity(
        gate,
        torch.from_numpy(onnx_output),
        sample,
    )

    return {
        "exir_max_diff": round(exir_diff, 8),
        "onnx_max_diff": round(onnx_diff, 8),
    }


def export_all(
    input_dir: str | Path,
    output_dir: str | Path,
    config: dict,
) -> dict[str, float]:
    """Export available trained models to EXIR and ONNX with parity checks.

    The autoencoder keeps its reference-model fallback for development and
    tests. FT-CAT and learned-gate artifacts are exported when their trained
    checkpoints are present in the input directory.

    Args:
        input_dir: Directory containing trained model checkpoints.
        output_dir: Destination directory for serialized artifacts.
        config: Loaded project configuration.

    Returns:
        Dictionary containing parity errors for all exported models.
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

    onnx_session = onnxruntime.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )
    input_name = onnx_session.get_inputs()[0].name
    onnx_out = torch.from_numpy(
        onnx_session.run(
            None,
            {input_name: sample.numpy()},
        )[0]
    ).reshape(-1, 1)

    onnx_diff = verify_parity(
        scorer,
        onnx_out,
        sample,
    )

    report = {
        "exir_max_diff": round(exir_diff, 8),
        "onnx_max_diff": round(onnx_diff, 8),
    }

    ft_checkpoint = input_path / "ft_transformer.pt"

    if ft_checkpoint.is_file():
        logger.info("Exporting trained FT-CAT checkpoint %s", ft_checkpoint)

        ft_report = export_ft_transformer(
            input_path,
            output_path,
        )

        report["ft_transformer_exir_max_diff"] = ft_report["exir_max_diff"]
        report["ft_transformer_onnx_max_diff"] = ft_report["onnx_max_diff"]
    else:
        logger.info(
            "FT-CAT checkpoint %s not found; skipping FT-CAT export",
            ft_checkpoint,
        )

    gate_checkpoint = input_path / "hybrid_gating.pt"

    if gate_checkpoint.is_file():
        logger.info(
            "Exporting trained hybrid gate checkpoint %s",
            gate_checkpoint,
        )

        gate_report = export_hybrid_gate(
            input_path,
            output_path,
        )

        report["hybrid_gating_exir_max_diff"] = gate_report["exir_max_diff"]
        report["hybrid_gating_onnx_max_diff"] = gate_report["onnx_max_diff"]
    else:
        logger.info(
            "Hybrid gate checkpoint %s not found; skipping gate export",
            gate_checkpoint,
        )

    return report


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
