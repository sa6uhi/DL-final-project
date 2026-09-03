"""Unit tests for EXIR/ONNX serialization and numerical parity."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from src.serving.model_serializer import (
    ReferenceEncoder,
    build_reference_model,
    export_exir,
    export_onnx,
    load_exir,
    verify_parity,
)

torch.manual_seed(42)


@pytest.fixture()
def ref_model() -> ReferenceEncoder:
    """Deterministic reference model in evaluation mode."""
    return build_reference_model(input_dim=20)


@pytest.fixture()
def sample_batch() -> torch.Tensor:
    """Fixed sample tensor for export."""
    return torch.from_numpy(np.random.default_rng(7).standard_normal((4, 20)).astype(np.float32))


def test_reference_model_output_shape(
    ref_model: ReferenceEncoder, sample_batch: torch.Tensor
) -> None:
    """Reference scores have shape (batch, 1) and are positive."""
    out = ref_model(sample_batch)
    assert tuple(out.shape) == (4, 1)
    assert (out >= 0).all()


def test_reference_model_meta_roundtrip(ref_model: ReferenceEncoder) -> None:
    """state_meta rebuilds a config-equivalent model."""
    rebuilt = ReferenceEncoder.build_from_state(ref_model.state_meta())
    assert rebuilt.input_dim == ref_model.input_dim
    assert sum(p.numel() for p in rebuilt.parameters()) == sum(
        p.numel() for p in ref_model.parameters()
    )


def test_export_exir_parity(
    ref_model: ReferenceEncoder, sample_batch: torch.Tensor, tmp_path: Path
) -> None:
    """EXIR export keeps outputs within 1e-4 of the original."""
    path = export_exir(ref_model, tmp_path / "ref.pt2", sample_batch)
    assert path.is_file()
    loaded = load_exir(path)
    assert torch.allclose(ref_model(sample_batch), loaded(sample_batch), atol=1e-4)
    verify_parity(ref_model, loaded, sample_batch)


def test_export_exir_training_model_raises(sample_batch: torch.Tensor, tmp_path: Path) -> None:
    """Training-mode models are rejected before export."""
    model = ReferenceEncoder(20)
    with pytest.raises(ValueError):
        export_exir(model, tmp_path / "x.pt2", sample_batch)


def test_export_onnx_parity(
    ref_model: ReferenceEncoder, sample_batch: torch.Tensor, tmp_path: Path
) -> None:
    """ONNX exports run under onnxruntime within 1e-4 of PyTorch."""
    onnxruntime = pytest.importorskip("onnxruntime")
    path = export_onnx(ref_model, tmp_path / "ref.onnx", sample_batch)
    assert path.is_file()
    session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    ort_input = {session.get_inputs()[0].name: sample_batch.numpy()}
    ort_output = session.run(None, ort_input)[0]
    y_ref = ref_model(sample_batch).detach().numpy()
    assert np.abs(y_ref - ort_output).max() < 1e-4
    verify_parity(ref_model, torch.from_numpy(ort_output), sample_batch)


def test_export_onnx_training_model_raises(sample_batch: torch.Tensor, tmp_path: Path) -> None:
    """Training-mode models are rejected before ONNX export."""
    model = ReferenceEncoder(20)
    with pytest.raises(ValueError):
        export_onnx(model, tmp_path / "x.onnx", sample_batch)


def test_export_onnx_dynamic_batch(ref_model: ReferenceEncoder, tmp_path: Path) -> None:
    """ONNX graph supports arbitrary batch sizes at runtime."""
    onnxruntime = pytest.importorskip("onnxruntime")
    sample = torch.randn(2, 20)
    path = export_onnx(ref_model, tmp_path / "dyn.onnx", sample)
    session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    out1 = session.run(None, {session.get_inputs()[0].name: np.ones((1, 20), dtype=np.float32)})[0]
    out2 = session.run(None, {session.get_inputs()[0].name: np.ones((9, 20), dtype=np.float32)})[0]
    assert out1.shape == (1, 1)
    assert out2.shape == (9, 1)


def test_verify_parity_detects_mismatch(
    ref_model: ReferenceEncoder, sample_batch: torch.Tensor
) -> None:
    """Parity check raises when outputs diverge beyond tolerance."""
    tampered = ReferenceEncoder(20)
    tampered.eval()
    for param, other in zip(tampered.parameters(), ref_model.parameters()):
        param.data.copy_(other.data + 0.01)
    with pytest.raises(RuntimeError):
        verify_parity(ref_model, tampered, sample_batch)


def test_load_exir_missing_raises(tmp_path: Path) -> None:
    """Loading a nonexistent EXIR artifact raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_exir(tmp_path / "missing.pt2")


def test_export_all_falls_back_to_reference(tmp_path: Path) -> None:
    """Missing checkpoints fall back to the reference scorer and still export."""
    from src.serving.model_serializer import export_all

    config = {"autoencoder": {"input_dim": 8}}
    report = export_all(tmp_path / "nope", tmp_path / "out", config)
    assert (tmp_path / "out" / "autoencoder.pt2").exists()
    assert (tmp_path / "out" / "autoencoder.onnx").exists()
    assert report["exir_max_diff"] < 1e-4
    assert report["onnx_max_diff"] < 1e-4


def test_export_all_cli_main(tmp_path: Path, monkeypatch) -> None:
    """CLI entry point wires config + dirs to export_all."""
    from src.serving.model_serializer import main

    out_dir = tmp_path / "out"
    main(
        [
            "--input",
            str(tmp_path / "nope"),
            "--output",
            str(out_dir),
            "--config",
            str(Path("config") / "config.yaml"),
        ]
    )
    assert (out_dir / "autoencoder.pt2").exists()
    assert (out_dir / "autoencoder.onnx").exists()


def test_reference_anomaly_score_reductions(
    ref_model: ReferenceEncoder, sample_batch: torch.Tensor
) -> None:
    """mean/sum reductions aggregate the per-sample reference scores."""
    scores = ref_model.anomaly_score(sample_batch)
    assert float(ref_model.anomaly_score(sample_batch, reduction="mean")) == pytest.approx(
        float(scores.mean())
    )
    assert float(ref_model.anomaly_score(sample_batch, reduction="sum")) == pytest.approx(
        float(scores.sum())
    )


def test_resolve_model_prefers_trained_checkpoint(tmp_path: Path) -> None:
    """_resolve_model loads the trained DAE when autoencoder.pt is present."""
    from src.models.autoencoder import DenoisingAutoencoder
    from src.serving.model_serializer import _resolve_model
    from src.training.train_autoencoder import save_checkpoint

    torch.manual_seed(0)
    model = DenoisingAutoencoder(input_dim=8, encoder_hidden_dims=[4], latent_dim=2)
    model.eval()
    input_dir = tmp_path / "checkpoints"
    input_dir.mkdir()
    save_checkpoint(model, input_dir / "autoencoder.pt")

    resolved, source = _resolve_model(input_dir, {"autoencoder": {"input_dim": 8}})

    assert source == input_dir / "autoencoder.pt"
    sample = torch.randn(2, 8)
    assert torch.allclose(
        resolved.anomaly_score(sample), model.anomaly_score(sample), atol=1e-6
    )
