"""Unit tests for EXIR/ONNX serialization and numerical parity."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from src.models.ft_transformer import FTCATransformer
from src.serving.model_serializer import (
    FTTransformerExportModule,
    HybridGateExportModule,
    ReferenceEncoder,
    build_reference_model,
    export_exir,
    export_ft_transformer,
    export_hybrid_gate,
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
    # Toy reference model (10-dim) easily holds 1e-4; prod 800-dim uses PARITY_TOLERANCE (2e-3).
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
    # Toy model has tiny FMA drift (< 1e-4); prod 800-dim uses PARITY_TOLERANCE.
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


def test_export_exir_dynamic_batch(ref_model: ReferenceEncoder, tmp_path: Path) -> None:
    """EXIR graph supports arbitrary batch sizes at runtime."""
    sample = torch.randn(4, 20)
    path = export_exir(ref_model, tmp_path / "dyn.pt2", sample)
    loaded = load_exir(path)
    out1 = loaded(torch.ones(1, 20))
    out9 = loaded(torch.ones(9, 20))
    assert out1.shape == (1, 1)
    assert out9.shape == (9, 1)


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
    # 8-dim fallback stays < 1e-4; prod 800-dim uses PARITY_TOLERANCE
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
    with torch.no_grad():
        scores = ref_model.anomaly_score(sample_batch)
        mean_score = ref_model.anomaly_score(sample_batch, reduction="mean")
        sum_score = ref_model.anomaly_score(sample_batch, reduction="sum")
    assert float(mean_score) == pytest.approx(float(scores.mean()))
    assert float(sum_score) == pytest.approx(float(scores.sum()))


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
    assert torch.allclose(resolved.anomaly_score(sample), model.anomaly_score(sample), atol=1e-6)


def test_hybrid_gate_export_module_matches_original() -> None:
    """Export-safe gate wrapper preserves learned-gate probabilities."""
    from src.models.hybrid_gating import LearnedHybridGate

    torch.manual_seed(42)

    gate = LearnedHybridGate(
        input_dim=4,
        hidden_dims=[16, 8],
        dropout=0.1,
    )
    gate.eval()

    wrapper = HybridGateExportModule(gate)
    wrapper.eval()

    sample = torch.randn(8, 4)

    with torch.no_grad():
        expected = gate(sample)
        actual = wrapper(sample)

    assert torch.allclose(expected, actual, atol=1e-7)


def test_export_hybrid_gate_realistic_checkpoint(tmp_path: Path) -> None:
    """Learned-gate checkpoint exports to EXIR and ONNX with numerical parity."""
    from src.models.hybrid_gating import LearnedHybridGate, PercentileNormalizer
    from src.training.train_hybrid_gating import save_checkpoint

    torch.manual_seed(42)

    gate = LearnedHybridGate(
        input_dim=4,
        hidden_dims=[16, 8],
        dropout=0.1,
    )
    gate.eval()

    normalizer = PercentileNormalizer()
    normalizer.fit(torch.tensor([1.0, 2.0, 3.0, 4.0]))

    checkpoint_dir = tmp_path / "checkpoints"
    output_dir = tmp_path / "serialized"
    checkpoint_dir.mkdir()

    save_checkpoint(
        model=gate,
        normalizer=normalizer,
        path=checkpoint_dir / "hybrid_gating.pt",
    )

    report = export_hybrid_gate(
        checkpoint_dir,
        output_dir,
    )

    assert (output_dir / "hybrid_gating.pt2").is_file()
    assert (output_dir / "hybrid_gating.onnx").is_file()

    assert report["exir_max_diff"] <= 2.0e-3
    assert report["onnx_max_diff"] <= 2.0e-3


def test_ft_transformer_export_wrapper_matches_original() -> None:
    """Export-safe FT-CAT wrapper preserves original model logits."""
    torch.manual_seed(42)

    model = FTCATransformer(
        n_continuous=3,
        categorical_cardinalities=[4, 5],
        seq_len=5,
        seq_dim=2,
        d_model=8,
        n_heads=2,
        dim_feedforward=16,
        n_layers=1,
        dropout=0.0,
        use_cross_attention=True,
    )
    model.eval()

    wrapper = FTTransformerExportModule(model)
    wrapper.eval()

    x_cont = torch.randn(4, 3)
    x_cat = torch.tensor(
        [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 4],
        ],
        dtype=torch.long,
    )
    seq = torch.randn(4, 5, 2)

    with torch.no_grad():
        expected = model(x_cont, x_cat, seq)
        actual = wrapper(x_cont, x_cat, seq)

    assert torch.allclose(expected, actual, atol=1e-7)


def _build_small_ft_transformer() -> FTCATransformer:
    """Build a compact deterministic FT-CAT model for serialization tests."""
    torch.manual_seed(42)

    model = FTCATransformer(
        n_continuous=3,
        categorical_cardinalities=[4, 5],
        seq_len=5,
        seq_dim=2,
        d_model=8,
        n_heads=2,
        dim_feedforward=16,
        n_layers=1,
        dropout=0.0,
        use_cross_attention=True,
    )
    model.eval()
    return model


def _ft_transformer_sample() -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Return valid FT-CAT inputs with realistic tensor dtypes."""
    x_cont = torch.randn(4, 3)

    x_cat = torch.tensor(
        [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 4],
        ],
        dtype=torch.long,
    )

    seq = torch.randn(4, 5, 2)

    return x_cont, x_cat, seq


def test_ft_transformer_exir_parity(tmp_path: Path) -> None:
    """FT-CAT EXIR output matches the original PyTorch model."""
    model = _build_small_ft_transformer()
    wrapper = FTTransformerExportModule(model)
    wrapper.eval()

    x_cont, x_cat, seq = _ft_transformer_sample()

    batch_dim = torch.export.Dim("batch")
    dynamic_shapes = (
        {0: batch_dim},
        {0: batch_dim},
        {0: batch_dim},
    )

    path = tmp_path / "ft_transformer.pt2"

    with torch.no_grad():
        exported = torch.export.export(
            wrapper,
            (x_cont, x_cat, seq),
            dynamic_shapes=dynamic_shapes,
        )
        torch.export.save(exported, path)

    assert path.is_file()

    loaded = torch.export.load(str(path)).module()

    with torch.no_grad():
        expected = model(x_cont, x_cat, seq)
        actual = loaded(x_cont, x_cat, seq)

    assert torch.allclose(
        expected,
        actual,
        atol=2.0e-3,
        rtol=0.0,
    )


def test_ft_transformer_onnx_parity(tmp_path: Path) -> None:
    """FT-CAT ONNX output matches the original PyTorch model."""
    onnxruntime = pytest.importorskip("onnxruntime")

    model = _build_small_ft_transformer()
    wrapper = FTTransformerExportModule(model)
    wrapper.eval()

    x_cont, x_cat, seq = _ft_transformer_sample()

    path = tmp_path / "ft_transformer.onnx"

    batch_dim = torch.export.Dim("batch")
    dynamic_shapes = (
        {0: batch_dim},
        {0: batch_dim},
        {0: batch_dim},
    )

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (x_cont, x_cat, seq),
            str(path),
            input_names=[
                "continuous_features",
                "categorical_features",
                "history",
            ],
            output_names=["fraud_logits"],
            dynamic_shapes=dynamic_shapes,
            opset_version=18,
        )

    assert path.is_file()

    session = onnxruntime.InferenceSession(
        str(path),
        providers=["CPUExecutionProvider"],
    )

    inputs = {
        session.get_inputs()[0].name: x_cont.numpy(),
        session.get_inputs()[1].name: x_cat.numpy(),
        session.get_inputs()[2].name: seq.numpy(),
    }

    actual = session.run(None, inputs)[0]

    with torch.no_grad():
        expected = model(x_cont, x_cat, seq).numpy()

    assert np.abs(expected - actual).max() <= 2.0e-3


def test_ft_transformer_exir_dynamic_batch(tmp_path: Path) -> None:
    """FT-CAT EXIR graph accepts batch sizes different from the export sample."""
    model = _build_small_ft_transformer()
    wrapper = FTTransformerExportModule(model)
    wrapper.eval()

    x_cont, x_cat, seq = _ft_transformer_sample()

    batch_dim = torch.export.Dim("batch")
    dynamic_shapes = (
        {0: batch_dim},
        {0: batch_dim},
        {0: batch_dim},
    )

    path = tmp_path / "ft_dynamic.pt2"

    with torch.no_grad():
        exported = torch.export.export(
            wrapper,
            (x_cont, x_cat, seq),
            dynamic_shapes=dynamic_shapes,
        )
        torch.export.save(exported, path)

    loaded = torch.export.load(str(path)).module()

    x_cont_large = torch.randn(9, 3)
    x_cat_large = torch.zeros(9, 2, dtype=torch.long)
    seq_large = torch.randn(9, 5, 2)

    with torch.no_grad():
        output = loaded(
            x_cont_large,
            x_cat_large,
            seq_large,
        )

    assert output.shape == (9,)


def test_export_ft_transformer_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FT-CAT checkpoint export creates EXIR and ONNX artifacts."""
    from src.serving import model_serializer

    model = _build_small_ft_transformer()

    checkpoint_dir = tmp_path / "checkpoints"
    output_dir = tmp_path / "serialized"
    checkpoint_dir.mkdir()

    checkpoint_path = checkpoint_dir / "ft_transformer.pt"
    checkpoint_path.touch()

    payload = {
        "feature_spec": {
            "continuous_cols": ["a", "b", "c"],
            "categorical_cols": ["cat1", "cat2"],
            "categorical_cardinalities": [4, 5],
            "sequence_cols": ["s1", "s2"],
            "seq_len": 5,
        }
    }

    def fake_load_ft_transformer(
        path: str | Path,
        device: str = "cpu",
    ) -> tuple[FTCATransformer, dict]:
        assert Path(path) == checkpoint_path
        assert device == "cpu"
        return model, payload

    monkeypatch.setattr(
        model_serializer,
        "load_ft_transformer",
        fake_load_ft_transformer,
    )

    report = export_ft_transformer(
        checkpoint_dir,
        output_dir,
    )

    assert (output_dir / "ft_transformer.pt2").is_file()
    assert (output_dir / "ft_transformer.onnx").is_file()

    assert report["exir_max_diff"] <= 2.0e-3
    assert report["onnx_max_diff"] <= 2.0e-3
