"""Unit tests for Exp-5 latency benchmark and Exp-6 anomaly evaluation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from src.evaluation.dae_anomaly_eval import evaluate_split, load_npz, sweep_alpha, write_eval_report
from src.evaluation.latency_benchmark import (
    LatencyStats,
    benchmark_backends,
    measure_latency,
    write_latency_report,
)


@pytest.fixture()
def simple_fn() -> torch.nn.Module:
    """Deterministic scoring module for latency measurements."""
    return torch.nn.Sequential(torch.nn.Linear(8, 1), torch.nn.Softplus())


def test_measure_latency_positive_percentiles(simple_fn: torch.nn.Module) -> None:
    """Latencies are positive and percentiles are monotonic."""
    stats = measure_latency(simple_fn, torch.randn(4, 8), n_warmup=2, n_runs=30)
    assert isinstance(stats, LatencyStats)
    assert stats.p50_ms >= 0.0 and stats.p95_ms >= stats.p50_ms
    assert stats.p99_ms >= stats.p95_ms
    assert stats.throughput_req_s > 0.0
    assert stats.batch_size == 4 and stats.n_runs == 30


def test_measure_latency_rejects_bad_runs(simple_fn: torch.nn.Module) -> None:
    """Non-positive run counts raise ValueError."""
    with pytest.raises(ValueError):
        measure_latency(simple_fn, torch.randn(2, 8), n_runs=0)


def test_measure_latency_rejects_wrong_batch(simple_fn: torch.nn.Module) -> None:
    """A callable returning wrong batch cardinality raises ValueError."""
    with pytest.raises(ValueError):
        measure_latency(lambda x: torch.ones(1, 1), torch.randn(4, 8), n_warmup=0, n_runs=2)


def test_measure_latency_numpy_output() -> None:
    """measure_latency accepts callable returning numpy arrays."""

    def np_fn(x: torch.Tensor) -> np.ndarray:
        return np.ones((x.shape[0], 1), dtype=np.float32)

    stats = measure_latency(np_fn, torch.randn(4, 8), n_warmup=1, n_runs=5, backend="onnx")
    assert stats.backend == "onnx"
    assert stats.batch_size == 4


def test_measure_latency_rejects_wrong_numpy_batch() -> None:
    """Callable returning wrong numpy batch size raises ValueError."""

    def np_wrong(x: torch.Tensor) -> np.ndarray:
        return np.ones((1, 1), dtype=np.float32)

    with pytest.raises(ValueError):
        measure_latency(np_wrong, torch.randn(4, 8), n_warmup=0, n_runs=2)


def test_benchmark_backends_multiple_batch_sizes() -> None:
    """Sweeping several batch sizes returns one record each."""
    module = torch.nn.Sequential(torch.nn.Linear(6, 1))
    module.eval()
    stats = benchmark_backends(module, batch_sizes=[1, 4], input_dim=6, n_warmup=1, n_runs=5)
    assert [s.batch_size for s in stats] == [1, 4]


def test_write_latency_report(tmp_path: Path) -> None:
    """JSON + CSV are persisted with all records."""
    stats = benchmark_backends(
        torch.nn.Linear(4, 1), batch_sizes=[1], input_dim=4, n_warmup=1, n_runs=3
    )
    csv_path = write_latency_report(stats, tmp_path)
    assert csv_path.exists()
    assert (tmp_path / "latency_summary.json").exists()
    assert (
        csv_path.read_text().splitlines()[0]
        == "backend,batch_size,n_runs,p50_ms,p95_ms,p99_ms,mean_ms,throughput_req_s"
    )


def test_load_npz_roundtrip(tmp_path: Path) -> None:
    """npz archives load into flat dicts with the original keys."""
    path = tmp_path / "scores.npz"
    np.savez(path, calibration_scores=np.array([1.0, 2.0]), labels=np.array([0, 1]))
    bundle = load_npz(path)
    assert set(bundle) == {"calibration_scores", "labels"}


def test_load_npz_missing_file(tmp_path: Path) -> None:
    """Missing archives raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_npz(tmp_path / "nope.npz")


def test_evaluate_split_separable() -> None:
    """A perfectly separable anomaly split reports AUC = 1.0."""
    scores = np.array([0.1, 0.2, 0.3, 5.0, 6.0])
    labels = np.array([0, 0, 0, 1, 1])
    metrics = evaluate_split(scores, labels, max_fpr=0.01)
    assert metrics["rocauc"] == pytest.approx(1.0)
    assert metrics["prevalence"] == pytest.approx(0.4)


def test_sweep_alpha_ends_are_pure_sources() -> None:
    """alpha=1 follows the anomaly residual; alpha=0 the transformer posterior."""
    from src.evaluation.metrics import roc_auc

    labels = np.array([0, 0, 0, 1, 1, 1, 1, 0])
    calib = np.array([0.1, 0.2, 0.15, 0.19, 0.11, 0.2, 0.18, 0.12])
    eval_scores = np.array([0.1, 0.2, 0.3, 8.0, 9.0, 9.5, 9.9, 0.3])
    probs = np.array([0.05, 0.06, 0.07, 0.95, 0.97, 0.4, 0.9, 0.04])
    records = sweep_alpha(
        calibration_scores=calib,
        eval_scores=eval_scores,
        probabilities_ft=probs,
        labels=labels,
        alphas=[0.0, 1.0],
        percentile=99.0,
    )
    assert [r["alpha"] for r in records] == [0.0, 1.0]
    assert all(0.0 <= r["rocauc"] <= 1.0 for r in records)
    assert records[0]["rocauc"] == pytest.approx(roc_auc(probs, labels))
    assert records[1]["rocauc"] >= 0.5


def test_sweep_alpha_ignores_calibration_leak() -> None:
    """The normalizer cap must be derived from calibration, not eval scores."""
    from src.models.hybrid_gating import PercentileNormalizer

    calib = np.array([1.0, 1.1, 0.9, 0.8])
    normalizer = PercentileNormalizer(percentile=99.0).fit(
        torch.as_tensor(calib, dtype=torch.float32)
    )
    assert float(normalizer.cap) <= float(np.max(calib)) + 1e-6


def test_write_eval_report(tmp_path: Path) -> None:
    """Anomaly eval report writes JSON and CSV artifacts."""
    split_metrics = {"rocauc": 0.9, "auprc": 0.8, "tpr_at_fpr": 0.7, "prevalence": 0.1}
    sweep = [{"alpha": 0.0, "rocauc": 0.9}, {"alpha": 1.0, "rocauc": 1.0}]
    csv_path = write_eval_report(split_metrics, sweep, tmp_path)
    assert csv_path.exists()
    assert (tmp_path / "anomaly_eval.json").exists()


def _tiny_eval_config(tmp_path: Path, payload: dict) -> str:
    """Write a minimal eval yaml config and return its path."""
    import yaml

    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return str(config_file)


def test_latency_main_writes_report(tmp_path: Path) -> None:
    """latency main() benchmarks the reference model and writes artifacts."""
    from src.evaluation.latency_benchmark import main

    config_path = _tiny_eval_config(
        tmp_path,
        {
            "evaluation": {"latency": {"batch_sizes": [1], "n_warmup": 1, "n_runs": 2}},
            "autoencoder": {"input_dim": 8},
            "paths": {"experiments": str(tmp_path)},
        },
    )
    out_dir = tmp_path / "latency_out"
    main(["--config", config_path, "--output-dir", str(out_dir)])
    assert (out_dir / "latency_summary.json").exists()
    assert (out_dir / "latency_table.csv").exists()


def test_latency_main_with_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """latency main() benchmarks trained checkpoint when present."""
    from src.evaluation.latency_benchmark import main
    from src.models.autoencoder import DenoisingAutoencoder
    from src.training.train_autoencoder import save_checkpoint

    model = DenoisingAutoencoder(input_dim=8, encoder_hidden_dims=[4], latent_dim=2)
    model.eval()
    ckpt_dir = tmp_path / "models/checkpoints"
    ckpt_dir.mkdir(parents=True)
    save_checkpoint(model, ckpt_dir / "autoencoder.pt")

    config_path = _tiny_eval_config(
        tmp_path,
        {
            "evaluation": {"latency": {"batch_sizes": [1], "n_warmup": 1, "n_runs": 2}},
            "autoencoder": {"input_dim": 8},
            "paths": {"experiments": str(tmp_path)},
        },
    )
    monkeypatch.chdir(tmp_path)
    out_dir = tmp_path / "latency_out_ckpt"
    main(["--config", config_path, "--output-dir", str(out_dir)])
    assert (out_dir / "latency_summary.json").exists()


def test_anomaly_eval_main_writes_report(tmp_path: Path) -> None:
    """anomaly main() scores a tiny npz archive and writes artifacts."""
    from src.evaluation.dae_anomaly_eval import main

    archive = tmp_path / "scores.npz"
    np.savez(
        archive,
        calibration_scores=np.array([0.1, 0.2, 0.15, 0.3]),
        eval_scores=np.array([0.1, 0.2, 5.0, 6.0, 0.3, 7.0]),
        probabilities_ft=np.array([0.05, 0.06, 0.9, 0.95, 0.07, 0.92]),
        labels=np.array([0, 0, 1, 1, 0, 1]),
    )
    config_path = _tiny_eval_config(
        tmp_path,
        {
            "evaluation": {"anomaly_eval": {"max_fpr": 0.5, "alpha_sweep": [0.0, 1.0]}},
            "autoencoder": {"anomaly_score": {"normalize_percentile": 99.0}},
            "paths": {"experiments": str(tmp_path)},
        },
    )
    out_dir = tmp_path / "anomaly_out"
    main(["--config", config_path, "--archive", str(archive), "--output-dir", str(out_dir)])
    assert (out_dir / "anomaly_eval.json").exists()
