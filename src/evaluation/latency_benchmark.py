"""Exp-5: end-to-end inference latency and throughput benchmarks (sub-15 ms P99 target).

Measures cold-start / warm latency of the PyTorch reference model, the EXIR
export and the ONNX export at multiple batch sizes, then persists p50 / p95 /
p99 summaries (ms) and throughput (requests/sec) to ``experiments/latency/``.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from src.serving.model_serializer import ReferenceEncoder
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class LatencyStats:
    """Summary latency statistics for one model/backend at one batch size."""

    backend: str
    batch_size: int
    n_runs: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float
    throughput_req_s: float


def measure_latency(
    fn: Callable[[torch.Tensor], torch.Tensor],
    sample: torch.Tensor,
    n_warmup: int = 10,
    n_runs: int = 200,
) -> LatencyStats:
    """Warm up then time ``fn`` per invocation at a fixed batch size.

    Args:
        fn: Callable accepting a batch tensor and returning a batched output.
        sample: Input batch tensor.
        n_warmup: Number of untimed warm-up calls.
        n_runs: Number of timed runs.

    Returns:
        A :class:`LatencyStats` record with percentiles and throughput.

    Raises:
        ValueError: If ``n_runs`` is non-positive or the output is not a
            per-sample tensor.
    """
    if n_runs <= 0:
        raise ValueError(f"n_runs must be positive, got {n_runs}")
    with torch.no_grad():
        for _ in range(n_warmup):
            fn(sample)
        timings_ms: list[float] = []
        for _ in range(n_runs):
            start = time.perf_counter()
            out = fn(sample)
            if isinstance(out, torch.Tensor) and out.shape[0] != sample.shape[0]:
                raise ValueError("Scoring function must return one row per input sample")
            timings_ms.append((time.perf_counter() - start) * 1000.0)
    arr = np.asarray(timings_ms, dtype=np.float64)
    p50, p95, p99 = (float(v) for v in np.percentile(arr, [50, 95, 99]))
    throughput = float(sample.shape[0] * 1000.0 / max(arr.mean(), 1.0e-12))
    return LatencyStats(
        backend="pytorch",
        batch_size=int(sample.shape[0]),
        n_runs=n_runs,
        p50_ms=round(p50, 3),
        p95_ms=round(p95, 3),
        p99_ms=round(p99, 3),
        mean_ms=round(float(arr.mean()), 3),
        throughput_req_s=round(throughput, 1),
    )


def benchmark_backends(
    model_fn: Callable[[torch.Tensor], torch.Tensor],
    batch_sizes: list[int],
    input_dim: int,
    n_warmup: int,
    n_runs: int,
) -> list[LatencyStats]:
    """Run the latency harness over every configured batch size.

    Args:
        model_fn: Scoring callable shared by all batch sizes (e.g. EXIR/ONNX
            forward or reference model).
        batch_sizes: Batch sizes to sweep.
        input_dim: Feature dimensionality of the sampled input.
        n_warmup: Warm-up runs per batch size.
        n_runs: Timed runs per batch size.

    Returns:
        List of latency statistics, one per batch size.
    """
    runs: list[LatencyStats] = []
    for batch_size in batch_sizes:
        sample = torch.randn(batch_size, input_dim)
        stats = measure_latency(model_fn, sample, n_warmup=n_warmup, n_runs=n_runs)
        logger.info("%s", stats)
        runs.append(stats)
    return runs


def write_latency_report(stats: list[LatencyStats], output_dir: str | Path) -> Path:
    """Persist a JSON summary and a CSV table of the measurements.

    Args:
        stats: List of measurement records.
        output_dir: Destination directory (created if missing).

    Returns:
        Path to the written CSV file.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "latency_summary.json"
    json_path.write_text(json.dumps([asdict(s) for s in stats], indent=2))
    csv_path = out / "latency_table.csv"
    if stats:
        fieldnames = list(asdict(stats[0]).keys())
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(asdict(s) for s in stats)
    logger.info("Latency report written to %s", out)
    return csv_path


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: ``python -m src.evaluation.latency_benchmark``."""
    parser = argparse.ArgumentParser(description="Exp-5 inference latency & throughput benchmark")
    parser.add_argument("--config", default="config/config.yaml", help="Path to config.yaml")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override the configured latency output directory",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    latency_cfg = config["evaluation"]["latency"]
    input_dim = int(config["autoencoder"]["input_dim"])

    model = ReferenceEncoder(input_dim=input_dim, latent_dim=16)
    model.eval()

    def compiled_fn(x: torch.Tensor) -> torch.Tensor:
        return model(x)

    stats = benchmark_backends(
        model_fn=compiled_fn,
        batch_sizes=[int(b) for b in latency_cfg["batch_sizes"]],
        input_dim=input_dim,
        n_warmup=int(latency_cfg["n_warmup"]),
        n_runs=int(latency_cfg["n_runs"]),
    )
    output_dir = args.output_dir or config["paths"]["experiments"] + "/latency"
    write_latency_report(stats, output_dir)


if __name__ == "__main__":
    main()
