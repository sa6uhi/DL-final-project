# Import necessary modules and libraries
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from src.models.ft_transformer import FTCATransformer
from src.models.hybrid_gating import LearnedHybridGate
from src.serving.model_serializer import (
    FTTransformerExportModule,
    HybridGateExportModule,
    ScoreModule,
)
from src.training.train_autoencoder import load_checkpoint as load_autoencoder_checkpoint
from src.training.train_hybrid_gating import (
    load_checkpoint as load_hybrid_gate_checkpoint,
)
from src.training.train_transformer import load_ft_transformer
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

BATCH_SIZES = (1, 32, 128, 512)
WARMUP_RUNS = 10
MEASURED_RUNS = 50


def benchmark_callable(
    fn: Callable[[], torch.Tensor],
    batch_size: int,
    warmup_runs: int = WARMUP_RUNS,
    measured_runs: int = MEASURED_RUNS,
) -> dict[str, float]:
    """Benchmark a callable and return latency and throughput statistics."""
    for _ in range(warmup_runs):
        with torch.no_grad():
            fn()

    latencies_ms: list[float] = []

    for _ in range(measured_runs):
        start = time.perf_counter()

        with torch.no_grad():
            fn()

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        latencies_ms.append(elapsed_ms)

    latency_array = np.asarray(latencies_ms, dtype=np.float64)

    median_ms = float(np.median(latency_array))
    p90_ms = float(np.percentile(latency_array, 90))
    p99_ms = float(np.percentile(latency_array, 99))
    mean_ms = float(np.mean(latency_array))

    throughput = float(batch_size / (mean_ms / 1000.0)) if mean_ms > 0.0 else float("inf")

    return {
        "batch_size": batch_size,
        "mean_latency_ms": mean_ms,
        "median_latency_ms": median_ms,
        "p90_latency_ms": p90_ms,
        "p99_latency_ms": p99_ms,
        "throughput_transactions_per_second": throughput,
    }


def make_dae_inputs(
    model: nn.Module,
    batch_size: int,
) -> torch.Tensor:
    """Create synthetic DAE inputs matching the trained feature dimension."""
    input_dim = int(model.input_dim)

    return torch.randn(
        batch_size,
        input_dim,
        dtype=torch.float32,
    )


def make_ft_inputs(
    model: FTCATransformer,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create synthetic FT-CAT inputs matching the trained checkpoint contract."""
    meta = model.state_meta()

    n_continuous = int(meta["n_continuous"])
    categorical_cardinalities = list(meta["categorical_cardinalities"])
    seq_len = int(meta["seq_len"])
    seq_dim = int(meta["seq_dim"])

    x_cont = torch.randn(
        batch_size,
        n_continuous,
        dtype=torch.float32,
    )

    x_cat = torch.zeros(
        batch_size,
        len(categorical_cardinalities),
        dtype=torch.long,
    )

    seq = torch.randn(
        batch_size,
        seq_len,
        seq_dim,
        dtype=torch.float32,
    )

    return x_cont, x_cat, seq


def make_gate_inputs(
    gate: LearnedHybridGate,
    batch_size: int,
) -> torch.Tensor:
    """Create synthetic learned-gate feature inputs."""
    return torch.randn(
        batch_size,
        gate.input_dim,
        dtype=torch.float32,
    )


def benchmark_models(
    checkpoint_dir: Path,
    batch_sizes: tuple[int, ...],
    warmup_runs: int,
    measured_runs: int,
    l1_gamma: float,
) -> list[dict[str, float | str]]:
    """Benchmark DAE, FT-CAT, gate, and composed neural inference."""
    autoencoder = load_autoencoder_checkpoint(
        checkpoint_dir / "autoencoder.pt",
    )
    autoencoder.eval()

    ft_model, _ = load_ft_transformer(
        checkpoint_dir / "ft_transformer.pt",
        device="cpu",
    )
    ft_model.eval()

    gate, _ = load_hybrid_gate_checkpoint(
        checkpoint_dir / "hybrid_gating.pt",
    )
    gate.eval()

    dae_model = ScoreModule(
        autoencoder,
        l1_gamma=l1_gamma,
    )
    dae_model.eval()

    ft_export_model = FTTransformerExportModule(ft_model)
    ft_export_model.eval()

    gate_export_model = HybridGateExportModule(gate)
    gate_export_model.eval()

    results: list[dict[str, float | str]] = []

    for batch_size in batch_sizes:
        logger.info("Benchmarking batch size %d", batch_size)

        dae_input = make_dae_inputs(
            autoencoder,
            batch_size,
        )

        x_cont, x_cat, seq = make_ft_inputs(
            ft_model,
            batch_size,
        )

        gate_input = make_gate_inputs(
            gate,
            batch_size,
        )

        dae_metrics = benchmark_callable(
            lambda: dae_model(dae_input),
            batch_size,
            warmup_runs,
            measured_runs,
        )
        results.append(
            {
                "component": "DAE",
                **dae_metrics,
            }
        )

        ft_metrics = benchmark_callable(
            lambda: ft_export_model(
                x_cont,
                x_cat,
                seq,
            ),
            batch_size,
            warmup_runs,
            measured_runs,
        )
        results.append(
            {
                "component": "FT-CAT",
                **ft_metrics,
            }
        )

        gate_metrics = benchmark_callable(
            lambda: gate_export_model(gate_input),
            batch_size,
            warmup_runs,
            measured_runs,
        )
        results.append(
            {
                "component": "Learned Gate",
                **gate_metrics,
            }
        )

        def end_to_end() -> torch.Tensor:
            dae_scores = dae_model(dae_input).squeeze(-1)

            ft_logits = ft_export_model(
                x_cont,
                x_cat,
                seq,
            )
            ft_probabilities = torch.sigmoid(ft_logits)

            history_density = torch.full(
                (batch_size,),
                0.5,
                dtype=torch.float32,
            )

            history_amount_intensity = torch.full(
                (batch_size,),
                1.0,
                dtype=torch.float32,
            )

            if gate.input_dim == 4:
                gate_features = torch.stack(
                    [
                        dae_scores,
                        ft_probabilities,
                        history_density,
                        history_amount_intensity,
                    ],
                    dim=1,
                )
            elif gate.input_dim == 2:
                gate_features = torch.stack(
                    [
                        dae_scores,
                        ft_probabilities,
                    ],
                    dim=1,
                )
            else:
                raise ValueError(f"Unsupported gate input_dim={gate.input_dim}")

            return gate_export_model(gate_features)

        end_to_end_metrics = benchmark_callable(
            end_to_end,
            batch_size,
            warmup_runs,
            measured_runs,
        )
        results.append(
            {
                "component": "End-to-End Neural Pipeline",
                **end_to_end_metrics,
            }
        )

    return results


def save_csv(
    results: list[dict[str, float | str]],
    output_path: Path,
) -> None:
    """Save benchmark results to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "component",
        "batch_size",
        "mean_latency_ms",
        "median_latency_ms",
        "p90_latency_ms",
        "p99_latency_ms",
        "throughput_transactions_per_second",
    ]

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(results)


def save_json(
    results: list[dict[str, float | str]],
    output_path: Path,
    warmup_runs: int,
    measured_runs: int,
) -> None:
    """Save benchmark results and experiment metadata to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "device": "cpu",
        "torch_version": torch.__version__,
        "warmup_runs": warmup_runs,
        "measured_runs": measured_runs,
        "results": results,
    }

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            payload,
            handle,
            indent=2,
        )


def plot_results(
    results: list[dict[str, float | str]],
    output_path: Path,
) -> None:
    """Plot throughput by component and batch size."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    components = [
        "DAE",
        "FT-CAT",
        "Learned Gate",
        "End-to-End Neural Pipeline",
    ]

    markers = ["o", "s", "^", "D"]

    plt.figure(figsize=(9, 6))

    for component, marker in zip(
        components,
        markers,
    ):
        component_rows = [row for row in results if row["component"] == component]

        component_rows.sort(key=lambda row: int(row["batch_size"]))

        batch_sizes = [int(row["batch_size"]) for row in component_rows]

        throughputs = [float(row["throughput_transactions_per_second"]) for row in component_rows]

        plt.plot(
            batch_sizes,
            throughputs,
            marker=marker,
            linewidth=2,
            label=component,
        )

    plt.xlabel("Batch Size")
    plt.ylabel("Throughput (transactions/second)")
    plt.title("Inference Throughput by Model Component")
    plt.xscale("log", base=2)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def main() -> None:
    """Run the trained-model CPU inference benchmark."""
    parser = argparse.ArgumentParser(
        description=("Benchmark trained fraud-detection model inference " "latency and throughput")
    )

    parser.add_argument(
        "--checkpoints",
        default="models/checkpoints",
        help="Directory containing trained checkpoints",
    )
    parser.add_argument(
        "--output",
        default="results/benchmark",
        help="Directory for benchmark result files",
    )
    parser.add_argument(
        "--figures",
        default="figures/inference_benchmark",
        help="Directory for benchmark figures",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=WARMUP_RUNS,
    )
    parser.add_argument(
        "--measured-runs",
        type=int,
        default=MEASURED_RUNS,
    )

    args = parser.parse_args()

    config = load_config()

    checkpoint_dir = Path(args.checkpoints)
    output_dir = Path(args.output)
    figure_dir = Path(args.figures)

    l1_gamma = float(config.get("autoencoder", {}).get("anomaly_score", {}).get("l1_gamma", 0.4))

    torch.set_grad_enabled(False)

    results = benchmark_models(
        checkpoint_dir=checkpoint_dir,
        batch_sizes=BATCH_SIZES,
        warmup_runs=args.warmup_runs,
        measured_runs=args.measured_runs,
        l1_gamma=l1_gamma,
    )

    csv_path = output_dir / "inference_benchmark.csv"
    json_path = output_dir / "inference_benchmark.json"
    figure_path = figure_dir / "inference_throughput.png"

    save_csv(
        results,
        csv_path,
    )

    save_json(
        results,
        json_path,
        args.warmup_runs,
        args.measured_runs,
    )

    plot_results(
        results,
        figure_path,
    )

    logger.info(
        "Saved benchmark CSV to %s",
        csv_path,
    )
    logger.info(
        "Saved benchmark JSON to %s",
        json_path,
    )
    logger.info(
        "Saved throughput plot to %s",
        figure_path,
    )

    for row in results:
        logger.info(
            "%-27s batch=%-4d median=%8.3f ms " "p99=%8.3f ms throughput=%10.1f tx/s",
            row["component"],
            row["batch_size"],
            row["median_latency_ms"],
            row["p99_latency_ms"],
            row["throughput_transactions_per_second"],
        )


if __name__ == "__main__":
    main()
