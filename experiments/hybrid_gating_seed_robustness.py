"""Evaluate learned hybrid-gate robustness across random training seeds."""

# Import necessary modules and libraries
from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from src.training.dae_features import resolve_dae_feature_columns
from src.training.feature_selection import FeatureSpec
from src.training.gate_velocity import extract_velocity_features
from src.training.hybrid_pipeline import (
    autoencoder_anomaly_scores,
    create_hybrid_data_split,
    learned_gate_probabilities,
    make_gate_data,
    transformer_probabilities,
)
from src.training.train_autoencoder import (load_checkpoint as load_autoencoder_checkpoint)
from src.training.train_hybrid_gating import GateData, train_gate
from src.training.train_transformer import (
    _load_split,
    load_ft_transformer,
    materialize_tensors,
)
from src.utils.config import Config, load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_CONFIG = Path("config/config.yaml")
DEFAULT_RESULTS_DIR = Path("results/hybrid_gating")
DEFAULT_FIGURES_DIR = Path("figures/hybrid_gating")
DEFAULT_SEEDS = (42, 123, 2026)


# Define helper function to extract DAE features from a DataFrame
def dae_features_from_frame(
    frame,
    non_feature_cols: list[str],
    expected_dim: int,
) -> torch.Tensor:
    """Materialize the trained DAE feature contract for every transaction."""
    feature_cols = resolve_dae_feature_columns(
        frame,
        non_feature_cols=non_feature_cols,
        expected_dim=expected_dim,
    )

    features = frame.loc[:, feature_cols].to_numpy(
        dtype=np.float32,
        copy=True,
    )

    if features.ndim != 2:
        raise ValueError("DAE features must be a two-dimensional matrix")

    if features.shape[0] != len(frame):
        raise ValueError("DAE feature rows must match the input frame")

    if features.shape[1] != expected_dim:
        raise ValueError(
            f"DAE feature dimension mismatch: expected {expected_dim}, " f"got {features.shape[1]}"
        )

    if not np.isfinite(features).all():
        raise ValueError("DAE features must contain only finite values")

    return torch.from_numpy(features).float()


# Define helper functions for gate data generation, evaluation, and result saving
def build_gate_data(
    frame,
    autoencoder: torch.nn.Module,
    ft_model: torch.nn.Module,
    feature_spec: FeatureSpec,
    l1_gamma: float,
    non_feature_cols: list[str],
    dae_expected_dim: int,
) -> GateData:
    """Generate aligned real upstream signals for one chronological frame."""
    dae_tensor = dae_features_from_frame(
        frame=frame,
        non_feature_cols=non_feature_cols,
        expected_dim=dae_expected_dim,
    )

    ft_bundle = materialize_tensors(frame, feature_spec)

    anomaly_scores = autoencoder_anomaly_scores(
        autoencoder=autoencoder,
        features=dae_tensor,
        l1_gamma=l1_gamma,
    )

    ft_probabilities = transformer_probabilities(
        transformer=ft_model,
        x_cont=ft_bundle.x_cont,
        x_cat=ft_bundle.x_cat,
        sequence=ft_bundle.seq,
    )

    velocity_features = extract_velocity_features(
        ft_bundle.seq.cpu().numpy(),
        transaction_amount_index=0,
    ).float()

    return make_gate_data(
        anomaly_scores=anomaly_scores,
        ft_probabilities=ft_probabilities,
        velocity_features=velocity_features,
        labels=ft_bundle.y.long(),
    )


def tpr_at_fpr(
    labels: np.ndarray,
    probabilities: np.ndarray,
    max_fpr: float,
) -> float:
    """Return the best TPR achievable without exceeding max_fpr."""
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)

    order = np.argsort(-probabilities, kind="stable")
    ordered_labels = labels[order]

    positives = int(np.sum(ordered_labels == 1))
    negatives = int(np.sum(ordered_labels == 0))

    if positives == 0 or negatives == 0:
        return float("nan")

    true_positives = np.cumsum(ordered_labels == 1)
    false_positives = np.cumsum(ordered_labels == 0)

    tpr = true_positives / positives
    fpr = false_positives / negatives

    valid = fpr <= max_fpr
    if not np.any(valid):
        return 0.0

    return float(np.max(tpr[valid]))


def evaluate_gate(
    gate: torch.nn.Module,
    normalizer,
    data: GateData,
    max_fpr: float,
) -> dict[str, float]:
    """Evaluate one learned gate on a fixed development-validation set."""
    probabilities = learned_gate_probabilities(
        gate=gate,
        normalizer=normalizer,
        anomaly_scores=data.anomaly_scores,
        ft_probabilities=data.ft_probabilities,
        velocity_features=data.velocity_features,
    )

    labels = data.labels.detach().cpu().numpy().astype(np.int64)
    scores = probabilities.detach().cpu().numpy().astype(np.float64)

    return {
        "pr_auc": float(average_precision_score(labels, scores)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "tpr_at_fpr": tpr_at_fpr(
            labels=labels,
            probabilities=scores,
            max_fpr=max_fpr,
        ),
    }


def save_results(
    rows: list[dict[str, Any]],
    results_dir: Path,
) -> None:
    """Write per-seed robustness metrics to CSV and JSON."""
    results_dir.mkdir(parents=True, exist_ok=True)

    csv_path = results_dir / "seed_robustness.csv"
    json_path = results_dir / "seed_robustness.json"

    fieldnames = list(rows[0].keys())

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)

    logger.info("Saved seed robustness CSV to %s", csv_path)
    logger.info("Saved seed robustness JSON to %s", json_path)


def save_plot(
    rows: list[dict[str, Any]],
    figures_dir: Path,
) -> None:
    """Plot gate-validation PR-AUC for each training seed."""
    figures_dir.mkdir(parents=True, exist_ok=True)

    seeds = [str(row["seed"]) for row in rows]
    pr_auc = [float(row["pr_auc"]) for row in rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        seeds,
        pr_auc,
        color=["#166534", "#15803d", "#22c55e"],
    )

    ax.set_xlabel("Training seed")
    ax.set_ylabel("Gate-validation PR-AUC")
    ax.set_title("Learned Hybrid Gate — Seed Robustness")
    ax.set_ylim(0.0, min(1.0, max(pr_auc) * 1.20))

    for bar, value in zip(bars, pr_auc, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.4f}",
            ha="center",
            va="bottom",
        )

    fig.tight_layout()

    output_path = figures_dir / "seed_robustness.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    logger.info("Saved seed robustness plot to %s", output_path)


def run_seed_robustness(
    config: Config,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    device: str = "cpu",
) -> list[dict[str, Any]]:
    """Train and compare the learned gate across fixed random seeds."""
    validation_path = Path(config.data.val_data_path)
    checkpoint_dir = Path(config.paths.checkpoints)

    validation_df = _load_split(validation_path)
    split = create_hybrid_data_split(validation_df, config)

    autoencoder_path = checkpoint_dir / "autoencoder.pt"
    transformer_path = checkpoint_dir / "ft_transformer.pt"

    logger.info("Loading trained DAE from %s", autoencoder_path)
    autoencoder = load_autoencoder_checkpoint(
        autoencoder_path,
        device=device,
    )
    autoencoder.eval()

    logger.info("Loading trained FT-CAT from %s", transformer_path)
    ft_model, ft_payload = load_ft_transformer(
        transformer_path,
        device=device,
    )
    ft_model.eval()

    feature_spec_payload = ft_payload.get("feature_spec")
    if not isinstance(feature_spec_payload, dict):
        raise ValueError("FT-CAT checkpoint does not contain a valid feature_spec")

    feature_spec = FeatureSpec.from_dict(feature_spec_payload)

    l1_gamma = float(config.autoencoder.anomaly_score.l1_gamma)
    non_feature_cols = [str(col) for col in config.data.non_feature_cols]
    dae_expected_dim = int(config.autoencoder.input_dim)

    logger.info("Generating real upstream signals for gate training")
    gate_train_data = build_gate_data(
        frame=split.gate_train,
        autoencoder=autoencoder,
        ft_model=ft_model,
        feature_spec=feature_spec,
        l1_gamma=l1_gamma,
        non_feature_cols=non_feature_cols,
        dae_expected_dim=dae_expected_dim,
    )

    logger.info("Generating real upstream signals for gate validation")
    gate_val_data = build_gate_data(
        frame=split.gate_val,
        autoencoder=autoencoder,
        ft_model=ft_model,
        feature_spec=feature_spec,
        l1_gamma=l1_gamma,
        non_feature_cols=non_feature_cols,
        dae_expected_dim=dae_expected_dim,
    )

    max_fpr = float(config.evaluation.anomaly_eval.max_fpr)

    rows: list[dict[str, Any]] = []

    for seed in seeds:
        logger.info("Training learned gate with seed %d", seed)

        seed_config = copy.deepcopy(config)
        seed_config["seed"] = int(seed)

        checkpoint_path = checkpoint_dir / f"hybrid_gating_seed_{seed}.pt"

        gate, normalizer = train_gate(
            train_data=gate_train_data,
            val_data=gate_val_data,
            config=seed_config,
            device=device,
            checkpoint_path=checkpoint_path,
        )

        metrics = evaluate_gate(
            gate=gate,
            normalizer=normalizer,
            data=gate_val_data,
            max_fpr=max_fpr,
        )

        row = {
            "seed": int(seed),
            "pr_auc": metrics["pr_auc"],
            "roc_auc": metrics["roc_auc"],
            "tpr_at_fpr": metrics["tpr_at_fpr"],
            "checkpoint": str(checkpoint_path),
        }
        rows.append(row)

        logger.info(
            "Seed %d - PR-AUC %.6f, ROC-AUC %.6f, TPR@FPR %.6f",
            seed,
            metrics["pr_auc"],
            metrics["roc_auc"],
            metrics["tpr_at_fpr"],
        )

    return rows


def main() -> None:
    """Run the learned-gate seed-robustness experiment."""
    parser = argparse.ArgumentParser(
        description="Evaluate learned hybrid-gate robustness across training seeds"
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=("cpu", "cuda"),
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_SEEDS),
    )
    parser.add_argument(
        "--results-dir",
        default=str(DEFAULT_RESULTS_DIR),
    )
    parser.add_argument(
        "--figures-dir",
        default=str(DEFAULT_FIGURES_DIR),
    )
    args = parser.parse_args()

    config = load_config(args.config)

    rows = run_seed_robustness(
        config=config,
        seeds=tuple(args.seeds),
        device=args.device,
    )

    if not rows:
        raise RuntimeError("Seed robustness experiment produced no results")

    save_results(
        rows=rows,
        results_dir=Path(args.results_dir),
    )
    save_plot(
        rows=rows,
        figures_dir=Path(args.figures_dir),
    )

    pr_auc_values = np.asarray(
        [row["pr_auc"] for row in rows],
        dtype=np.float64,
    )

    logger.info(
        "Seed robustness complete - validation PR-AUC mean %.6f, std %.6f",
        float(pr_auc_values.mean()),
        float(pr_auc_values.std(ddof=0)),
    )


if __name__ == "__main__":
    main()
