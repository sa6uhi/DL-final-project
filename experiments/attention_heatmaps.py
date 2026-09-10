"""Generate FT-CAT temporal cross-attention heatmaps from a trained checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.evaluation.attention_maps import (
    AttentionCapture,
    collect_attention,
    full_history_mask,
    mean_history_attention,
    plot_case_study,
    plot_cross_attention_heatmap,
    top_attended_features,
)
from src.training.feature_selection import FeatureSpec
from src.training.train_transformer import DEFAULT_CHECKPOINT, materialize_tensors
from src.training.trainer_utils import resolve_device
from src.utils.config import load_config
from src.utils.logger import get_logger, setup_logging
from src.utils.seed import seed_everything

logger = get_logger(__name__)

DEFAULT_CONFIG = Path("config/config.yaml")
DEFAULT_HEATMAP = Path("figures/attention_cross_heatmap.png")
DEFAULT_CASE_STUDY = Path("figures/attention_case_study.png")
DEFAULT_RESULTS = Path("results/transformer/attention_summary.json")

DEFAULT_PER_CLASS = 512
DEFAULT_TOP_FEATURES = 10


def balanced_row_index(labels: np.ndarray, per_class: int, seed: int) -> np.ndarray:
    """Draw an equal number of fraudulent and legitimate row indices.

    Args:
        labels: Binary fraud labels for the split.
        per_class: Target rows per class; capped by availability.
        seed: Seed controlling the draw.

    Returns:
        Sorted row indices covering both classes.

    Raises:
        ValueError: If either class is absent, leaving nothing to contrast.
    """
    rng = np.random.default_rng(seed)
    fraud = np.flatnonzero(labels == 1)
    legit = np.flatnonzero(labels == 0)
    if fraud.size == 0 or legit.size == 0:
        raise ValueError(
            f"attention heatmaps need both classes (fraud={fraud.size}, legit={legit.size})"
        )

    quota = max(1, int(per_class))
    picked = [
        rng.choice(members, size=min(quota, members.size), replace=False)
        for members in (fraud, legit)
    ]
    index = np.sort(np.concatenate(picked))
    logger.info(
        "Sampled %d rows for attention capture (%d fraud, %d legitimate)",
        index.size,
        min(quota, fraud.size),
        min(quota, legit.size),
    )
    return index


def summarize_capture(
    capture: AttentionCapture,
    feature_names: list[str],
    top_n: int = DEFAULT_TOP_FEATURES,
) -> dict[str, Any]:
    """Reduce a capture to the numbers the paper quotes.

    Args:
        capture: A capture produced by
            :func:`~src.evaluation.attention_maps.collect_attention`.
        feature_names: Names of the non-``[CLS]`` query tokens, in tokenizer
            order.
        top_n: Number of history-driven feature tokens to report.

    Returns:
        A JSON-serialisable summary of the fraud/legitimate attention contrast.
    """
    fraud_mask = capture.fraud_mask()
    legit_mask = capture.legitimate_mask()
    fraud_attention = mean_history_attention(capture, fraud_mask)
    legit_attention = mean_history_attention(capture, legit_mask)
    ranking_mask = fraud_mask & full_history_mask(capture)
    if not ranking_mask.any():
        logger.warning(
            "No fraudulent row has a full %d-slot history; ranking over padded windows",
            capture.seq_len,
        )
        ranking_mask = fraud_mask

    return {
        "n_rows": len(capture),
        "n_fraud": int(fraud_mask.sum()),
        "n_legitimate": int(legit_mask.sum()),
        "n_full_history": int(full_history_mask(capture).sum()),
        "n_heads": capture.n_heads,
        "n_query_tokens": capture.n_queries,
        "seq_len": capture.seq_len,
        "padded_slot_rate": float(capture.padding.mean()),
        "fraud_cls_attention": fraud_attention.tolist(),
        "legitimate_cls_attention": legit_attention.tolist(),
        # Head-averaged mass on the most recent prior transaction: the single
        # number that says whether fraud leans harder on immediate velocity.
        "fraud_most_recent_slot": float(fraud_attention[:, 0].mean()),
        "legitimate_most_recent_slot": float(legit_attention[:, 0].mean()),
        "top_history_driven_features": [
            {"feature": name, "peak_attention": value}
            for name, value in top_attended_features(
                capture, feature_names, top_n=top_n, mask=ranking_mask
            )
        ],
    }


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for the attention heatmap generator.

    Args:
        argv: Command line arguments; uses ``sys.argv`` when omitted.

    Raises:
        FileNotFoundError: If the checkpoint or the test split is missing.
        KeyError: If the checkpoint carries no ``feature_spec`` payload.
    """
    parser = argparse.ArgumentParser(description="Render FT-CAT cross-attention heatmaps")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--test-data", type=str, default=None)
    parser.add_argument("--device", type=str, default=None, help="cuda, cpu; auto when omitted")
    parser.add_argument("--per-class", type=int, default=DEFAULT_PER_CLASS)
    parser.add_argument("--heatmap", type=str, default=str(DEFAULT_HEATMAP))
    parser.add_argument("--case-study", type=str, default=str(DEFAULT_CASE_STUDY))
    parser.add_argument("--out", type=str, default=str(DEFAULT_RESULTS))
    parser.add_argument("--top-features", type=int, default=DEFAULT_TOP_FEATURES)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    setup_logging(
        level=str(config.logging.level), log_file=str(config.get_path("logging.log_file"))
    )
    seed = int(config.seed)
    seed_everything(seed)

    # Imported lazily so this module can be imported without torch touching the
    # checkpoint path at collection time.
    from src.training.train_transformer import load_ft_transformer

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. Train FT-CAT first with "
            "`python src/training/train_transformer.py --variant ft_cat`."
        )

    device = resolve_device(args.device)
    model, payload = load_ft_transformer(checkpoint_path, device=device)
    if "feature_spec" not in payload:
        raise KeyError(
            f"Checkpoint {checkpoint_path} carries no 'feature_spec'; it was not written "
            "by src.training.train_transformer."
        )
    spec = FeatureSpec.from_dict(payload["feature_spec"])

    test_path = Path(args.test_data or config.get_path("data.test_data_path"))
    if not test_path.is_file():
        raise FileNotFoundError(
            f"Processed test split not found: {test_path}. "
            "Run `python -m src.data.prepare_data` first."
        )
    test_df = pd.read_parquet(test_path)
    bundle = materialize_tensors(test_df, spec)

    index = torch.from_numpy(
        balanced_row_index(bundle.labels_numpy(), args.per_class, seed).astype(np.int64)
    )
    capture = collect_attention(
        model,
        bundle.x_cont[index],
        bundle.x_cat[index],
        bundle.seq[index],
        bundle.y[index],
        device=device,
        max_rows=None,
    )

    feature_names = list(spec.continuous_cols) + list(spec.categorical_cols)
    plot_cross_attention_heatmap(capture, args.heatmap)

    # The most instructive case study is the transaction the deployed policy
    # would auto-block -- but only among rows carrying a full history window.
    # On a padded window the attention mask forces the entire distribution onto
    # the surviving slots, producing a saturated figure that says nothing about
    # what the model learned.
    eligible = capture.fraud_mask() & full_history_mask(capture)
    if not eligible.any():
        logger.warning(
            "No fraudulent row has a full %d-slot history; the case study will "
            "show a padded window",
            capture.seq_len,
        )
        eligible = capture.fraud_mask()
    candidate_rows = np.flatnonzero(eligible)
    case_row = int(candidate_rows[int(np.argmax(capture.scores[candidate_rows]))])
    plot_case_study(capture, case_row, feature_names, args.case_study)

    summary = summarize_capture(capture, feature_names, top_n=args.top_features)
    summary["case_study_row"] = case_row
    summary["case_study_score"] = float(capture.scores[case_row])
    summary["case_study_label"] = int(capture.labels[case_row])
    summary["checkpoint"] = str(checkpoint_path)

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    logger.info("Saved attention summary to %s", output)

    logger.info(
        "Attention heatmaps complete. Mean [CLS] mass on t-1: fraud %.4f vs legitimate %.4f",
        summary["fraud_most_recent_slot"],
        summary["legitimate_most_recent_slot"],
    )


if __name__ == "__main__":
    main()
