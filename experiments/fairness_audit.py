"""Experiment: subgroup performance-disparity audit for fraud triage.

The IEEE-CIS data has no demographic labels (race, gender, age, etc.), so
this is NOT a protected-attribute fairness audit. It instead checks whether
triage behavior (review burden, missed-fraud rate, ranking quality) is
stable across business-segment proxies that correlate with how a customer
transacts: device type, card network/type, product category, and email
provider. A large gap across categories of the same attribute is a signal
worth investigating even without demographic ground truth.

Reuses the already-calibrated evaluation split in
``results/conformal/conformal_inputs.npz`` (row-aligned with
``data/processed/test.parquet``) and the same
:class:`~src.uncertainty.conformal_predictor.SplitConformalPredictor` used
by serving, so the reported review/recall/fpr rates match what the
deployed triage threshold actually does in production, not an arbitrary
cutoff.
"""

# Import necessary modules and libraries
from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.evaluation.metrics import average_precision
from src.uncertainty.conformal_predictor import SplitConformalPredictor
from src.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_ARCHIVE = Path("results") / "conformal" / "conformal_inputs.npz"
DEFAULT_TEST_PARQUET = Path("data") / "processed" / "test.parquet"
DEFAULT_PREPROCESSOR = Path("data") / "processed" / "preprocessor.pkl"
DEFAULT_OUTPUT_DIR = Path("results") / "fairness"
DEFAULT_ALPHA = 0.01

# Proxy business-segment columns audited in place of unavailable demographic
# attributes. All four are integer-encoded categoricals in the processed
# frame; label_maps() decodes them back to their original string values.
SUBGROUP_COLUMNS = ("DeviceType", "card4", "card6", "ProductCD", "P_emaildomain")

# A subgroup below these counts gets flagged unreliable rather than dropped,
# so a rare category is still visible in the report instead of silently
# vanishing from it.
MIN_SUBGROUP_SIZE = 30
MIN_SUBGROUP_FRAUD_COUNT = 10


def load_audit_inputs(
    archive_path: str | Path,
    test_parquet_path: str | Path,
    preprocessor_path: str | Path,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, dict[str, dict[int, str]]]:
    """Load the evaluation split, its subgroup columns, and their labels.

    Returns:
        A tuple of ``(eval_probabilities, eval_labels, subgroup_frame,
        label_maps)`` where ``subgroup_frame`` has exactly
        ``len(eval_probabilities)`` rows in the same order as the archive,
        and ``label_maps`` decodes each subgroup column's integer codes
        back to their original category names.

    Raises:
        ValueError: If the test frame and the archive's evaluation split
            do not have matching lengths or labels, which would mean they
            are no longer row-aligned.
    """
    archive = np.load(archive_path)
    eval_probabilities = archive["eval_probabilities"]
    eval_labels = archive["eval_labels"]

    frame = pd.read_parquet(
        test_parquet_path,
        columns=list(SUBGROUP_COLUMNS) + ["isFraud"],
    )

    if len(frame) != len(eval_labels):
        raise ValueError(
            f"test parquet has {len(frame)} rows but the archive's evaluation "
            f"split has {len(eval_labels)}; they must be row-aligned"
        )

    if not np.array_equal(frame["isFraud"].to_numpy(), eval_labels):
        raise ValueError(
            "test parquet's isFraud column does not match the archive's "
            "eval_labels; refusing to audit a misaligned join"
        )

    with Path(preprocessor_path).open("rb") as handle:
        preprocessor = pickle.load(handle)

    label_maps = {
        column: {code: label for label, code in preprocessor.cat_vocabularies[column].items()}
        for column in SUBGROUP_COLUMNS
    }

    return eval_probabilities, eval_labels, frame[list(SUBGROUP_COLUMNS)], label_maps


def fit_serving_predictor(
    archive_path: str | Path,
    alpha: float,
) -> SplitConformalPredictor:
    """Calibrate a split-conformal predictor matching the deployed alpha."""
    archive = np.load(archive_path)

    predictor = SplitConformalPredictor(alpha=alpha)
    predictor.fit(
        fraud_probabilities=torch.as_tensor(
            archive["calibration_probabilities"],
            dtype=torch.float32,
        ),
        labels=torch.as_tensor(
            archive["calibration_labels"],
            dtype=torch.long,
        ),
    )

    return predictor


def triage_decisions(
    predictor: SplitConformalPredictor,
    eval_probabilities: np.ndarray,
) -> np.ndarray:
    """Return the operational triage decision for every evaluation row."""
    return np.array(
        [predictor.predict_triage(float(probability)) for probability in eval_probabilities]
    )


def subgroup_metrics(
    subgroup_frame: pd.DataFrame,
    eval_probabilities: np.ndarray,
    eval_labels: np.ndarray,
    decisions: np.ndarray,
    label_maps: dict[str, dict[int, str]],
) -> list[dict[str, object]]:
    """Compute triage and ranking metrics for every category of every column.

    For each subgroup:
      - ``fraud_rate``: share of transactions that are actually fraud.
      - ``review_rate``: share routed to human review (operational burden).
      - ``recall``: share of true fraud NOT auto-approved (caught by
        auto_block or human_review). A miss is a fraud that slipped through
        as auto_approve, so this is the metric that matters for exposure.
      - ``fpr``: share of legitimate transactions NOT auto-approved
        (auto_block or human_review). This is the friction/false-alarm
        burden imposed on legitimate customers in that subgroup.
      - ``pr_auc``: ranking quality of the model's raw probabilities within
        the subgroup, independent of where the conformal threshold sits.
    """
    rows: list[dict[str, object]] = []

    for column in SUBGROUP_COLUMNS:
        codes = subgroup_frame[column].to_numpy()

        for code in sorted(np.unique(codes)):
            mask = codes == code
            n = int(mask.sum())

            group_labels = eval_labels[mask]
            group_probabilities = eval_probabilities[mask]
            group_decisions = decisions[mask]

            n_fraud = int(group_labels.sum())
            n_legit = n - n_fraud

            not_auto_approved = np.isin(group_decisions, ["auto_block", "human_review"])

            recall = (
                float(not_auto_approved[group_labels == 1].mean()) if n_fraud > 0 else float("nan")
            )
            fpr = (
                float(not_auto_approved[group_labels == 0].mean()) if n_legit > 0 else float("nan")
            )
            pr_auc = (
                float(average_precision(group_probabilities, group_labels))
                if n_fraud > 0 and n_legit > 0
                else float("nan")
            )

            rows.append(
                {
                    "attribute": column,
                    "category": label_maps[column].get(int(code), f"code_{code}"),
                    "n": n,
                    "fraud_rate": float(n_fraud / n) if n > 0 else float("nan"),
                    "review_rate": float((group_decisions == "human_review").mean()),
                    "recall": recall,
                    "fpr": fpr,
                    "pr_auc": pr_auc,
                    "reliable": n >= MIN_SUBGROUP_SIZE and n_fraud >= MIN_SUBGROUP_FRAUD_COUNT,
                }
            )

    return rows


def export_subgroup_metrics(
    rows: list[dict[str, object]],
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """Export subgroup audit metrics to CSV and JSON."""
    if not rows:
        raise ValueError("rows must not be empty")

    output_directory = Path(output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)

    csv_path = output_directory / "subgroup_metrics.csv"
    json_path = output_directory / "subgroup_metrics.json"

    fieldnames = list(rows[0].keys())

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with json_path.open("w", encoding="utf-8") as json_file:
        json.dump(rows, json_file, indent=2, allow_nan=True)

    logger.info("Saved subgroup fairness CSV metrics to %s", csv_path)
    logger.info("Saved subgroup fairness JSON metrics to %s", json_path)

    return csv_path, json_path


# Define the main function to run the subgroup fairness audit from saved artifacts
def main(argv: list[str] | None = None) -> None:
    """Run the subgroup performance-disparity audit from saved artifacts."""
    parser = argparse.ArgumentParser(
        description="Audit fraud-triage performance disparity across business-segment proxies."
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=DEFAULT_ARCHIVE,
        help="NPZ archive with calibration/eval probabilities and labels.",
    )
    parser.add_argument(
        "--test-parquet",
        type=Path,
        default=DEFAULT_TEST_PARQUET,
        help="Processed test split providing the subgroup columns.",
    )
    parser.add_argument(
        "--preprocessor",
        type=Path,
        default=DEFAULT_PREPROCESSOR,
        help="Fitted FraudPreprocessor pickle, used to decode category codes.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
        help="Conformal miscoverage level, matching the deployed serving alpha.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Directory for subgroup_metrics.{csv,json}. Tests MUST override this "
            "to an isolated tmp_path so a run can never overwrite the real report."
        ),
    )
    args = parser.parse_args(argv)

    eval_probabilities, eval_labels, subgroup_frame, label_maps = load_audit_inputs(
        archive_path=args.archive,
        test_parquet_path=args.test_parquet,
        preprocessor_path=args.preprocessor,
    )

    predictor = fit_serving_predictor(archive_path=args.archive, alpha=args.alpha)
    decisions = triage_decisions(predictor=predictor, eval_probabilities=eval_probabilities)

    rows = subgroup_metrics(
        subgroup_frame=subgroup_frame,
        eval_probabilities=eval_probabilities,
        eval_labels=eval_labels,
        decisions=decisions,
        label_maps=label_maps,
    )

    export_subgroup_metrics(rows=rows, output_dir=args.output_dir)

    unreliable = [row for row in rows if not row["reliable"]]
    if unreliable:
        logger.info(
            "%d of %d subgroups have fewer than %d transactions or %d fraud "
            "cases and are flagged unreliable in the report, not dropped.",
            len(unreliable),
            len(rows),
            MIN_SUBGROUP_SIZE,
            MIN_SUBGROUP_FRAUD_COUNT,
        )


if __name__ == "__main__":
    main()
