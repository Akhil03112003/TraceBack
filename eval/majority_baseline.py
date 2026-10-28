#!/usr/bin/env python3
"""Majority-class anomaly-detection baseline."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Sequence

try:
    from .evaluation_utils import (
        calculate_metrics,
        normalize_label,
        read_labels,
        stratified_split,
    )
except ImportError:
    from evaluation_utils import (  # type: ignore[no-redef]
        calculate_metrics,
        normalize_label,
        read_labels,
        stratified_split,
    )
DEFAULT_DATASET_PATH = Path(
    r"C:\Users\akhil\Downloads\HDFS_v1\preprocessed\anomaly_label.csv"
)
DEFAULT_OUTPUT_PATH = Path("eval") / "results" / "majority_baseline.json"


def majority_predictions(labels: Sequence[str]) -> list[str]:
    if not labels:
        raise ValueError("At least one label is required")
    return ["Normal"] * len(labels)


def evaluate_dataset(
    dataset_path: Path,
    output_path: Path,
    split_seed: int = 42,
    test_fraction: float = 0.2,
) -> dict[str, object]:
    identifiers, labels = read_labels(dataset_path)
    predictions = majority_predictions(labels)
    train_ids, test_ids = stratified_split(
        identifiers, labels, test_fraction=test_fraction, seed=split_seed
    )
    label_by_id = dict(zip(identifiers, labels))
    test_labels = [label_by_id[identifier] for identifier in test_ids]
    result: dict[str, object] = {
        "method": "majority-class baseline",
        "prediction_class": "Normal",
        "target_positive_class": "Anomaly",
        "dataset": {
            "path": str(dataset_path),
            "source_file": dataset_path.name,
            "label_column": "Label",
            "identifier_column": "BlockId",
            "records": len(labels),
            "label_counts": dict(Counter(labels)),
        },
        "split": {
            "policy": "stratified random split by normalized label; no model fitting",
            "seed": split_seed,
            "test_fraction": test_fraction,
            "train_records": len(train_ids),
            "test_records": len(test_ids),
            "test_label_counts": dict(Counter(test_labels)),
        },
        "whole_dataset": calculate_metrics(labels, predictions),
        "held_out_test": calculate_metrics(
            test_labels, majority_predictions(test_labels)
        ),
        "limitations": [
            "The predictor never inspects features and always predicts Normal.",
            "BlockId and Label are evaluation metadata, not prediction features.",
            "The held-out split is recorded for consistent future detector comparisons.",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    args = parser.parse_args()
    result = evaluate_dataset(
        args.dataset, args.output, args.seed, args.test_fraction
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
