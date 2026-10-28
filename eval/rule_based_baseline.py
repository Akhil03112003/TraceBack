#!/usr/bin/env python3
"""Deterministic, explainable rule-based HDFS anomaly baseline."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

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

DEFAULT_OCCURRENCE_PATH = Path(
    r"C:\Users\akhil\Downloads\HDFS_v1\preprocessed\Event_occurrence_matrix.csv"
)
DEFAULT_LABEL_PATH = Path(
    r"C:\Users\akhil\Downloads\HDFS_v1\preprocessed\anomaly_label.csv"
)
DEFAULT_OUTPUT_PATH = Path("eval") / "results" / "rule_based_baseline.json"


@dataclass(frozen=True)
class FeatureRecord:
    identifier: str
    label: str
    counts: dict[str, int]


@dataclass(frozen=True)
class Rule:
    name: str
    description: str
    event_ids: tuple[str, ...]
    threshold: int = 1

    def predicts_anomaly(self, record: FeatureRecord) -> bool:
        return any(record.counts[event_id] >= self.threshold for event_id in self.event_ids)


ANOMALY_ASSOCIATED_EVENTS = (
    "E6", "E7", "E10", "E12", "E13", "E14", "E15", "E17",
    "E18", "E19", "E20", "E24", "E25", "E27", "E28", "E29",
)

CANDIDATE_RULES = (
    Rule(
        "anomaly_associated_event_presence",
        "Predict anomaly when at least one event from the predefined "
        "anomaly-associated event family occurs.",
        ANOMALY_ASSOCIATED_EVENTS,
    ),
    Rule(
        "transfer_and_exception_event_presence",
        "Predict anomaly when at least one transfer, exception, or "
        "replication-related event occurs.",
        ("E6", "E7", "E10", "E12", "E13", "E14", "E15", "E17", "E18", "E19", "E20", "E24", "E25"),
    ),
    Rule(
        "rare_error_event_presence",
        "Predict anomaly when at least one rare error event occurs.",
        ("E7", "E10", "E12", "E14", "E15", "E17", "E19", "E24", "E27", "E28", "E29"),
    ),
)


def _read_occurrence_records(path: Path) -> tuple[list[FeatureRecord], tuple[str, ...]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = tuple(reader.fieldnames or ())
        event_ids = tuple(column for column in columns if column.startswith("E"))
        if "BlockId" not in columns or "Label" not in columns:
            raise ValueError("Occurrence data must contain BlockId and Label columns")
        if not event_ids:
            raise ValueError("Occurrence data contains no event-count columns")
        if "Type" in columns:
            excluded = {"BlockId", "Label", "Type"}
        else:
            excluded = {"BlockId", "Label"}
        if set(event_ids) & excluded:
            raise ValueError("Feature schema overlaps excluded metadata")
        records: list[FeatureRecord] = []
        for row_number, row in enumerate(reader, start=2):
            identifier = row["BlockId"]
            if not identifier:
                raise ValueError(f"Empty BlockId at row {row_number}")
            counts: dict[str, int] = {}
            for event_id in event_ids:
                raw = row[event_id]
                try:
                    value = float(raw)
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        f"Non-numeric feature {event_id!r} at row {row_number}: {raw!r}"
                    ) from error
                if not math.isfinite(value) or value < 0 or not value.is_integer():
                    raise ValueError(
                        f"Invalid count {event_id!r} at row {row_number}: {raw!r}"
                    )
                counts[event_id] = int(value)
            records.append(FeatureRecord(identifier, normalize_label(row["Label"]), counts))
    if not records:
        raise ValueError("Occurrence data contains no records")
    return records, event_ids


def _select_rule(records: Sequence[FeatureRecord], train_ids: set[str]) -> tuple[Rule, list[dict[str, object]]]:
    training = [record for record in records if record.identifier in train_ids]
    labels = [record.label for record in training]
    scores: list[dict[str, object]] = []
    for rule in CANDIDATE_RULES:
        predictions = [
            "Anomaly" if rule.predicts_anomaly(record) else "Normal"
            for record in training
        ]
        metrics = calculate_metrics(labels, predictions)
        scores.append({"rule": rule.name, "metrics": metrics})
    selected = max(
        zip(CANDIDATE_RULES, scores),
        key=lambda item: (
            item[1]["metrics"]["anomaly_f1"],
            item[1]["metrics"]["anomaly_recall"],
            item[1]["metrics"]["anomaly_precision"],
            -item[1]["metrics"]["false_positives"],
        ),
    )[0]
    return selected, scores


def _id_digest(identifiers: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(sorted(identifiers)).encode("utf-8")).hexdigest()


def evaluate_dataset(
    occurrence_path: Path,
    label_path: Path,
    output_path: Path,
    split_seed: int = 42,
    test_fraction: float = 0.2,
) -> dict[str, object]:
    started = time.perf_counter()
    occurrence, event_ids = _read_occurrence_records(occurrence_path)
    label_ids, labels = read_labels(label_path)
    labels_by_id = dict(zip(label_ids, labels))
    occurrence_by_id = {record.identifier: record for record in occurrence}
    if set(label_ids) != set(occurrence_by_id):
        raise ValueError("Label and occurrence BlockId sets do not match")
    if len(occurrence_by_id) != len(occurrence):
        raise ValueError("Occurrence data contains duplicate BlockId values")
    for record in occurrence:
        if record.label != labels_by_id[record.identifier]:
            raise ValueError(f"Label mismatch for {record.identifier}")
    train_ids, test_ids = stratified_split(
        label_ids, labels, test_fraction=test_fraction, seed=split_seed
    )
    rule, candidate_scores = _select_rule(occurrence, set(train_ids))
    def predict(ids: Sequence[str]) -> list[str]:
        return [
            "Anomaly" if rule.predicts_anomaly(occurrence_by_id[identifier]) else "Normal"
            for identifier in ids
        ]
    train_labels = [labels_by_id[identifier] for identifier in train_ids]
    test_labels = [labels_by_id[identifier] for identifier in test_ids]
    result: dict[str, object] = {
        "method": "deterministic rule-based baseline",
        "target_positive_class": "Anomaly",
        "feature_source": str(occurrence_path),
        "excluded_columns": ["BlockId", "Label", "Type"],
        "event_count_features": list(event_ids),
        "rule_selection": {
            "selected_rule": rule.name,
            "selection_data": "training records only",
            "objective": "maximize anomaly F1, then recall, precision, and minimize false positives",
            "candidate_scores": candidate_scores,
            "frozen_rule": {
                "name": rule.name,
                "description": rule.description,
                "event_ids": list(rule.event_ids),
                "threshold": rule.threshold,
                "threshold_definition": "event count >= 1; semantic event-presence threshold",
            },
        },
        "dataset": {
            "label_path": str(label_path),
            "occurrence_path": str(occurrence_path),
            "records": len(labels),
            "label_counts": dict(Counter(labels)),
        },
        "split": {
            "policy": "same stratified per-label shuffle used by Step 2",
            "seed": split_seed,
            "test_fraction": test_fraction,
            "train_records": len(train_ids),
            "test_records": len(test_ids),
            "train_id_sha256": _id_digest(train_ids),
            "test_id_sha256": _id_digest(test_ids),
            "test_label_counts": dict(Counter(test_labels)),
        },
        "training_selection": calculate_metrics(train_labels, predict(train_ids)),
        "held_out_test": calculate_metrics(test_labels, predict(test_ids)),
        "whole_dataset": calculate_metrics(labels, predict(label_ids)),
        "runtime_seconds": time.perf_counter() - started,
        "limitations": [
            "The rule family is selected from a small predefined candidate set.",
            "Event presence is interpretable but does not establish causality.",
            "The NPZ is not used and no records are joined by row position.",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--occurrence", type=Path, default=DEFAULT_OCCURRENCE_PATH)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABEL_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    args = parser.parse_args()
    print(json.dumps(evaluate_dataset(
        args.occurrence, args.labels, args.output, args.seed, args.test_fraction
    ), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
