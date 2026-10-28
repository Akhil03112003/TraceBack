"""Shared label, metric, and split utilities for Phase 1 evaluations."""

from __future__ import annotations

import csv
import random
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

NORMAL_LABELS = frozenset({"Normal", "Success"})
ANOMALY_LABELS = frozenset({"Anomaly", "Fail"})
CONFUSION_MATRIX_CLASS_ORDER = ("Normal", "Anomaly")


def normalize_label(value: str) -> str:
    if value in NORMAL_LABELS:
        return "Normal"
    if value in ANOMALY_LABELS:
        return "Anomaly"
    raise ValueError(f"Unsupported label value: {value!r}")


def calculate_metrics(
    actual_labels: Iterable[str], predicted_labels: Iterable[str]
) -> dict[str, object]:
    actual = [normalize_label(label) for label in actual_labels]
    predicted = [normalize_label(label) for label in predicted_labels]
    if not actual:
        raise ValueError("At least one actual label is required")
    if len(actual) != len(predicted):
        raise ValueError("Actual and predicted labels must have equal lengths")
    tn = sum(a == "Normal" and p == "Normal" for a, p in zip(actual, predicted))
    fp = sum(a == "Normal" and p == "Anomaly" for a, p in zip(actual, predicted))
    fn = sum(a == "Anomaly" and p == "Normal" for a, p in zip(actual, predicted))
    tp = sum(a == "Anomaly" and p == "Anomaly" for a, p in zip(actual, predicted))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "evaluated_records": len(actual),
        "accuracy": (tn + tp) / len(actual),
        "anomaly_precision": precision,
        "anomaly_recall": recall,
        "anomaly_f1": f1,
        "confusion_matrix_class_order": list(CONFUSION_MATRIX_CLASS_ORDER),
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
        "actual_label_counts": dict(Counter(actual)),
        "predicted_label_counts": dict(Counter(predicted)),
    }


def read_labels(path: Path) -> tuple[list[str], list[str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "BlockId" not in reader.fieldnames:
            raise ValueError("Dataset must contain a BlockId column")
        if "Label" not in reader.fieldnames:
            raise ValueError("Dataset must contain a Label column")
        identifiers: list[str] = []
        labels: list[str] = []
        for row in reader:
            if not row["BlockId"]:
                raise ValueError("BlockId cannot be empty")
            identifiers.append(row["BlockId"])
            labels.append(normalize_label(row["Label"]))
    if not labels:
        raise ValueError("Dataset contains no records")
    return identifiers, labels


def stratified_split(
    identifiers: Sequence[str],
    labels: Sequence[str],
    test_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    if len(identifiers) != len(labels) or not labels:
        raise ValueError("Identifiers and labels must be non-empty and aligned")
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")
    by_label: dict[str, list[str]] = {"Normal": [], "Anomaly": []}
    for identifier, label in zip(identifiers, labels):
        by_label[normalize_label(label)].append(identifier)
    rng = random.Random(seed)
    test_ids: list[str] = []
    for label in ("Normal", "Anomaly"):
        candidates = by_label[label][:]
        rng.shuffle(candidates)
        test_ids.extend(candidates[: round(len(candidates) * test_fraction)])
    test_set = set(test_ids)
    return (
        [identifier for identifier in identifiers if identifier not in test_set],
        test_ids,
    )
