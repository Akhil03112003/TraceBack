#!/usr/bin/env python3
"""Isolation Forest anomaly baseline with training-only threshold selection."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.ensemble import IsolationForest

try:
    from .evaluation_utils import calculate_metrics, read_labels, stratified_split
except ImportError:
    from evaluation_utils import calculate_metrics, read_labels, stratified_split  # type: ignore[no-redef]

DEFAULT_OCCURRENCE_PATH = Path(
    r"C:\Users\akhil\Downloads\HDFS_v1\preprocessed\Event_occurrence_matrix.csv"
)
DEFAULT_LABEL_PATH = Path(
    r"C:\Users\akhil\Downloads\HDFS_v1\preprocessed\anomaly_label.csv"
)
DEFAULT_OUTPUT_PATH = Path("eval") / "results" / "isolation_forest_baseline.json"
MODEL_RANDOM_STATE = 42
VALIDATION_SEED = 43
TEST_FRACTION = 0.2
VALIDATION_FRACTION = 0.2
N_ESTIMATORS = 100
MAX_SAMPLES = 256
CONTAMINATION = "auto"
N_JOBS = -1
THRESHOLD_QUANTILES = (0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50)


def _id_digest(identifiers: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(sorted(identifiers)).encode("utf-8")).hexdigest()


def _read_features(
    path: Path, expected_ids: set[str]
) -> tuple[list[str], np.ndarray, tuple[str, ...]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = tuple(reader.fieldnames or ())
        event_ids = tuple(column for column in columns if column.startswith("E"))
        if "BlockId" not in columns or "Label" not in columns:
            raise ValueError("Occurrence data must contain BlockId and Label columns")
        if not event_ids:
            raise ValueError("Occurrence data contains no event-count columns")
        if {"BlockId", "Label", "Type"} & set(event_ids):
            raise ValueError("Feature columns overlap excluded metadata")
        identifiers: list[str] = []
        values: list[list[float]] = []
        seen: set[str] = set()
        for row_number, row in enumerate(reader, start=2):
            identifier = row["BlockId"]
            if not identifier or identifier in seen:
                raise ValueError(f"Missing or duplicate BlockId at row {row_number}")
            seen.add(identifier)
            row_values: list[float] = []
            for event_id in event_ids:
                raw = row[event_id]
                try:
                    value = float(raw)
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        f"Non-numeric feature {event_id!r} at row {row_number}"
                    ) from error
                if not math.isfinite(value) or value < 0:
                    raise ValueError(f"Invalid feature {event_id!r} at row {row_number}")
                row_values.append(value)
            identifiers.append(identifier)
            values.append(row_values)
    if seen != expected_ids:
        raise ValueError("Occurrence and label BlockId sets do not match")
    return identifiers, np.asarray(values, dtype=np.float32), event_ids


def _validation_split(
    train_ids: Sequence[str], labels_by_id: dict[str, str]
) -> tuple[list[str], list[str]]:
    train_labels = [labels_by_id[identifier] for identifier in train_ids]
    return stratified_split(
        train_ids,
        train_labels,
        test_fraction=VALIDATION_FRACTION,
        seed=VALIDATION_SEED,
    )


def _select_threshold(
    scores: np.ndarray,
    labels: Sequence[str],
) -> tuple[float, dict[str, object], list[dict[str, object]]]:
    if len(scores) != len(labels) or not labels:
        raise ValueError("Validation scores and labels must be non-empty and aligned")
    candidates: list[dict[str, object]] = []
    for quantile in THRESHOLD_QUANTILES:
        threshold = float(np.quantile(scores, quantile))
        predictions = ["Anomaly" if score <= threshold else "Normal" for score in scores]
        metrics = calculate_metrics(labels, predictions)
        candidates.append(
            {"quantile": quantile, "threshold": threshold, "metrics": metrics}
        )
    selected = max(
        candidates,
        key=lambda item: (
            item["metrics"]["anomaly_f1"],
            item["metrics"]["anomaly_recall"],
            item["metrics"]["anomaly_precision"],
            -item["metrics"]["false_positives"],
        ),
    )
    return float(selected["threshold"]), selected, candidates


def evaluate_dataset(
    occurrence_path: Path,
    label_path: Path,
    output_path: Path,
) -> dict[str, object]:
    started = time.perf_counter()
    label_ids, labels = read_labels(label_path)
    labels_by_id = dict(zip(label_ids, labels))
    train_ids, test_ids = stratified_split(
        label_ids, labels, TEST_FRACTION, MODEL_RANDOM_STATE
    )
    feature_ids, features, event_ids = _read_features(
        occurrence_path, set(label_ids)
    )
    index_by_id = {identifier: index for index, identifier in enumerate(feature_ids)}
    train_indices = [index_by_id[identifier] for identifier in train_ids]
    test_indices = [index_by_id[identifier] for identifier in test_ids]
    validation_train_ids, validation_ids = _validation_split(
        train_ids, labels_by_id
    )
    fit_indices = [index_by_id[identifier] for identifier in validation_train_ids]
    validation_indices = [index_by_id[identifier] for identifier in validation_ids]
    model = IsolationForest(
        n_estimators=N_ESTIMATORS,
        max_samples=MAX_SAMPLES,
        contamination=CONTAMINATION,
        random_state=MODEL_RANDOM_STATE,
        n_jobs=N_JOBS,
    )
    fit_started = time.perf_counter()
    model.fit(features[fit_indices])
    fit_seconds = time.perf_counter() - fit_started
    validation_scores = model.score_samples(features[validation_indices])
    validation_labels = [labels_by_id[identifier] for identifier in validation_ids]
    threshold, selected_threshold, threshold_candidates = _select_threshold(
        validation_scores, validation_labels
    )
    prediction_started = time.perf_counter()
    test_scores = model.score_samples(features[test_indices])
    test_predictions = [
        "Anomaly" if score <= threshold else "Normal" for score in test_scores
    ]
    prediction_seconds = time.perf_counter() - prediction_started
    test_labels = [labels_by_id[identifier] for identifier in test_ids]
    result: dict[str, object] = {
        "method": "Isolation Forest unsupervised baseline",
        "target_positive_class": "Anomaly",
        "feature_source": str(occurrence_path),
        "features": list(event_ids),
        "excluded_columns": ["BlockId", "Label", "Type"],
        "model": {
            "class": "sklearn.ensemble.IsolationForest",
            "n_estimators": N_ESTIMATORS,
            "max_samples": MAX_SAMPLES,
            "contamination": CONTAMINATION,
            "random_state": MODEL_RANDOM_STATE,
            "n_jobs": N_JOBS,
        },
        "threshold_selection": {
            "method": "training-only validation score quantiles",
            "validation_seed": VALIDATION_SEED,
            "validation_fraction_of_training": VALIDATION_FRACTION,
            "candidate_quantiles": list(THRESHOLD_QUANTILES),
            "selected": selected_threshold,
            "candidates": threshold_candidates,
            "held_out_test_labels_used": False,
        },
        "dataset": {
            "label_path": str(label_path),
            "occurrence_path": str(occurrence_path),
            "records": len(labels),
            "label_counts": dict(Counter(labels)),
        },
        "split": {
            "policy": "Step 2 exact stratified split; validation split is nested within training only",
            "seed": MODEL_RANDOM_STATE,
            "test_fraction": TEST_FRACTION,
            "train_records": len(train_ids),
            "test_records": len(test_ids),
            "train_id_sha256": _id_digest(train_ids),
            "test_id_sha256": _id_digest(test_ids),
            "fit_records": len(fit_indices),
            "validation_records": len(validation_indices),
            "test_label_counts": dict(Counter(test_labels)),
        },
        "fit": {
            "fit_seconds": fit_seconds,
            "prediction_seconds": prediction_seconds,
            "fit_used_labels": False,
        },
        "held_out_test": calculate_metrics(test_labels, test_predictions),
        "runtime_seconds": time.perf_counter() - started,
        "limitations": [
            "The score threshold is selected with labels on a validation subset inside training.",
            "The model uses only event counts and does not establish causal explanations.",
            "No model pickle is persisted; rerunning refits the fixed configuration.",
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
    args = parser.parse_args()
    print(json.dumps(evaluate_dataset(args.occurrence, args.labels, args.output), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
