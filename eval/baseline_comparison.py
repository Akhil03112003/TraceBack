"""Verify and consolidate the existing Phase 1 baseline result artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

BASELINES = (
    ("majority", "majority_baseline.json"),
    ("rule_based", "rule_based_baseline.json"),
    ("isolation_forest", "isolation_forest_baseline.json"),
)
CLASS_ORDER = ["Normal", "Anomaly"]
METRIC_KEYS = (
    "accuracy",
    "anomaly_precision",
    "anomaly_recall",
    "anomaly_f1",
)
CSV_FIELDS = (
    "Baseline",
    "Accuracy",
    "Anomaly precision",
    "Anomaly recall",
    "Anomaly F1",
    "TN",
    "FP",
    "FN",
    "TP",
    "Test sample count",
    "Test split hash",
    "Key configuration or threshold",
)


class ComparisonValidationError(ValueError):
    """Raised when a baseline artifact cannot be safely compared."""


def _require(mapping: dict[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise ComparisonValidationError(f"Missing required field {context}.{key}")
    return mapping[key]


def _metric_values(matrix: list[list[int]]) -> dict[str, float]:
    tn, fp = matrix[0]
    fn, tp = matrix[1]
    total = tn + fp + fn + tp
    if total <= 0:
        raise ComparisonValidationError("Confusion matrix has no evaluated records")
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "accuracy": (tn + tp) / total,
        "anomaly_precision": precision,
        "anomaly_recall": recall,
        "anomaly_f1": f1,
    }


def verify_artifact(name: str, artifact: dict[str, Any]) -> dict[str, Any]:
    split = _require(artifact, "split", name)
    test = _require(artifact, "held_out_test", name)
    dataset = _require(artifact, "dataset", name)
    order = _require(test, "confusion_matrix_class_order", f"{name}.held_out_test")
    matrix = _require(test, "confusion_matrix", f"{name}.held_out_test")
    if order != CLASS_ORDER:
        raise ComparisonValidationError(f"{name} uses unexpected class order: {order}")
    if (
        not isinstance(matrix, list)
        or len(matrix) != 2
        or any(not isinstance(row, list) or len(row) != 2 for row in matrix)
        or any(not isinstance(value, int) or value < 0 for row in matrix for value in row)
    ):
        raise ComparisonValidationError(f"{name} has malformed confusion_matrix")
    expected_count = sum(value for row in matrix for value in row)
    if expected_count != test.get("evaluated_records"):
        raise ComparisonValidationError(
            f"{name} confusion matrix total does not equal evaluated_records"
        )
    recomputed = _metric_values(matrix)
    metric_consistent = all(
        math.isclose(recomputed[key], float(test.get(key)), rel_tol=1e-12, abs_tol=1e-12)
        for key in METRIC_KEYS
    )
    if not metric_consistent:
        raise ComparisonValidationError(f"{name} metrics disagree with confusion matrix")
    split_fields = ("seed", "test_fraction", "train_records", "test_records")
    if any(field not in split for field in split_fields):
        raise ComparisonValidationError(f"{name} has incomplete split metadata")
    if test["evaluated_records"] != split["test_records"]:
        raise ComparisonValidationError(f"{name} test count disagrees with split metadata")
    dataset_counts = dataset.get("label_counts")
    if dataset_counts != {"Normal": 558223, "Anomaly": 16838}:
        raise ComparisonValidationError(f"{name} has unexpected dataset label counts")
    return {
        "metric_consistency": True,
        "confusion_matrix_total": expected_count,
        "split_metadata_consistency": True,
        "test_split_hash": split.get("test_id_sha256"),
        "train_split_hash": split.get("train_id_sha256"),
        "metrics": {key: float(test[key]) for key in METRIC_KEYS},
        "confusion_matrix": matrix,
        "test_records": test["evaluated_records"],
    }


def _configuration(name: str, artifact: dict[str, Any]) -> dict[str, Any]:
    if name == "majority":
        return {
            "methodology": "Always predicts Normal; labels are evaluation targets only.",
            "feature_policy": "No features; always predicts Normal",
            "configuration": {"prediction_class": artifact["prediction_class"]},
            "runtime_seconds": None,
        }
    if name == "rule_based":
        frozen = artifact["rule_selection"]["frozen_rule"]
        return {
            "methodology": "Training-only rule selection over event-presence candidates; frozen rule evaluated on the held-out split.",
            "feature_policy": "E1-E29 event counts; excludes BlockId, Label, Type",
            "configuration": {
                "rule": frozen["name"],
                "event_ids": frozen["event_ids"],
                "threshold": frozen["threshold"],
            },
            "runtime_seconds": artifact.get("runtime_seconds"),
        }
    return {
        "methodology": "Isolation Forest fitted without labels; score threshold selected with labels on a validation subset inside training.",
        "feature_policy": "E1-E29 event counts; excludes BlockId, Label, Type",
        "configuration": {
            "model": artifact["model"],
            "threshold": artifact["threshold_selection"]["selected"],
        },
        "runtime_seconds": artifact.get("runtime_seconds"),
    }


def build_comparison(results_dir: Path) -> dict[str, Any]:
    artifacts: dict[str, dict[str, Any]] = {}
    verification: dict[str, dict[str, Any]] = {}
    for name, filename in BASELINES:
        path = results_dir / filename
        if not path.is_file():
            raise ComparisonValidationError(f"Missing result artifact: {path}")
        try:
            artifact = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ComparisonValidationError(f"Cannot read {path}: {error}") from error
        if not isinstance(artifact, dict):
            raise ComparisonValidationError(f"Result artifact is not an object: {path}")
        artifacts[name] = artifact
        verification[name] = verify_artifact(name, artifact)

    reference = verification["rule_based"]["test_split_hash"]
    hash_consistency = (
        reference is not None
        and verification["isolation_forest"]["test_split_hash"] == reference
    )
    metadata_consistency = all(
        artifacts[name]["split"]["seed"] == 42
        and artifacts[name]["split"]["test_fraction"] == 0.2
        and artifacts[name]["split"]["test_records"] == 115013
        for name, _ in BASELINES
    )
    if not hash_consistency or not metadata_consistency:
        raise ComparisonValidationError("Baseline split metadata is inconsistent")

    rows = []
    consolidated_baselines = []
    for name, _ in BASELINES:
        artifact = artifacts[name]
        checked = verification[name]
        config = _configuration(name, artifact)
        consolidated_baselines.append(
            {
                "name": name,
                "method": artifact["method"],
                "methodology": config["methodology"],
                "limitations": artifact.get("limitations", []),
                "feature_policy": config["feature_policy"],
                "metrics": checked["metrics"],
                "true_negatives": checked["confusion_matrix"][0][0],
                "false_positives": checked["confusion_matrix"][0][1],
                "false_negatives": checked["confusion_matrix"][1][0],
                "true_positives": checked["confusion_matrix"][1][1],
                "confusion_matrix": checked["confusion_matrix"],
                "configuration": config["configuration"],
                "runtime_seconds": config["runtime_seconds"],
                "verification": {
                    **checked,
                    "split_hash_consistency": (
                        checked["test_split_hash"] in (None, reference)
                    ),
                    "passed_all_verification_checks": True,
                },
            }
        )
        rows.append(
            {
                "Baseline": name,
                "Accuracy": checked["metrics"]["accuracy"],
                "Anomaly precision": checked["metrics"]["anomaly_precision"],
                "Anomaly recall": checked["metrics"]["anomaly_recall"],
                "Anomaly F1": checked["metrics"]["anomaly_f1"],
                "TN": checked["confusion_matrix"][0][0],
                "FP": checked["confusion_matrix"][0][1],
                "FN": checked["confusion_matrix"][1][0],
                "TP": checked["confusion_matrix"][1][1],
                "Test sample count": checked["test_records"],
                "Test split hash": checked["test_split_hash"] or "unavailable_in_source_artifact",
                "Key configuration or threshold": json.dumps(
                    config["configuration"], sort_keys=True, separators=(",", ":")
                ),
            }
        )
    return {
        "schema_version": 1,
        "dataset": {
            "name": "HDFS_v1",
            "records": 575061,
            "label_counts": {"Normal": 558223, "Anomaly": 16838},
            "class_order": CLASS_ORDER,
        },
        "evaluation_split": {
            "test_records": 115013,
            "test_label_counts": {"Normal": 111645, "Anomaly": 3368},
            "seed": 42,
            "test_fraction": 0.2,
            "test_id_sha256": reference,
            "train_id_sha256": verification["rule_based"]["train_split_hash"],
            "hashes_available_in_source_artifacts": {
                "majority": False,
                "rule_based": True,
                "isolation_forest": True,
            },
            "majority_split_hash_note": (
                "The majority artifact stores split seed/counts but no ID hash; "
                "its split metadata matches the hash-bearing artifacts."
            ),
        },
        "baselines": consolidated_baselines,
        "verification_summary": {
            "all_three_artifacts_present_once": len(consolidated_baselines) == 3,
            "split_metadata_consistent": metadata_consistency,
            "hash_bearing_artifacts_match": hash_consistency,
            "all_baseline_checks_passed": True,
        },
        "csv_rows": rows,
    }


def write_outputs(results_dir: Path, output_json: Path, output_csv: Path) -> dict[str, Any]:
    comparison = build_comparison(results_dir)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps({key: value for key, value in comparison.items() if key != "csv_rows"}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(comparison["csv_rows"])
    return comparison


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("eval/results"))
    parser.add_argument("--output-json", type=Path, default=Path("eval/results/baseline_comparison.json"))
    parser.add_argument("--output-csv", type=Path, default=Path("eval/results/baseline_comparison.csv"))
    args = parser.parse_args()
    comparison = write_outputs(args.results_dir, args.output_json, args.output_csv)
    print(json.dumps(comparison["verification_summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
