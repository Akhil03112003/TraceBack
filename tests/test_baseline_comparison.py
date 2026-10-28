import csv
import json
import tempfile
import unittest
from pathlib import Path

from eval.baseline_comparison import (
    ComparisonValidationError,
    build_comparison,
    verify_artifact,
    write_outputs,
)


def artifact(matrix: list[list[int]]) -> dict[str, object]:
    tn, fp = matrix[0]
    fn, tp = matrix[1]
    total = tn + fp + fn + tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "split": {
            "seed": 42,
            "test_fraction": 0.2,
            "train_records": 4,
            "test_records": total,
        },
        "dataset": {"label_counts": {"Normal": 558223, "Anomaly": 16838}},
        "held_out_test": {
            "evaluated_records": total,
            "confusion_matrix_class_order": ["Normal", "Anomaly"],
            "confusion_matrix": matrix,
            "accuracy": (tn + tp) / total,
            "anomaly_precision": precision,
            "anomaly_recall": recall,
            "anomaly_f1": f1,
        },
    }


class BaselineComparisonTests(unittest.TestCase):
    def test_metric_verification_and_malformed_result(self) -> None:
        checked = verify_artifact("synthetic", artifact([[2, 1], [0, 1]]))
        self.assertTrue(checked["metric_consistency"])
        broken = artifact([[2, 1], [0, 1]])
        broken["held_out_test"]["accuracy"] = 0.0
        with self.assertRaises(ComparisonValidationError):
            verify_artifact("synthetic", broken)

    def test_real_comparison_has_three_unique_baselines(self) -> None:
        comparison = build_comparison(Path("eval/results"))
        names = [baseline["name"] for baseline in comparison["baselines"]]
        self.assertEqual(names, ["majority", "rule_based", "isolation_forest"])
        self.assertEqual(len(set(names)), 3)
        self.assertTrue(comparison["verification_summary"]["hash_bearing_artifacts_match"])

    def test_json_and_csv_outputs_agree_and_are_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = root / "results"
            results.mkdir()
            source = Path("eval/results")
            for name in (
                "majority_baseline.json",
                "rule_based_baseline.json",
                "isolation_forest_baseline.json",
            ):
                (results / name).write_bytes((source / name).read_bytes())
            output_json = results / "comparison.json"
            output_csv = results / "comparison.csv"
            write_outputs(results, output_json, output_csv)
            first_json = output_json.read_bytes()
            first_csv = output_csv.read_bytes()
            write_outputs(results, output_json, output_csv)
            self.assertEqual(first_json, output_json.read_bytes())
            self.assertEqual(first_csv, output_csv.read_bytes())
            data = json.loads(output_json.read_text(encoding="utf-8"))
            with output_csv.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), len(data["baselines"]))
            for row, baseline in zip(rows, data["baselines"]):
                self.assertEqual(row["Baseline"], baseline["name"])
                self.assertEqual(float(row["Accuracy"]), baseline["metrics"]["accuracy"])

    def test_missing_result_fails_clearly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ComparisonValidationError):
                build_comparison(Path(directory))


if __name__ == "__main__":
    unittest.main()
