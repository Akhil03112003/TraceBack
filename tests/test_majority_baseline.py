import unittest

from eval.majority_baseline import (
    calculate_metrics,
    majority_predictions,
    stratified_split,
)


class MajorityBaselineTests(unittest.TestCase):
    def test_all_predictions_are_normal(self) -> None:
        self.assertEqual(
            majority_predictions(["Normal", "Anomaly", "Success"]),
            ["Normal", "Normal", "Normal"],
        )

    def test_metrics_and_confusion_matrix(self) -> None:
        metrics = calculate_metrics(
            ["Normal", "Normal", "Anomaly", "Anomaly"],
            ["Normal", "Anomaly", "Normal", "Anomaly"],
        )
        self.assertEqual(metrics["confusion_matrix"], [[1, 1], [1, 1]])
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["anomaly_precision"], 0.5)
        self.assertEqual(metrics["anomaly_recall"], 0.5)
        self.assertEqual(metrics["anomaly_f1"], 0.5)

    def test_zero_anomaly_predictions_are_zero(self) -> None:
        metrics = calculate_metrics(
            ["Normal", "Anomaly"], ["Normal", "Normal"]
        )
        self.assertEqual(metrics["anomaly_precision"], 0.0)
        self.assertEqual(metrics["anomaly_recall"], 0.0)
        self.assertEqual(metrics["anomaly_f1"], 0.0)
        self.assertEqual(metrics["confusion_matrix"], [[1, 0], [1, 0]])

    def test_invalid_and_empty_inputs(self) -> None:
        with self.assertRaises(ValueError):
            majority_predictions([])
        with self.assertRaises(ValueError):
            calculate_metrics([], [])
        with self.assertRaises(ValueError):
            calculate_metrics(["Normal"], ["Normal", "Anomaly"])
        with self.assertRaises(ValueError):
            calculate_metrics(["Unknown"], ["Normal"])

    def test_split_is_deterministic_and_stratified(self) -> None:
        identifiers = [f"n{i}" for i in range(10)] + [f"a{i}" for i in range(4)]
        labels = ["Normal"] * 10 + ["Anomaly"] * 4
        first = stratified_split(identifiers, labels, seed=7)
        second = stratified_split(identifiers, labels, seed=7)
        self.assertEqual(first, second)
        test_ids = set(first[1])
        self.assertEqual(sum(identifier.startswith("n") for identifier in test_ids), 2)
        self.assertEqual(sum(identifier.startswith("a") for identifier in test_ids), 1)


if __name__ == "__main__":
    unittest.main()
