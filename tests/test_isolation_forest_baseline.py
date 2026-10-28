import tempfile
import unittest
from pathlib import Path

import numpy as np
from sklearn.ensemble import IsolationForest

from eval.evaluation_utils import stratified_split
from eval.isolation_forest_baseline import (
    _read_features,
    _select_threshold,
)


class IsolationForestBaselineTests(unittest.TestCase):
    def test_feature_columns_exclude_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "occurrence.csv"
            path.write_text(
                "BlockId,Label,Type,E1,E2\nb1,Success,,1,0\n",
                encoding="utf-8",
            )
            identifiers, features, event_ids = _read_features(path, {"b1"})
            self.assertEqual(identifiers, ["b1"])
            self.assertEqual(event_ids, ("E1", "E2"))
            self.assertEqual(features.tolist(), [[1.0, 0.0]])

    def test_invalid_features_are_rejected(self) -> None:
        for value in ("bad", "NaN", "inf", "-1"):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "occurrence.csv"
                path.write_text(
                    f"BlockId,Label,Type,E1\nb1,Success,,{value}\n",
                    encoding="utf-8",
                )
                with self.assertRaises(ValueError):
                    _read_features(path, {"b1"})

    def test_split_is_reproducible(self) -> None:
        identifiers = [f"n{i}" for i in range(20)] + [f"a{i}" for i in range(8)]
        labels = ["Normal"] * 20 + ["Anomaly"] * 8
        self.assertEqual(
            stratified_split(identifiers, labels, seed=42),
            stratified_split(identifiers, labels, seed=42),
        )

    def test_fixed_seed_model_predictions_are_deterministic(self) -> None:
        features = np.asarray(
            [[0, 0], [0, 1], [1, 0], [1, 1], [10, 10], [11, 10]],
            dtype=np.float32,
        )
        first = IsolationForest(
            n_estimators=20, max_samples=4, random_state=42, n_jobs=1
        ).fit(features)
        second = IsolationForest(
            n_estimators=20, max_samples=4, random_state=42, n_jobs=1
        ).fit(features)
        self.assertTrue(np.array_equal(first.predict(features), second.predict(features)))
        self.assertTrue(np.array_equal(first.score_samples(features), second.score_samples(features)))

    def test_threshold_selection_uses_only_supplied_validation_data(self) -> None:
        scores = np.asarray([-1.0, -0.8, -0.1, 0.1], dtype=np.float32)
        labels = ["Anomaly", "Anomaly", "Normal", "Normal"]
        first = _select_threshold(scores, labels)
        second = _select_threshold(scores, ["Normal"] * len(labels))
        self.assertNotEqual(first[1]["metrics"], second[1]["metrics"])
        self.assertEqual(len(first[2]), 9)

    def test_no_anomaly_prediction_metrics_are_zero(self) -> None:
        from eval.evaluation_utils import calculate_metrics

        metrics = calculate_metrics(["Normal", "Anomaly"], ["Normal", "Normal"])
        self.assertEqual(metrics["anomaly_precision"], 0.0)
        self.assertEqual(metrics["anomaly_recall"], 0.0)
        self.assertEqual(metrics["anomaly_f1"], 0.0)


if __name__ == "__main__":
    unittest.main()
