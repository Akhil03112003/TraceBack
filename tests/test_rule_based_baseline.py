import tempfile
import unittest
from pathlib import Path

from eval.evaluation_utils import stratified_split
from eval.majority_baseline import stratified_split as majority_split
from eval.rule_based_baseline import (
    CANDIDATE_RULES,
    FeatureRecord,
    _read_occurrence_records,
    _select_rule,
)


def record(identifier: str, **counts: int) -> FeatureRecord:
    event_counts = {f"E{i}": 0 for i in range(1, 30)}
    event_counts.update(counts)
    return FeatureRecord(identifier, "Normal", event_counts)


class RuleBasedBaselineTests(unittest.TestCase):
    def test_each_rule_threshold_below_at_and_above(self) -> None:
        for rule in CANDIDATE_RULES:
            event_id = rule.event_ids[0]
            below = record("below")
            at = record("at", **{event_id: rule.threshold})
            above = record("above", **{event_id: rule.threshold + 1})
            self.assertFalse(rule.predicts_anomaly(below), rule.name)
            self.assertTrue(rule.predicts_anomaly(at), rule.name)
            self.assertTrue(rule.predicts_anomaly(above), rule.name)

    def test_combined_rule_and_determinism(self) -> None:
        records = [
            record("n1"),
            FeatureRecord("a1", "Anomaly", record("a1", E20=1).counts),
            FeatureRecord("a2", "Anomaly", record("a2", E27=2).counts),
        ]
        train_ids = {"n1", "a1", "a2"}
        first = _select_rule(records, train_ids)
        second = _select_rule(records, train_ids)
        self.assertEqual(first, second)
        self.assertTrue(first[0].predicts_anomaly(records[1]))
        self.assertTrue(first[0].predicts_anomaly(records[2]))

    def test_split_matches_step_two_utility(self) -> None:
        ids = [f"n{i}" for i in range(20)] + [f"a{i}" for i in range(8)]
        labels = ["Normal"] * 20 + ["Anomaly"] * 8
        self.assertEqual(
            stratified_split(ids, labels, seed=42),
            majority_split(ids, labels, seed=42),
        )

    def test_test_labels_do_not_affect_training_selection(self) -> None:
        records = [
            record("train-normal"),
            record("train-anomaly", E20=1),
            record("test-one", E20=1),
        ]
        records[1] = FeatureRecord("train-anomaly", "Anomaly", records[1].counts)
        records[2] = FeatureRecord("test-one", "Normal", records[2].counts)
        first = _select_rule(records, {"train-normal", "train-anomaly"})
        records[2] = FeatureRecord("test-one", "Anomaly", records[2].counts)
        second = _select_rule(records, {"train-normal", "train-anomaly"})
        self.assertEqual(first, second)

    def test_feature_schema_excludes_metadata_and_rejects_invalid_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "occurrence.csv"
            path.write_text(
                "BlockId,Label,Type,E1,E2\nb1,Success,,1,0\n",
                encoding="utf-8",
            )
            rows, event_ids = _read_occurrence_records(path)
            self.assertEqual(event_ids, ("E1", "E2"))
            self.assertNotIn("Label", event_ids)
            self.assertNotIn("Type", event_ids)
            self.assertEqual(rows[0].counts, {"E1": 1, "E2": 0})

            for value in ("bad", "NaN", "inf", "-1", "1.5"):
                path.write_text(
                    f"BlockId,Label,Type,E1\nb1,Success,,{value}\n",
                    encoding="utf-8",
                )
                with self.assertRaises(ValueError):
                    _read_occurrence_records(path)

    def test_missing_columns_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "occurrence.csv"
            path.write_text("BlockId,Type,E1\nb1,,1\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                _read_occurrence_records(path)


if __name__ == "__main__":
    unittest.main()
