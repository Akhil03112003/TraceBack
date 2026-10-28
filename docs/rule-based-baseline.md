# Deterministic rule-based baseline

Phase 1 Step 3 uses `Event_occurrence_matrix.csv` and predicts an anomaly
from transparent event-count rules. It does not use the NPZ archive or join
records by row position.

## Features and exclusions

The evaluator reads the `E1` through `E29` event-count columns. It explicitly
excludes:

- `BlockId`, which is an identifier;
- `Label`, which is the evaluation target;
- `Type`, which is label-associated metadata and target leakage.

Counts must be finite, non-negative integers. Missing columns, non-numeric
values, infinite values, negative counts, and fractional counts produce a
validation error rather than a normal prediction.

## Split and rule selection

The split is exactly the Step 2 split: per-label ID lists are shuffled using
`random.Random(42)`, and 20% of each class is held out using `round`. The
implementation is shared with the majority baseline. The generated result
contains SHA-256 digests of the train and test ID sets for reproducibility.

Three small, predefined event-presence candidates were scored using training
records only. Selection prioritizes anomaly F1, then anomaly recall, anomaly
precision, and fewer false positives. The held-out labels are not consulted
until the selected rule is frozen.

The selected rule is `anomaly_associated_event_presence`:

> Predict `Anomaly` if any of `E6, E7, E10, E12, E13, E14, E15, E17, E18,
> E19, E20, E24, E25, E27, E28, E29` has a count greater than or equal to 1.

The threshold is the semantic event-presence threshold (`count >= 1`), not a
threshold tuned on held-out data.

## Reproduction

From `D:\Projects\Traceback`:

```powershell
python -m unittest discover -s tests -v
python eval\rule_based_baseline.py `
  --occurrence C:\Users\akhil\Downloads\HDFS_v1\preprocessed\Event_occurrence_matrix.csv `
  --labels C:\Users\akhil\Downloads\HDFS_v1\preprocessed\anomaly_label.csv `
  --output eval\results\rule_based_baseline.json
```

## Results

The source contains 575,061 records: 558,223 normal and 16,838 anomalous.
The fixed held-out test contains 115,013 records: 111,645 normal and 3,368
anomalous.

### Training selection data

- Records: 460,048
- Accuracy: **0.9913661183**
- Anomaly precision: **0.8724121706**
- Anomaly recall: **0.8259094284**
- Anomaly F1: **0.8485241400**
- Confusion matrix, class order `[Normal, Anomaly]`:
  `[[444951, 1627], [2345, 11125]]`

### Held-out test

- Records: 115,013
- Accuracy: **0.9914096667**
- Anomaly precision: **0.8782581055**
- Anomaly recall: **0.8203681710**
- Anomaly F1: **0.8483266810**
- True negatives: **111,262**
- False positives: **383**
- False negatives: **605**
- True positives: **2,763**
- Confusion matrix: `[[111262, 383], [605, 2763]]`

### Whole dataset reference

- Records: 575,061
- Accuracy: **0.9913748281**
- Anomaly precision: **0.8735690024**
- Anomaly recall: **0.8248010453**
- Anomaly F1: **0.8484848485**
- Confusion matrix: `[[556213, 2010], [2950, 13888]]`

Compared with the Step 2 majority baseline on the same held-out records
(accuracy **0.9707163538**, anomaly F1 **0.0**), this rule improves anomaly
recall and F1 substantially while introducing 383 normal-record false
positives.

## Limitations

The candidate family is intentionally small and event presence does not prove
causality. The rule was selected using the training portion of one fixed
split, so generalization requires later comparative evaluation. No machine
learning or tuning against the test set was performed.
