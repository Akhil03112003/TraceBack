# Majority-class baseline

The Phase 1 Step 2 baseline predicts `Normal` for every record. It does not
inspect `BlockId`, features, or ground-truth labels while predicting. The
validated CSV label file is used only as the evaluation target.

## Label policy

The evaluator normalizes `Normal` and `Success` to `Normal`, and `Anomaly` and
`Fail` to `Anomaly`. The actual HDFS label file uses `Normal` and `Anomaly`.
Anomaly is the positive class.

The confusion matrix uses class order `[Normal, Anomaly]`:

```text
[[TN, FP],
 [FN, TP]]
```

When no anomaly is predicted, anomaly precision, recall, and F1 are explicitly
reported as `0.0`.

## Evaluation protocol

The whole dataset is evaluated for the reference score. A deterministic
stratified 80/20 split is also recorded for future baseline comparisons,
using seed `42`. No training or tuning is performed, so the baseline's
held-out score is produced directly from the test labels.

The evaluator reads only `anomaly_label.csv`; it does not join NPZ records by
position and does not copy or modify the source dataset.

## Reproduction

From the TraceBack workspace:

```powershell
python -m unittest discover -s tests -v
python eval\majority_baseline.py
```

The evaluator defaults to
`C:\Users\akhil\Downloads\HDFS_v1\preprocessed\anomaly_label.csv` and writes
to `eval/results/majority_baseline.json`. Override either location with
`--dataset` or `--output` when using a different dataset or output location.

The generated JSON contains the actual record counts, split metadata,
confusion matrices, and metrics. It is the source for the values reported
below.

## Actual result

The evaluation used 575,061 records: 558,223 normal and 16,838 anomalous.
The whole-dataset metrics were:

- Accuracy: **0.9707196280** (97.07196280%)
- Anomaly precision: **0.0**
- Anomaly recall: **0.0**
- Anomaly F1: **0.0**
- Confusion matrix: `[[558223, 0], [16838, 0]]`

The fixed stratified held-out test split contains 115,013 records (111,645
normal and 3,368 anomalous). Its accuracy is **0.9707163538**; anomaly
precision, recall, and F1 are all **0.0**.

## Limitations

This predictor is intentionally non-discriminating: it cannot identify any
anomaly. Its accuracy reflects class imbalance and should not be interpreted
as anomaly-detection quality. It is a reference point for the later
deterministic and machine-learning baselines.
