# Isolation Forest baseline

Phase 1 Step 4 uses scikit-learn's `IsolationForest` on the event occurrence
counts in `Event_occurrence_matrix.csv`.

## Features and data separation

The model uses `E1` through `E29` as `float32` event-count features. It
excludes `BlockId`, `Label`, and `Type`; labels are never passed to
`IsolationForest.fit`.

The exact Step 2/Step 3 held-out split is reused:

- seed: `42`;
- stratified test fraction: `0.2`;
- training records: `460,048`;
- test records: `115,013`;
- test ID SHA-256:
  `9bb8bb56d14f0fd495c37db45727e913561c28250489a3c10fb9091da277a2d4`.

Within the training portion only, a second deterministic stratified split
uses seed `43`. The model is fitted on 368,038 records and score thresholds
are selected using the 92,010-record validation subset. The held-out test is
not used for fitting, threshold selection, or parameter selection.

## Model and threshold

Configuration:

- `n_estimators=100`
- `max_samples=256`
- `contamination="auto"`
- `random_state=42`
- `n_jobs=-1`

The model's default `predict` threshold is not used. Instead, lower
`score_samples` values are treated as more anomalous. Candidate validation
quantiles are `[0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50]`.
Selection maximizes anomaly F1, then recall, precision, and minimizes false
positives.

The selected threshold is:

- validation quantile: **0.02**
- score threshold: **-0.6722511721**
- selection validation F1: **0.6984126984**

This is an unsupervised model with validation-label-based threshold
selection, not a fully label-free pipeline.

## Reproduction

From `D:\Projects\Traceback`:

```powershell
python -m unittest discover -s tests -v
python -m eval.isolation_forest_baseline `
  --occurrence C:\Users\akhil\Downloads\HDFS_v1\preprocessed\Event_occurrence_matrix.csv `
  --labels C:\Users\akhil\Downloads\HDFS_v1\preprocessed\anomaly_label.csv `
  --output eval\results\isolation_forest_baseline.json
```

## Held-out results

On the same 115,013 test records used by the earlier baselines:

- Accuracy: **0.9845756567**
- Anomaly precision: **0.8529672276**
- Anomaly recall: **0.5718527316**
- Anomaly F1: **0.6846782794**
- True negatives: **111,313**
- False positives: **332**
- False negatives: **1,442**
- True positives: **1,926**
- Confusion matrix, class order `[Normal, Anomaly]`:
  `[[111313, 332], [1442, 1926]]`

Fit time was **0.364 seconds** and test prediction time was **0.447 seconds**
in the recorded run. Runtime varies by machine and thread scheduling.

Compared with the same-test results:

| Baseline | Accuracy | Anomaly precision | Anomaly recall | Anomaly F1 |
|---|---:|---:|---:|---:|
| Majority class | 0.970716 | 0.000000 | 0.000000 | 0.000000 |
| Rule-based | 0.991410 | 0.878258 | 0.820368 | 0.848327 |
| Isolation Forest | 0.984576 | 0.852967 | 0.571853 | 0.684678 |

Isolation Forest detects substantially more anomalies than the majority
baseline, but its recall and F1 are below the deterministic rule baseline and
it produces 332 false positives.

## Limitations

The model is unsupervised only during fitting; threshold selection uses labels
from the training validation subset. Event counts alone do not provide causal
explanations. No model pickle is saved, avoiding an untrusted-model loading
surface; rerunning refits the fixed configuration.
