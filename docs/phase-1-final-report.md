# TraceBack Phase 1 final report

## 1. Executive summary

Phase 1 established and independently verified three anomaly-detection
baselines on the validated HDFS_v1 preprocessed data. All three use the same
115,013-record held-out test split. The deterministic rule-based baseline is
the strongest on held-out anomaly recall and F1 in this evaluation:

- Accuracy: **0.9914096667**
- Anomaly precision: **0.8782581055**
- Anomaly recall: **0.8203681710**
- Anomaly F1: **0.8483266810**

The majority baseline demonstrates that 97.07% accuracy is not useful anomaly
performance on this imbalanced dataset. Isolation Forest improves substantially
over majority-class prediction but is below the rule-based baseline on recall
and F1.

## 2. Dataset overview and label distribution

The read-only HDFS_v1 preprocessed dataset contains 575,061 block records:

- Normal: 558,223
- Anomaly: 16,838

The occurrence matrix contains event-count features `E1` through `E29`.
The validated label mapping is `Normal`/`Success` to normal and
`Anomaly`/`Fail` to anomalous.

## 3. Data integrity and leakage precautions

The earlier validation established identical `BlockId` sets across the three
CSV artifacts, with no duplicates or missing IDs. Phase 1 evaluations use
CSV data and do not join NPZ records by row position.

All model features exclude `BlockId`, `Label`, and `Type`. The raw 1.58 GB
`HDFS.log` was not loaded. The source dataset was not copied or modified.

The majority predictor uses no features. The rule-based candidate selection
uses training labels only. Isolation Forest fitting does not receive labels,
but its score threshold is selected using labels from a validation subset
inside the training portion. Therefore the complete Isolation Forest pipeline
is not fully label-free.

## 4. Evaluation methodology and reproducibility

All baselines use the same stratified split policy:

- random seed: `42`;
- test fraction: `0.2`;
- training records: 460,048;
- held-out test records: 115,013;
- test labels: 111,645 normal and 3,368 anomalous.

The test ID SHA-256 is:

```text
9bb8bb56d14f0fd495c37db45727e913561c28250489a3c10fb9091da277a2d4
```

The training ID SHA-256 is:

```text
bd6f609d5943979202bbf09b8fdd3a911d901015473eaa2e39116a7c93c08cb2
```

The rule-based and Isolation Forest artifacts contain these hashes. The
majority artifact predates hash recording and contains the same seed, fraction,
record counts, and test label counts; its source hash fields are therefore
reported as unavailable rather than fabricated.

The shared verifier is [baseline_comparison.py](../eval/baseline_comparison.py).
It recomputes metrics from each confusion matrix, checks matrix totals, checks
split metadata, and writes the JSON and CSV comparison artifacts.

## 5. Baseline comparison table

| Baseline | Accuracy | Anomaly precision | Anomaly recall | Anomaly F1 | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Majority class | 0.9707163538 | 0.0000000000 | 0.0000000000 | 0.0000000000 | 111,645 | 0 | 3,368 | 0 |
| Rule-based | 0.9914096667 | 0.8782581055 | 0.8203681710 | 0.8483266810 | 111,262 | 383 | 605 | 2,763 |
| Isolation Forest | 0.9845756567 | 0.8529672276 | 0.5718527316 | 0.6846782794 | 111,313 | 332 | 1,442 | 1,926 |

Machine-readable versions are [baseline_comparison.json](../eval/results/baseline_comparison.json)
and [baseline_comparison.csv](../eval/results/baseline_comparison.csv).

## 6. Confusion matrices and metric definitions

Every matrix uses class order `[Normal, Anomaly]`:

```text
[[TN, FP],
 [FN, TP]]
```

Accuracy is `(TN + TP) / total`. Anomaly precision is `TP / (TP + FP)`,
recall is `TP / (TP + FN)`, and F1 is the harmonic mean of precision and
recall. When a denominator is zero, the shared evaluation convention reports
the corresponding metric as `0.0`.

The verified matrices are:

```text
Majority:
[[111645, 0],
 [  3368, 0]]

Rule-based:
[[111262, 383],
 [   605, 2763]]

Isolation Forest:
[[111313, 332],
 [  1442, 1926]]
```

Each matrix sums to 115,013.

## 7. Interpretation of results

The majority predictor has high accuracy because normal records dominate, but
it has zero anomaly recall and F1. It is a useful lower-bound reference only.

The rule-based detector is the strongest measured baseline on anomaly recall
and F1. Its event-presence rule is transparent and produces 383 false
positives, but the selected event family may be specific to HDFS_v1 and does
not establish causality.

Isolation Forest produces fewer false positives than the rule-based detector
in this run (332 versus 383), but it misses more anomalies (1,442 versus 605)
and has lower recall and F1. It is unsupervised during fitting, while its
threshold is validation-label-selected.

## 8. Limitations and risks

- Results are from HDFS_v1 only and do not establish generalization to other
  systems, workloads, or datasets.
- The rule family and Isolation Forest threshold were selected using the
  training portion of one fixed split.
- The rule-based detector is interpretable but dataset-specific risk remains.
- Isolation Forest provides anomaly scores, not causal explanations.
- The majority source artifact does not contain an ID hash; its split
  consistency is verified from metadata, while the other two artifacts provide
  direct hashes.
- No production ingestion, serving, monitoring, or drift handling exists yet.

## 9. Final recommendation

Use the deterministic rule-based detector as the current Phase 1 reference
because it has the best held-out anomaly recall and F1. Retain Isolation
Forest as a complementary unsupervised score and false-positive comparison,
not as the primary detector under this configuration.

## 10. Phase 1 acceptance checklist

- [x] Dataset validation completed.
- [x] Majority-class baseline implemented and verified.
- [x] Deterministic rule-based baseline implemented and verified.
- [x] Isolation Forest baseline implemented and verified.
- [x] Same held-out split used for all baseline comparisons.
- [x] Confusion matrices and metrics independently recomputed.
- [x] JSON and CSV comparison artifacts generated.
- [x] Full automated test suite passes.
- [x] Source dataset remains outside the project and unchanged.
- [x] No Spring Boot, frontend, or Phase 2 work started.

Phase 1 is complete. Phase 2 remains not started.

## 11. Reproduction commands

From `D:\Projects\Traceback`:

```powershell
python -m unittest discover -s tests -v
python -m eval.baseline_comparison `
  --results-dir eval\results `
  --output-json eval\results\baseline_comparison.json `
  --output-csv eval\results\baseline_comparison.csv
```

The comparison command reads the three existing baseline JSON files and does
not rerun or retune any model.

## 12. Phase 2 readiness

The measurable evaluation foundation is ready for Phase 2. The next phase may
design the Spring Boot foundation, persistence, and ingestion API while
preserving the verified baseline artifacts and dataset-read-only policy.
