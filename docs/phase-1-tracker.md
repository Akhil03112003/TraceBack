# TraceBack Phase 1 tracker

## Step 1: Dataset inspection and validation

- **Status:** Completed.
- **Purpose:** Verify the supplied HDFS CSV artifacts and NPZ consistency before evaluation.
- **Deliverables:** Existing validation implementation and report in the
  source dataset directory: `C:\Users\akhil\Downloads\HDFS_v1`.
- **Verification/test evidence:** 575,061 unique BlockIds in each CSV; no
  duplicates or missing IDs; labels align; NPZ matches trace sequence/label
  multisets but not row order.
- **Metrics/findings:** Normal/Success = 558,223; Anomaly/Fail = 16,838.
- **Remaining issues:** NPZ has no BlockId and must not be joined positionally.
- **Next action:** Use the validated CSV label file for baseline evaluation.

## Step 2: Majority-class baseline

- **Status:** Completed.
- **Purpose:** Establish an always-normal reference with anomaly as the positive class.
- **Deliverables:** `eval/majority_baseline.py`, synthetic tests,
  `eval/results/majority_baseline.json`, and
  `docs/majority-baseline.md`.
- **Verification/test evidence:** Five synthetic unit tests passed. The
  evaluator ran successfully against all 575,061 records and the generated
  JSON was independently checked.
- **Metrics/findings:** Whole-dataset accuracy 0.9707196280; anomaly
  precision, recall, and F1 are all 0.0. Confusion matrix, in
  `[Normal, Anomaly]` order, is `[[558223, 0], [16838, 0]]`. The held-out
  115,013-record accuracy is 0.9707163538, with 3,368 anomalies.
- **Remaining issues:** This baseline has no feature use and cannot detect anomalies.
- **Next action:** Compare against later Phase 1 baselines in Step 5.

## Step 3: Deterministic rule-based baseline

- **Status:** Completed.
- **Purpose:** Build an interpretable detector from validated log-event evidence.
- **Deliverables:** `eval/rule_based_baseline.py`,
  `eval/evaluation_utils.py`, synthetic tests,
  `eval/results/rule_based_baseline.json`, and
  `docs/rule-based-baseline.md`.
- **Verification/test evidence:** Complete test suite passed with 11 tests.
  Real evaluation passed on 575,061 records. Independent checks confirmed
  source counts, split membership, confusion-matrix totals, and metric
  calculations.
- **Metrics/findings:** Selected `anomaly_associated_event_presence`: any
  count >= 1 among E6, E7, E10, E12, E13, E14, E15, E17, E18, E19, E20, E24,
  E25, E27, E28, E29. Held-out accuracy 0.9914096667, anomaly precision
  0.8782581055, recall 0.8203681710, F1 0.8483266810; confusion matrix
  `[[111262, 383], [605, 2763]]`.
- **Remaining issues:** Event presence is explainable but not causal; the
  candidate family is intentionally small and selected on one training split.
- **Next action:** Keep Step 4 unstarted and later perform comparative Phase 1 evaluation.

## Step 4: Machine-learning baseline

- **Status:** Completed.
- **Purpose:** Establish a reproducible feature-based detector for comparison.
- **Deliverables:** `eval/isolation_forest_baseline.py`,
  `eval/evaluation_utils.py`, scikit-learn dependency declaration, synthetic
  tests, `eval/results/isolation_forest_baseline.json`, and
  `docs/isolation-forest-baseline.md`.
- **Verification/test evidence:** Full suite passed with 17 tests. Real
  Isolation Forest evaluation passed on the complete dataset. Independent
  checks confirmed source counts, exact Step 2/3 test membership, threshold
  separation, confusion-matrix totals, metric arithmetic, and preservation
  of prior result artifacts.
- **Metrics/findings:** Isolation Forest with 100 estimators, max_samples
  256, random_state 42, and validation-selected score threshold
  -0.6722511721 achieved held-out accuracy 0.9845756567, anomaly precision
  0.8529672276, recall 0.5718527316, and F1 0.6846782794. Confusion matrix
  `[[111313, 332], [1442, 1926]]`.
- **Remaining issues:** Threshold selection uses labels in a training-only
  validation subset; performance is below the deterministic rule baseline.
- **Next action:** Complete comparative evaluation and final Phase 1 report in Step 5.

## Step 5: Comparative evaluation, automated tests, and final Phase 1 report

- **Status:** Completed.
- **Purpose:** Compare baseline approaches and publish Phase 1 conclusions.
- **Deliverables:** `eval/baseline_comparison.py`,
  `eval/results/baseline_comparison.json`,
  `eval/results/baseline_comparison.csv`,
  `tests/test_baseline_comparison.py`, and
  `docs/phase-1-final-report.md`.
- **Verification/test evidence:** Full suite passed with 21 tests. The
  comparison generator passed. Independent verification confirmed all source
  artifacts, split hashes where available, confusion-matrix totals, metric
  arithmetic, JSON/CSV agreement, deterministic regeneration, and
  preservation of the three individual result files.
- **Metrics/findings:** On the common 115,013-record test split, majority
  F1 was 0.0000000000, rule-based F1 was 0.8483266810, and Isolation Forest
  F1 was 0.6846782794. The rule-based detector had the highest recall
  (0.8203681710) and F1.
- **Remaining issues:** The majority artifact predates split-hash recording;
  its seed/count metadata matches the hash-bearing artifacts. Results are
  limited to HDFS_v1 and Phase 2 has not started.
- **Next action:** Begin Phase 2 design and implementation after preserving
  the final Phase 1 artifacts.
