# Day 14 — Classic ML Mini-Project #9: Anomaly Detection

Part of my #75DaysOfCode challenge.

## What it does
Compares two unsupervised anomaly detection approaches on a synthetic
fraud detection scenario: Isolation Forest (tree-based, efficient in
high dimensions) vs Local Outlier Factor (density-based, sensitive to
local patterns).

## Results
- Isolation Forest: 99% precision, 99% recall, 1.0 ROC-AUC
- Local Outlier Factor: 3% precision, 3% recall, 0.53 ROC-AUC
- Teaching point: Isolation Forest excels at separated anomalies; LOF
  needs truly sparse neighborhoods to work well.

## Files
- day14_anomaly_detection.py — main script (synthetic 1000-sample dataset)
- anomaly_detection_comparison.png — 4-panel PCA visualization (predictions + scores)
- confusion_matrices.png — side-by-side confusion matrices
- roc_curve_comparison.png — ROC curves for both models
