# Day 12 — Classic ML Mini-Project #7: Ensemble Methods Comparison

Part of my #75DaysOfCode challenge.

## What it does
Compares three ensemble learning strategies on the same binary
classification problem: Random Forest (bagging), AdaBoost (boosting),
and Gradient Boosting.

## Results
- Random Forest: 94.0% accuracy, 0.977 ROC-AUC
- Gradient Boosting: 93.0% accuracy, 0.972 ROC-AUC
- AdaBoost: 84.5% accuracy, 0.927 ROC-AUC
- Uses sklearn's make_classification (synthetic, fully reproducible)

## Files
- day12_ensemble_methods_comparison.py — main script
- roc_curve_comparison.png — ROC curves for all 3 models
- confusion_matrices.png — side-by-side confusion matrices
- feature_importance_comparison.png — top features per model
- performance_comparison.png — accuracy/F1/AUC bar chart
