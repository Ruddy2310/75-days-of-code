# Day 13 — Classic ML Mini-Project #8: Ridge vs Lasso vs SVR Regression

Part of my #75DaysOfCode challenge.

## What it does
Compares Ridge Regression, Lasso Regression, and Support Vector
Regression (SVR) on predicting diabetes disease progression, revisiting
the Day 7 dataset with regularized and kernel-based methods.

## Results
- SVR (RBF): R2 = 0.508 (test), CV R2 = 0.429
- Lasso: R2 = 0.461 (test), CV R2 = 0.482
- Ridge: R2 = 0.454 (test), CV R2 = 0.482
- Lasso automatically zeroed out 1 of 10 features (built-in feature selection)

## Files
- day13_ridge_lasso_svr_comparison.py — main script
- predicted_vs_actual.png — prediction accuracy for all 3 models
- ridge_vs_lasso_coefficients.png — coefficient shrinkage comparison
- performance_comparison.png — R2 score bar chart
