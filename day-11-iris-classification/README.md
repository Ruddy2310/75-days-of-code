# Day 11 — Classic ML Mini-Project #6: Iris Flower Classification

Part of my #75DaysOfCode challenge.

## What it does
Classifies iris flowers into 3 species (setosa, versicolor, virginica)
using Decision Tree vs Gaussian Naive Bayes, then compares performance
and visualizes decision boundaries.

## Results
- Decision Tree test accuracy: 93.3%
- Naive Bayes test accuracy: 96.7%
- Both models: 95.3% 5-fold CV accuracy
- Uses sklearn's built-in iris dataset, no download needed

## Files
- day11_iris_decisiontree_naivebayes.py — main script
- confusion_matrices.png — Decision Tree vs Naive Bayes confusion matrices
- decision_tree_structure.png — visualized tree splits
- decision_boundaries.png — decision boundary comparison on 2 features
