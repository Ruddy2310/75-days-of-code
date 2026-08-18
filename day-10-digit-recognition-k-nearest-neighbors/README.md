# Day 10 — Classic ML Mini-Project #5: Handwritten Digit Recognition

Part of my #75DaysOfCode challenge.

## What it does
Classifies handwritten digits (0-9) from 8x8 pixel images using K-Nearest
Neighbors and Support Vector Machine, then compares their performance.

## Results
- SVM accuracy: 97.5%
- KNN 5-fold CV accuracy: 94.4%
- SVM 5-fold CV accuracy: 94.9%
- Uses sklearn's built-in digits dataset, no download needed

## Files
- day10_digit_recognition_knn_svm.py — main script
- sample_digits.png — example digit images from the dataset
- confusion_matrices.png — KNN vs SVM confusion matrices
- misclassified_examples.png — digits the SVM got wrong
