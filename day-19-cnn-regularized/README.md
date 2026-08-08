# Day 19 — Deep Learning Basics: NN Experiment #4 (Regularized CNN)

Part of my #75DaysOfCode challenge.

## What it does
Builds on Day 18's CNN by adding Dropout and data augmentation
(random rotation/translation/zoom) to fight overfitting.

## Results
- Test accuracy: 98.06% (up from Day 18's plain CNN: 97.78%)
- Train accuracy (88%) came in below validation accuracy (95%) -
  dropout is active during training but disabled at test time, so this
  gap confirms regularization is working as intended, not overfitting.

## Files
- day19_cnn_regularized.py — main script
- day19_training_curves.png — loss/accuracy over epochs
- day19_confusion_matrix.png — test set confusion matrix
- day19_augmentation_examples.png — same digit under random augmentation
