# Day 25 — Deep Learning Basics: NN Experiment #10 (Capstone)

Part of my #75DaysOfCode challenge. Final project of Phase 1's Deep Learning arc (Days 16-25).

## What it does
Combines everything from the Deep Learning arc into one production-style
pipeline: Functional API, Batch Normalization, Dropout, ModelCheckpoint
(saves best model automatically), ReduceLROnPlateau (adaptive learning
rate), and EarlyStopping.

## Architecture
Conv2D(32)+BN -> Pool -> Dropout -> Conv2D(64)+BN -> Dropout ->
Flatten -> Dense(64)+BN -> Dropout -> Dense(10, softmax)

## Results
- Test accuracy: 99.17% (best of the entire Days 16-25 Deep Learning arc)
- Learning rate auto-reduced twice as validation loss plateaued
- Full accuracy progression: 96.1% (from-scratch) -> 96.67% (Keras) ->
  97.78% (CNN) -> 98.06% (regularized CNN) -> 99.17% (capstone)

## Files
- day25_capstone_cnn.py — main script
- training_curves.png — loss, accuracy, and learning rate over epochs
- confusion_matrix.png — test set confusion matrix
- phase1_dl_journey.png — accuracy comparison across all 5 CNN/NN experiments
