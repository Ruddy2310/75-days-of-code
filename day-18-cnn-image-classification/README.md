# Day 18 — Deep Learning Basics: NN Experiment #3 (CNN)

Part of my #75DaysOfCode challenge.

## What it does
Moves from plain dense networks (Day 16-17) to a Convolutional Neural
Network, letting the model learn spatial patterns instead of treating
pixels as independent features.

## Architecture
Conv2D(16) -> MaxPool -> Conv2D(32) -> Flatten -> Dense(32) -> Dense(10, softmax)

## Results
- Test accuracy: 97.78% (up from Day 17's dense NN: 96.67%)
- Same digits dataset as Day 16/17, reshaped to 8x8x1 images

## Files
- day18_cnn_image_classification.py — main script
- day18_training_curves.png — loss/accuracy over epochs
- day18_confusion_matrix.png — test set confusion matrix
- day18_learned_filters.png — visualized first-layer conv filters
