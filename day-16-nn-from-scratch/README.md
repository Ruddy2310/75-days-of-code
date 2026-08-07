# Day 16 — Deep Learning Basics: NN Experiment #1 (Neural Network from Scratch)

Part of my #75DaysOfCode challenge. First project of Phase 2: Deep Learning.

## What it does
Builds a feedforward neural network entirely from scratch using only
NumPy - no TensorFlow, no PyTorch. Implements forward propagation,
backpropagation, and gradient descent by hand to understand exactly
what's happening inside a neural network before using frameworks.

## Architecture
64 (input) -> 32 (hidden, ReLU) -> 10 (output, softmax)

## Results
- Training accuracy: 100%
- Test accuracy: 96.1%
- Uses sklearn's built-in digits dataset, no download needed

## Files
- day16_neural_network_from_scratch.py — main script (full NN implementation)
- training_curves.png — loss and accuracy over 300 epochs
- confusion_matrix.png — test set confusion matrix
- sample_predictions.png — example predictions (green=correct, red=wrong)
