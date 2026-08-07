"""
Day 16 - 75 Days of Code Challenge
Deep Learning Basics - NN Experiment #1: Neural Network from Scratch
Author: Rudra (Ruddy2310)

Goal: Build a feedforward neural network completely from scratch using
only NumPy - no TensorFlow, no PyTorch - to understand exactly what's
happening inside a neural network: forward propagation, backpropagation,
and gradient descent, before using deep learning frameworks in later
experiments.

Architecture: 64 (input) -> 32 (hidden, ReLU) -> 10 (output, softmax)
Dataset: sklearn's built-in digits dataset (8x8 handwritten digit images).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# -----------------------------
# 1. Load & preprocess data
# -----------------------------
data = load_digits()
X = data.data   # (1797, 64)
y = data.target  # (1797,) - digit labels 0-9

print("Dataset shape:", X.shape)
print("Classes:", np.unique(y))

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# One-hot encode labels for softmax + cross-entropy
n_classes = 10
y_onehot = np.eye(n_classes)[y]

X_train, X_test, y_train, y_test, y_train_labels, y_test_labels = train_test_split(
    X_scaled, y_onehot, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# -----------------------------
# 2. Neural network building blocks
# -----------------------------
def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

def softmax(z):
    # subtract max per row for numerical stability
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def cross_entropy_loss(y_pred, y_true):
    n = y_true.shape[0]
    eps = 1e-9  # avoid log(0)
    return -np.sum(y_true * np.log(y_pred + eps)) / n


class NeuralNetworkFromScratch:
    """A minimal 2-layer (1 hidden layer) feedforward neural network."""

    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        # He-initialization (good default for ReLU layers)
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((1, output_size))
        self.lr = learning_rate

    def forward(self, X):
        self.Z1 = X @ self.W1 + self.b1
        self.A1 = relu(self.Z1)
        self.Z2 = self.A1 @ self.W2 + self.b2
        self.A2 = softmax(self.Z2)
        return self.A2

    def backward(self, X, y_true):
        n = X.shape[0]

        # Output layer gradient (softmax + cross-entropy combined derivative)
        dZ2 = self.A2 - y_true                  # (n, output_size)
        dW2 = self.A1.T @ dZ2 / n
        db2 = np.sum(dZ2, axis=0, keepdims=True) / n

        # Hidden layer gradient
        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * relu_derivative(self.Z1)
        dW1 = X.T @ dZ1 / n
        db1 = np.sum(dZ1, axis=0, keepdims=True) / n

        # Gradient descent update
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def train(self, X, y, epochs=300, verbose_every=30):
        history = {"loss": [], "accuracy": []}
        for epoch in range(epochs):
            y_pred = self.forward(X)
            loss = cross_entropy_loss(y_pred, y)
            acc = accuracy_score(np.argmax(y, axis=1), np.argmax(y_pred, axis=1))
            history["loss"].append(loss)
            history["accuracy"].append(acc)

            self.backward(X, y)

            if (epoch + 1) % verbose_every == 0 or epoch == 0:
                print(f"Epoch {epoch+1:4d}/{epochs} | Loss: {loss:.4f} | Train Accuracy: {acc:.4f}")

        return history

    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)


# -----------------------------
# 3. Train the network
# -----------------------------
print("\nTraining neural network from scratch...\n")
nn = NeuralNetworkFromScratch(input_size=64, hidden_size=32, output_size=10, learning_rate=0.5)
history = nn.train(X_train, y_train, epochs=300, verbose_every=30)

# -----------------------------
# 4. Evaluate on test set
# -----------------------------
test_pred = nn.predict(X_test)
test_acc = accuracy_score(y_test_labels, test_pred)
print(f"\nFinal Test Accuracy: {test_acc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test_labels, test_pred))

# -----------------------------
# 5. Visualizations
# -----------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].plot(history["loss"], color="crimson")
axes[0].set_title("Training Loss over Epochs")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Cross-Entropy Loss")

axes[1].plot(history["accuracy"], color="seagreen")
axes[1].set_title("Training Accuracy over Epochs")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].set_ylim(0, 1)

plt.tight_layout()
plt.savefig("training_curves.png", dpi=150)
plt.close()

cm = confusion_matrix(y_test_labels, test_pred)
plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title(f"Confusion Matrix - From-Scratch NN (Test Acc: {test_acc:.2%})")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.close()

# Visualize a few test predictions
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
sample_idx = np.random.choice(len(X_test), 10, replace=False)
for i, ax in enumerate(axes.flat):
    idx = sample_idx[i]
    img = X_test[idx].reshape(8, 8)
    pred = test_pred[idx]
    actual = y_test_labels[idx]
    color = "green" if pred == actual else "red"
    ax.imshow(img, cmap="gray")
    ax.set_title(f"Pred: {pred}, True: {actual}", color=color, fontsize=10)
    ax.axis("off")
plt.suptitle("Sample Test Predictions (green=correct, red=wrong)")
plt.tight_layout()
plt.savefig("sample_predictions.png", dpi=150)
plt.close()

print("\nSaved plots: training_curves.png, confusion_matrix.png, sample_predictions.png")
print("\nDone. Built and trained a neural network entirely from scratch with NumPy.")
