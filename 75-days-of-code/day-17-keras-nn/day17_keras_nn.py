"""
Day 17 - 75 Days of Code Challenge
Deep Learning Basics - NN Experiment #2: Neural Network with TensorFlow/Keras
Author: Rudra (Ruddy2310)

Goal: Rebuild the same digit classifier from Day 16 (from-scratch NumPy NN)
using TensorFlow/Keras, to directly compare how much boilerplate a deep
learning framework saves - automatic differentiation, built-in optimizers,
and training loops - versus writing forward/backward passes by hand.

Architecture: 64 (input) -> 32 (hidden, ReLU) -> 10 (output, softmax)
Same as Day 16, so results are directly comparable.
Dataset: sklearn's built-in digits dataset (8x8 handwritten digit images).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

# -----------------------------
# 1. Load & preprocess data (identical to Day 16 for a fair comparison)
# -----------------------------
data = load_digits()
X = data.data
y = data.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

print("Dataset shape:", X.shape)
print("Train/Test split:", X_train.shape, X_test.shape)

# -----------------------------
# 2. Build the model - same architecture as Day 16's from-scratch NN
# -----------------------------
model = keras.Sequential([
    layers.Input(shape=(64,)),
    layers.Dense(32, activation="relu"),
    layers.Dense(10, activation="softmax"),
])

model.compile(
    optimizer=keras.optimizers.SGD(learning_rate=0.5),  # same optimizer/lr as Day 16
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# -----------------------------
# 3. Train
# -----------------------------
print("\nTraining with TensorFlow/Keras...\n")
history = model.fit(
    X_train, y_train,
    epochs=300,
    batch_size=len(X_train),  # full-batch gradient descent, matching Day 16
    verbose=0,
    validation_data=(X_test, y_test),
)

# Print progress at the same intervals as Day 16 for comparison
for epoch in [0, 29, 59, 89, 119, 149, 179, 209, 239, 269, 299]:
    if epoch < len(history.history["loss"]):
        print(f"Epoch {epoch+1:4d}/300 | Loss: {history.history['loss'][epoch]:.4f} "
              f"| Train Accuracy: {history.history['accuracy'][epoch]:.4f}")

# -----------------------------
# 4. Evaluate
# -----------------------------
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nFinal Test Accuracy: {test_acc:.4f}")

y_pred_probs = model.predict(X_test, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# -----------------------------
# 5. Visualizations
# -----------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].plot(history.history["loss"], label="Train Loss", color="crimson")
axes[0].plot(history.history["val_loss"], label="Val Loss", color="orange")
axes[0].set_title("Loss over Epochs (Keras)")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Cross-Entropy Loss")
axes[0].legend()

axes[1].plot(history.history["accuracy"], label="Train Accuracy", color="seagreen")
axes[1].plot(history.history["val_accuracy"], label="Val Accuracy", color="teal")
axes[1].set_title("Accuracy over Epochs (Keras)")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].set_ylim(0, 1)
axes[1].legend()

plt.tight_layout()
plt.savefig("training_curves_keras.png", dpi=150)
plt.close()

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Purples")
plt.title(f"Confusion Matrix - Keras NN (Test Acc: {test_acc:.2%})")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix_keras.png", dpi=150)
plt.close()

# Sample predictions
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
sample_idx = np.random.choice(len(X_test), 10, replace=False)
for i, ax in enumerate(axes.flat):
    idx = sample_idx[i]
    img = X_test[idx].reshape(8, 8)
    pred = y_pred[idx]
    actual = y_test[idx]
    color = "green" if pred == actual else "red"
    ax.imshow(img, cmap="gray")
    ax.set_title(f"Pred: {pred}, True: {actual}", color=color, fontsize=10)
    ax.axis("off")
plt.suptitle("Sample Test Predictions - Keras (green=correct, red=wrong)")
plt.tight_layout()
plt.savefig("sample_predictions_keras.png", dpi=150)
plt.close()

print("\nSaved plots: training_curves_keras.png, confusion_matrix_keras.png, sample_predictions_keras.png")
print(f"\nDone. Keras model: {model.count_params()} parameters, {test_acc:.2%} test accuracy.")
print("Compare against Day 16's from-scratch NumPy NN (same architecture, same data split).")
