"""
Day 18 - 75 Days of Code Challenge
Deep Learning Basics - NN Experiment #3: Convolutional Neural Network (CNN)
Author: Rudra (Ruddy2310)

Goal: Move from plain dense (fully-connected) networks (Day 16-17) to a
Convolutional Neural Network - the standard architecture for image data.
Convolutions let the network learn spatial patterns (edges, curves,
strokes) instead of treating each pixel as an independent feature like
the flattened dense networks did.

Dataset: sklearn's digits dataset reshaped to 8x8x1 images (same data as
Day 16/17, for a direct architecture comparison).
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

# -----------------------------
# 1. Load & preprocess data - reshape to images for CNN input
# -----------------------------
data = load_digits()
X = data.images  # (1797, 8, 8) - already 2D images, unlike the flattened version used before
y = data.target

# Normalize pixel values to 0-1, add channel dimension -> (n, 8, 8, 1)
X = X / 16.0  # digits dataset pixel values range 0-16
X = X[..., np.newaxis]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

print("Dataset shape:", X.shape, "(images, height, width, channels)")
print("Train/Test split:", X_train.shape[0], "/", X_test.shape[0])

# -----------------------------
# 2. Build the CNN
# -----------------------------
model = keras.Sequential([
    layers.Input(shape=(8, 8, 1)),
    layers.Conv2D(16, (3, 3), activation="relu", padding="same", name="conv1"),
    layers.MaxPooling2D((2, 2), name="pool1"),
    layers.Conv2D(32, (3, 3), activation="relu", padding="same", name="conv2"),
    layers.Flatten(),
    layers.Dense(32, activation="relu", name="dense1"),
    layers.Dense(10, activation="softmax", name="output"),
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# -----------------------------
# 3. Train
# -----------------------------
early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=15, restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    validation_split=0.15,
    epochs=100,
    batch_size=32,
    callbacks=[early_stop],
    verbose=2,
)

actual_epochs = len(history.history["loss"])
print(f"\nTraining stopped after {actual_epochs} epochs.")

# -----------------------------
# 4. Evaluate
# -----------------------------
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nFinal Test Accuracy: {test_acc:.4f}")

y_pred_proba = model.predict(X_test, verbose=0)
y_pred = np.argmax(y_pred_proba, axis=1)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# -----------------------------
# 5. Visualizations
# -----------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].plot(history.history["loss"], label="Train Loss", color="crimson")
axes[0].plot(history.history["val_loss"], label="Validation Loss", color="orange")
axes[0].set_title("CNN Loss over Epochs")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()

axes[1].plot(history.history["accuracy"], label="Train Accuracy", color="seagreen")
axes[1].plot(history.history["val_accuracy"], label="Validation Accuracy", color="teal")
axes[1].set_title("CNN Accuracy over Epochs")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].set_ylim(0, 1)
axes[1].legend()

plt.tight_layout()
plt.savefig("day18_training_curves.png", dpi=150)
plt.close()

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title(f"CNN Confusion Matrix (Test Acc: {test_acc:.2%})")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("day18_confusion_matrix.png", dpi=150)
plt.close()

# Visualize learned convolution filters from the first conv layer
first_conv_weights = model.get_layer("conv1").get_weights()[0]  # shape (3,3,1,16)
fig, axes = plt.subplots(2, 8, figsize=(14, 4))
for i, ax in enumerate(axes.flat):
    filt = first_conv_weights[:, :, 0, i]
    ax.imshow(filt, cmap="viridis")
    ax.set_title(f"Filter {i+1}", fontsize=8)
    ax.axis("off")
plt.suptitle("Learned Conv Layer 1 Filters (3x3, 16 filters)")
plt.tight_layout()
plt.savefig("day18_learned_filters.png", dpi=150)
plt.close()

print("\nSaved plots: day18_training_curves.png, day18_confusion_matrix.png, day18_learned_filters.png")
print(f"\nDone. CNN test accuracy: {test_acc:.2%} vs Day 17's dense NN (96.67%) on the same digits.")
