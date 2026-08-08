"""
Day 19 - 75 Days of Code Challenge
Deep Learning Basics - NN Experiment #4: CNN with Regularization
Author: Rudra (Ruddy2310)

Goal: Build directly on Day 18's CNN by adding two standard
regularization techniques that fight overfitting:
  1. Dropout - randomly disables neurons during training, forcing the
     network to not rely too heavily on any single one.
  2. Data augmentation - randomly rotates/shifts/zooms training images,
     so the network sees more variation and generalizes better instead
     of memorizing exact pixel positions.

We compare against Day 18's plain CNN to see whether these techniques
close the gap between training and validation accuracy (i.e. reduce
overfitting), even if raw test accuracy doesn't change dramatically on
this small, relatively easy dataset.

Dataset: sklearn's digits dataset (same as Day 16-18, for comparison).
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
# 1. Load & preprocess data (identical to Day 18)
# -----------------------------
data = load_digits()
X = data.images / 16.0
X = X[..., np.newaxis]
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

print("Dataset shape:", X.shape)
print("Train/Test split:", X_train.shape[0], "/", X_test.shape[0])

# -----------------------------
# 2. Data augmentation layer - only active during training
# -----------------------------
data_augmentation = keras.Sequential([
    layers.RandomRotation(0.08),       # up to ~30 degrees
    layers.RandomTranslation(0.08, 0.08),
    layers.RandomZoom(0.08),
], name="augmentation")

# -----------------------------
# 3. Build CNN with dropout + augmentation
# -----------------------------
inputs = keras.Input(shape=(8, 8, 1))
x = data_augmentation(inputs)
x = layers.Conv2D(16, (3, 3), activation="relu", padding="same")(x)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Dropout(0.25)(x)
x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
x = layers.Dropout(0.25)(x)
x = layers.Flatten()(x)
x = layers.Dense(32, activation="relu")(x)
x = layers.Dropout(0.4)(x)
outputs = layers.Dense(10, activation="softmax")(x)

model = keras.Model(inputs, outputs, name="regularized_cnn")

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# -----------------------------
# 4. Train
# -----------------------------
early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=20, restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    validation_split=0.15,
    epochs=150,
    batch_size=32,
    callbacks=[early_stop],
    verbose=2,
)

actual_epochs = len(history.history["loss"])
print(f"\nTraining stopped after {actual_epochs} epochs.")

# -----------------------------
# 5. Evaluate
# -----------------------------
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nFinal Test Accuracy: {test_acc:.4f}")

y_pred_proba = model.predict(X_test, verbose=0)
y_pred = np.argmax(y_pred_proba, axis=1)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Overfitting gap: train accuracy - validation accuracy (smaller = better generalization)
final_train_acc = history.history["accuracy"][-1]
final_val_acc = history.history["val_accuracy"][-1]
gap = final_train_acc - final_val_acc
print(f"\nFinal train accuracy: {final_train_acc:.4f}")
print(f"Final val accuracy:   {final_val_acc:.4f}")
print(f"Train-Val gap (overfitting indicator): {gap:.4f}")

# -----------------------------
# 6. Visualizations
# -----------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].plot(history.history["loss"], label="Train Loss", color="crimson")
axes[0].plot(history.history["val_loss"], label="Validation Loss", color="orange")
axes[0].set_title("Regularized CNN - Loss over Epochs")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()

axes[1].plot(history.history["accuracy"], label="Train Accuracy", color="seagreen")
axes[1].plot(history.history["val_accuracy"], label="Validation Accuracy", color="teal")
axes[1].set_title("Regularized CNN - Accuracy over Epochs")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].set_ylim(0, 1)
axes[1].legend()

plt.tight_layout()
plt.savefig("day19_training_curves.png", dpi=150)
plt.close()

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title(f"Regularized CNN Confusion Matrix (Test Acc: {test_acc:.2%})")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("day19_confusion_matrix.png", dpi=150)
plt.close()

# Show a few augmented versions of the same digit, to visualize what the model sees
sample_digit = X_train[0:1]
fig, axes = plt.subplots(1, 6, figsize=(12, 2.5))
axes[0].imshow(sample_digit[0, :, :, 0], cmap="gray")
axes[0].set_title("Original")
axes[0].axis("off")
for i in range(1, 6):
    augmented = data_augmentation(sample_digit, training=True)
    axes[i].imshow(augmented[0, :, :, 0], cmap="gray")
    axes[i].set_title(f"Augmented {i}")
    axes[i].axis("off")
plt.suptitle("Data Augmentation Examples (same digit, randomly transformed)")
plt.tight_layout()
plt.savefig("day19_augmentation_examples.png", dpi=150)
plt.close()

print("\nSaved plots: day19_training_curves.png, day19_confusion_matrix.png, day19_augmentation_examples.png")
print(f"\nDone. Day 18 (plain CNN) vs Day 19 (regularized CNN) - compare train-val gaps to see")
print(f"whether dropout + augmentation reduced overfitting.")
