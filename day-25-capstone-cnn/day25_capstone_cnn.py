"""
Day 25 - 75 Days of Code Challenge
Deep Learning Basics - NN Experiment #10: Deep Learning Capstone
Author: Rudra (Ruddy2310)

Goal: Wrap up Phase 1's Deep Learning arc (Days 16-24: from-scratch NN,
Keras NN, CNN, regularized CNN, autoencoder, RNN/LSTM, NLP sentiment,
GAN, Siamese network) with a production-style capstone that combines:
  - Functional API (more flexible than Sequential, needed for real projects)
  - Batch Normalization (stabilizes and speeds up training)
  - Dropout (regularization, from Day 19)
  - ModelCheckpoint (saves the best model automatically during training)
  - ReduceLROnPlateau (lowers the learning rate when progress stalls,
    instead of a fixed learning rate for the whole run)
  - EarlyStopping (from Day 17 onward)

Dataset: sklearn's digits dataset (same as most of Days 16-24, so this
capstone's result is directly comparable to every prior experiment).
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
# 1. Load & preprocess data
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
# 2. Build the capstone model with the Functional API
# -----------------------------
inputs = keras.Input(shape=(8, 8, 1), name="digit_image")

x = layers.Conv2D(32, (3, 3), padding="same")(inputs)
x = layers.BatchNormalization()(x)
x = layers.Activation("relu")(x)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Dropout(0.25)(x)

x = layers.Conv2D(64, (3, 3), padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.Activation("relu")(x)
x = layers.Dropout(0.25)(x)

x = layers.Flatten()(x)
x = layers.Dense(64)(x)
x = layers.BatchNormalization()(x)
x = layers.Activation("relu")(x)
x = layers.Dropout(0.4)(x)

outputs = layers.Dense(10, activation="softmax", name="digit_class")(x)

model = keras.Model(inputs, outputs, name="capstone_cnn")
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()

# -----------------------------
# 3. Production-style callbacks
# -----------------------------
callbacks = [
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True),
    keras.callbacks.ModelCheckpoint("best_model.keras", monitor="val_accuracy", save_best_only=True, verbose=0),
    keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=8, min_lr=1e-6, verbose=1),
]

# -----------------------------
# 4. Train
# -----------------------------
history = model.fit(
    X_train, y_train,
    validation_split=0.15,
    epochs=150,
    batch_size=32,
    callbacks=callbacks,
    verbose=2,
)

actual_epochs = len(history.history["loss"])
print(f"\nTraining stopped after {actual_epochs} epochs.")

# Load the best checkpointed model (in case the run didn't end on the best epoch)
best_model = keras.models.load_model("best_model.keras")

# -----------------------------
# 5. Evaluate
# -----------------------------
test_loss, test_acc = best_model.evaluate(X_test, y_test, verbose=0)
print(f"\nFinal Test Accuracy (best checkpoint): {test_acc:.4f}")

y_pred_proba = best_model.predict(X_test, verbose=0)
y_pred = np.argmax(y_pred_proba, axis=1)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# -----------------------------
# 6. Compare against every prior Deep Learning experiment (Days 16-24)
# -----------------------------
prior_results = {
    "Day 16: From-scratch NN": 0.961,
    "Day 17: Keras NN": 0.9667,
    "Day 18: Plain CNN": 0.9778,
    "Day 19: Regularized CNN": 0.9806,
    "Day 25: Capstone CNN": test_acc,
}

# -----------------------------
# 7. Visualizations
# -----------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(history.history["loss"], label="Train Loss", color="crimson")
axes[0].plot(history.history["val_loss"], label="Validation Loss", color="orange")
axes[0].set_title("Capstone CNN - Loss over Epochs")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()

axes[1].plot(history.history["accuracy"], label="Train Accuracy", color="seagreen")
axes[1].plot(history.history["val_accuracy"], label="Validation Accuracy", color="teal")
axes[1].set_title("Capstone CNN - Accuracy over Epochs")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].set_ylim(0, 1)
axes[1].legend()

if "lr" in history.history:
    axes[2].plot(history.history["lr"], color="purple")
    axes[2].set_title("Learning Rate over Epochs (ReduceLROnPlateau)")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Learning Rate")
    axes[2].set_yscale("log")

plt.tight_layout()
plt.savefig("training_curves.png", dpi=150)
plt.close()

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title(f"Capstone CNN Confusion Matrix (Test Acc: {test_acc:.2%})")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.close()

# Phase 1 Deep Learning journey - accuracy progression across all NN experiments
plt.figure(figsize=(10, 5))
names = list(prior_results.keys())
values = list(prior_results.values())
colors = ["steelblue"] * (len(names) - 1) + ["crimson"]
bars = plt.bar(names, values, color=colors)
plt.ylabel("Test Accuracy")
plt.title("Deep Learning Basics Journey - Accuracy Across Experiments (Days 16-25)")
plt.ylim(0.9, 1.0)
plt.xticks(rotation=20, ha="right")
for bar, val in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width()/2, val + 0.002, f"{val:.2%}", ha="center", fontsize=9)
plt.tight_layout()
plt.savefig("phase1_dl_journey.png", dpi=150)
plt.close()

print("\nSaved plots: training_curves.png, confusion_matrix.png, phase1_dl_journey.png")
print(f"\nDone. Phase 1 (Deep Learning Basics) complete.")
print(f"Capstone CNN test accuracy: {test_acc:.2%}")
print("Next up: Phase 2 - Big Projects (RAG chatbot / CV project / NLP tool).")
