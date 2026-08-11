"""
Day 20 - 75 Days of Code Challenge
Deep Learning Basics - NN Experiment #5: Autoencoder (Compression & Denoising)
Author: Rudra (Ruddy2310)

Goal: Build an Autoencoder - an unsupervised architecture that learns to
compress data into a small latent representation and then reconstruct
it. Unlike every previous experiment (16-19, all supervised classifiers),
this network's target output is the input itself, forcing it to learn
which features actually matter.

Two experiments in one script:
  1. Plain autoencoder - compress 64 pixels down to a small latent
     space, then reconstruct.
  2. Denoising autoencoder - same architecture, but trained to map
     NOISY images back to their CLEAN originals, a classic trick for
     teaching a network what's signal vs what's noise.

Dataset: sklearn's digits dataset (same as Day 16-19, for comparison).
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

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
X = data.data / 16.0  # normalize to 0-1
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

print("Dataset shape:", X.shape)
print("Train/Test split:", X_train.shape[0], "/", X_test.shape[0])

LATENT_DIM = 8  # compress 64 pixels down to just 8 numbers

# -----------------------------
# 2. Build the plain autoencoder
# -----------------------------
def build_autoencoder(latent_dim):
    inputs = keras.Input(shape=(64,))
    x = layers.Dense(32, activation="relu")(inputs)
    latent = layers.Dense(latent_dim, activation="relu", name="latent_space")(x)
    x = layers.Dense(32, activation="relu")(latent)
    outputs = layers.Dense(64, activation="sigmoid")(x)  # sigmoid since pixels are 0-1

    autoencoder = keras.Model(inputs, outputs, name="autoencoder")
    encoder = keras.Model(inputs, latent, name="encoder")
    return autoencoder, encoder

autoencoder, encoder = build_autoencoder(LATENT_DIM)
autoencoder.compile(optimizer="adam", loss="mse")
autoencoder.summary()

print("\nTraining plain autoencoder...")
history_ae = autoencoder.fit(
    X_train, X_train,  # input = target, this is what makes it unsupervised
    validation_split=0.15,
    epochs=100,
    batch_size=32,
    verbose=0,
)
print(f"Final reconstruction loss (MSE): {history_ae.history['loss'][-1]:.5f}")

# -----------------------------
# 3. Build the denoising autoencoder (same architecture, noisy input)
# -----------------------------
noise_factor = 0.4
X_train_noisy = np.clip(X_train + noise_factor * np.random.randn(*X_train.shape), 0, 1)
X_test_noisy = np.clip(X_test + noise_factor * np.random.randn(*X_test.shape), 0, 1)

denoiser, _ = build_autoencoder(LATENT_DIM)
denoiser.compile(optimizer="adam", loss="mse")

print("\nTraining denoising autoencoder...")
history_denoise = denoiser.fit(
    X_train_noisy, X_train,  # noisy input -> clean target
    validation_split=0.15,
    epochs=100,
    batch_size=32,
    verbose=0,
)
print(f"Final denoising loss (MSE): {history_denoise.history['loss'][-1]:.5f}")

# -----------------------------
# 4. Evaluate reconstructions
# -----------------------------
reconstructed = autoencoder.predict(X_test, verbose=0)
denoised = denoiser.predict(X_test_noisy, verbose=0)

test_mse_plain = np.mean((X_test - reconstructed) ** 2)
test_mse_denoise = np.mean((X_test - denoised) ** 2)
print(f"\nTest reconstruction MSE (plain AE, clean input): {test_mse_plain:.5f}")
print(f"Test reconstruction MSE (denoising AE, noisy input): {test_mse_denoise:.5f}")

# -----------------------------
# 5. Visualizations
# -----------------------------
# Loss curves for both autoencoders
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].plot(history_ae.history["loss"], label="Train Loss")
axes[0].plot(history_ae.history["val_loss"], label="Val Loss")
axes[0].set_title("Plain Autoencoder - Reconstruction Loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("MSE")
axes[0].legend()

axes[1].plot(history_denoise.history["loss"], label="Train Loss", color="darkorange")
axes[1].plot(history_denoise.history["val_loss"], label="Val Loss", color="brown")
axes[1].set_title("Denoising Autoencoder - Reconstruction Loss")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("MSE")
axes[1].legend()

plt.tight_layout()
plt.savefig("training_curves.png", dpi=150)
plt.close()

# Original vs Reconstructed (plain autoencoder)
n_examples = 8
fig, axes = plt.subplots(2, n_examples, figsize=(14, 4))
for i in range(n_examples):
    axes[0, i].imshow(X_test[i].reshape(8, 8), cmap="gray")
    axes[0, i].axis("off")
    axes[1, i].imshow(reconstructed[i].reshape(8, 8), cmap="gray")
    axes[1, i].axis("off")
axes[0, 0].set_ylabel("Original", fontsize=10)
axes[1, 0].set_ylabel("Reconstructed", fontsize=10)
plt.suptitle(f"Plain Autoencoder: Original (top) vs Reconstructed (bottom) - Latent dim = {LATENT_DIM}")
plt.tight_layout()
plt.savefig("reconstruction_comparison.png", dpi=150)
plt.close()

# Noisy -> Denoised -> Clean comparison
fig, axes = plt.subplots(3, n_examples, figsize=(14, 6))
for i in range(n_examples):
    axes[0, i].imshow(X_test[i].reshape(8, 8), cmap="gray")
    axes[0, i].axis("off")
    axes[1, i].imshow(X_test_noisy[i].reshape(8, 8), cmap="gray")
    axes[1, i].axis("off")
    axes[2, i].imshow(denoised[i].reshape(8, 8), cmap="gray")
    axes[2, i].axis("off")
fig.text(0.08, 0.78, "Clean", fontsize=10, rotation=90, va="center")
fig.text(0.08, 0.5, "Noisy Input", fontsize=10, rotation=90, va="center")
fig.text(0.08, 0.22, "Denoised", fontsize=10, rotation=90, va="center")
plt.suptitle("Denoising Autoencoder: Clean -> Noisy Input -> Denoised Output")
plt.tight_layout()
plt.savefig("denoising_comparison.png", dpi=150)
plt.close()

# Visualize the latent space (compressed to 2D with PCA for plotting)
latent_vectors = encoder.predict(X_test, verbose=0)
latent_2d = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(latent_vectors)

plt.figure(figsize=(8, 7))
scatter = plt.scatter(latent_2d[:, 0], latent_2d[:, 1], c=y_test, cmap="tab10", s=40, edgecolor="k", alpha=0.8)
plt.colorbar(scatter, label="Digit label")
plt.title(f"Learned Latent Space (dim={LATENT_DIM}, PCA-projected to 2D)\nColored by true digit label")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.tight_layout()
plt.savefig("latent_space_visualization.png", dpi=150)
plt.close()

print("\nSaved plots: training_curves.png, reconstruction_comparison.png,")
print("denoising_comparison.png, latent_space_visualization.png")
print("\nDone. Autoencoder compresses 64 pixels down to just 8 numbers and reconstructs them,")
print("and even though it never saw digit labels, the latent space still clusters by digit.")
