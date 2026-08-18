"""
Day 21 - 75 Days of Code Challenge
Deep Learning Basics - NN Experiment #6: RNN vs LSTM for Sequence Forecasting
Author: Rudra (Ruddy2310)

Goal: Move into a new architecture family - Recurrent Neural Networks -
after Days 16-19 covered dense networks and CNNs. RNNs/LSTMs process
sequences step by step, carrying a "memory" of what came before, which
makes them suited to time series, text, and other ordered data.

Task: given a window of past values from a noisy sine wave, predict the
next value. Compares a SimpleRNN against an LSTM on the same task, since
LSTMs were specifically designed to fix SimpleRNN's struggle with
longer-range dependencies (the "vanishing gradient" problem).

Dataset: synthetically generated noisy sine wave (no download required,
fully reproducible with a fixed seed).
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

# -----------------------------
# 1. Generate a noisy sine wave and build windowed sequences
# -----------------------------
t = np.linspace(0, 100, 2000)
series = np.sin(t) + 0.15 * np.random.randn(len(t))  # sine wave + noise

WINDOW_SIZE = 30  # how many past steps the model sees to predict the next one

def make_windows(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)

X, y = make_windows(series, WINDOW_SIZE)
X = X[..., np.newaxis]  # shape: (samples, window_size, 1) - required by RNN layers

split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print("Total windows:", len(X))
print("Train/Test split:", len(X_train), "/", len(X_test))

# -----------------------------
# 2. Build both models
# -----------------------------
def build_simple_rnn():
    return keras.Sequential([
        layers.Input(shape=(WINDOW_SIZE, 1)),
        layers.SimpleRNN(32, activation="tanh"),
        layers.Dense(1),
    ], name="simple_rnn")

def build_lstm():
    return keras.Sequential([
        layers.Input(shape=(WINDOW_SIZE, 1)),
        layers.LSTM(32),
        layers.Dense(1),
    ], name="lstm")

models = {"SimpleRNN": build_simple_rnn(), "LSTM": build_lstm()}
histories = {}
predictions = {}
results = {}

early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

for name, model in models.items():
    print(f"\n{'='*50}\nTraining {name}...")
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])

    history = model.fit(
        X_train, y_train,
        validation_split=0.15,
        epochs=60,
        batch_size=32,
        callbacks=[early_stop],
        verbose=0,
    )
    histories[name] = history

    pred = model.predict(X_test, verbose=0).flatten()
    predictions[name] = pred

    rmse = np.sqrt(mean_squared_error(y_test, pred))
    mae = mean_absolute_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    results[name] = {"RMSE": rmse, "MAE": mae, "R2": r2, "Epochs trained": len(history.history["loss"])}

    print(f"{name} -> RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}, "
          f"trained {len(history.history['loss'])} epochs")

print(f"\n{'='*50}\n=== Final Comparison ===")
for name, metrics in results.items():
    print(f"{name}: {metrics}")

# -----------------------------
# 3. Visualizations
# -----------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for name, history in histories.items():
    axes[0].plot(history.history["loss"], label=f"{name} Train")
    axes[0].plot(history.history["val_loss"], label=f"{name} Val", linestyle="--")
axes[0].set_title("Training Loss (MSE) - SimpleRNN vs LSTM")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("MSE Loss")
axes[0].legend()

axes[1].bar(results.keys(), [m["R2"] for m in results.values()], color=["steelblue", "seagreen"])
axes[1].set_title("Test R2 Score Comparison")
axes[1].set_ylabel("R2 Score")
axes[1].set_ylim(0, 1)

plt.tight_layout()
plt.savefig("training_and_performance.png", dpi=150)
plt.close()

# Predicted vs actual sequence for both models
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
plot_range = 150  # first N test points, for readability

for ax, (name, pred) in zip(axes, predictions.items()):
    ax.plot(y_test[:plot_range], label="Actual", color="black", linewidth=1.5)
    ax.plot(pred[:plot_range], label="Predicted", color="crimson", linestyle="--")
    ax.set_title(f"{name} - Predicted vs Actual (R2 = {results[name]['R2']:.3f})")
    ax.set_ylabel("Value")
    ax.legend()

axes[1].set_xlabel("Time step")
plt.tight_layout()
plt.savefig("predicted_vs_actual_sequence.png", dpi=150)
plt.close()

print("\nSaved plots: training_and_performance.png, predicted_vs_actual_sequence.png")
print("\nDone. SimpleRNN vs LSTM sequence forecasting comparison complete.")
