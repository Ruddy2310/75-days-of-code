# Day 21 — Deep Learning Basics: NN Experiment #6 (RNN vs LSTM)

Part of my #75DaysOfCode challenge. First recurrent architecture in the challenge.

## What it does
Predicts the next value in a noisy sine wave given the previous 30 steps,
comparing SimpleRNN against LSTM - the architecture LSTM was designed to
improve on for longer-range dependencies.

## Results
- SimpleRNN: RMSE 0.158, R2 0.951
- LSTM: RMSE 0.156, R2 0.952 (slight edge)
- Small gap is expected: this task only needs short-range memory, where
  LSTM's main advantage (handling long-range dependencies) doesn't show
  up as strongly.

## Files
- day21_rnn_lstm_forecasting.py — main script (synthetic sine wave data)
- training_and_performance.png — loss curves + R2 comparison
- predicted_vs_actual_sequence.png — predicted vs actual for both models
