# Day 22 — Deep Learning Basics: NN Experiment #7 (Text Sentiment - Embedding + LSTM)

Part of my #75DaysOfCode challenge. First NLP project in the challenge.

## What it does
Classifies sentence sentiment (positive/negative) using a Word Embedding
layer + LSTM. Trained on a templated synthetic dataset generated
in-script (word banks x sentence templates x subjects), then tested on
brand-new hand-written sentences it never saw during training.

## Architecture
Embedding(16-dim) -> LSTM(24) -> Dense(16, ReLU) -> Dropout -> Dense(1, sigmoid)

## Results
- Test accuracy: 100% on held-out generated sentences
- Correctly classified all 6 hand-written novel sentences (true generalization test)
- Learned word embeddings naturally cluster positive vs negative words

## Files
- day22_sentiment_lstm.py — main script (self-contained dataset generation + model)
- training_curves.png — loss/accuracy over epochs
- confusion_matrix.png — test set confusion matrix
- word_embeddings_visualization.png — PCA-projected word embeddings, colored by sentiment
