# Day 22 — LSTM Sentiment Classifier 🧠

A complete PyTorch NLP project that trains an LSTM-based sentiment classifier on IMDB-style movie reviews.

## Features
- PyTorch LSTM sentiment classifier
- Offline-friendly synthetic IMDB-style fallback dataset
- Real IMDB CSV support
- Vocabulary building with `<PAD>` and `<UNK>`
- Packed sequences for efficient LSTM processing
- Training / validation loop
- Accuracy and loss tracking
- Single-review inference
- Jupyter notebook walkthrough
- Clean GitHub-ready project structure

## Project Structure

```text
day22-lstm-sentiment/
├── data/
│   └── README.md
├── notebooks/
│   └── day22_lstm_sentiment.ipynb
├── src/
│   ├── __init__.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   └── predict.py
├── tests/
│   └── smoke_test.py
├── .gitignore
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Train

From the project root:

```bash
python -m src.train
```

The default configuration uses a small offline synthetic dataset so the project can run without downloading external data.

For a larger experiment, place an IMDB CSV at:

```text
data/IMDB Dataset.csv
```

with columns:

```text
review,sentiment
```

Then run:

```bash
python -m src.train --csv data/IMDB Dataset.csv --epochs 5
```

## Predict

```bash
python -m src.predict "This movie was fantastic and I really enjoyed it."
```

Example:

```text
Sentiment: positive
Confidence: 0.XX
```

## What I Learned

- Text must be converted into integer token IDs before entering an embedding layer.
- LSTMs can model sequential dependencies in language.
- Padding lets reviews with different lengths share a batch.
- Packed sequences prevent the LSTM from wasting computation on padding.
- Validation accuracy is useful for checking whether the model generalizes.

## Day 22 Goal

Build a practical NLP model from preprocessing to inference using an LSTM in PyTorch.
