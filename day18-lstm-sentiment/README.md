# Day 18 — LSTM Sentiment Classifier

Part of [#75DaysOfCode](https://github.com/Ruddy2310) — daily ML / deep learning exercises.

A bidirectional LSTM text classifier built in PyTorch for binary sentiment
classification (positive / negative), with a packed-sequence input pipeline,
mean+max pooling over the LSTM outputs, and a small classifier head.

## Project structure

```
day18-lstm-sentiment/
├── src/
│   ├── dataset.py     # tokenizer, vocab, Dataset, collate_fn, data sources
│   ├── model.py        # LSTMSentimentClassifier
│   ├── train.py         # training loop + checkpointing (CLI)
│   └── predict.py       # load a checkpoint and run inference (CLI)
├── notebooks/
│   └── demo.ipynb        # walkthrough: data -> vocab -> model -> train -> predict
├── tests/
│   └── test_pipeline.py  # lightweight sanity tests, no framework needed
├── saved_models/         # trained checkpoints land here
├── data/                 # placeholder for cached/downloaded datasets
├── requirements.txt
└── README.md
```

## Why it runs offline

Training data for this repo comes from a synthetic-but-realistic review
generator in `src/dataset.py`, so the whole pipeline (data → train → predict)
runs end-to-end with **zero external downloads**. This makes it easy to clone
and run immediately, including in sandboxed/CI environments without network
access to dataset mirrors.

To use the real IMDB dataset instead, it's a one-line change:

```python
# src/dataset.py
USE_REAL_IMDB = True
```

(requires `pip install datasets` and network access to HuggingFace Hub).

## Model

```
Embedding -> 2-layer bidirectional LSTM -> mean+max pool over time -> MLP head -> 1 logit
```

- Padding is masked out of both the LSTM (via `pack_padded_sequence`) and the
  pooling step, so it never influences predictions.
- Trained with `BCEWithLogitsLoss` + gradient clipping.

## Quickstart

```bash
pip install -r requirements.txt

# train (saves best checkpoint by val accuracy to saved_models/best_model.pt)
python src/train.py --epochs 8 --batch-size 64

# predict on new text
python src/predict.py --checkpoint saved_models/best_model.pt \
    --text "what a fantastic and touching film"

# sanity tests
python tests/test_pipeline.py
```

Or open `notebooks/demo.ipynb` for an annotated walkthrough.

## Notes

- The synthetic dataset exists to make the repo self-contained and reviewable
  end-to-end; it's intentionally simple (template + word-bank based) so
  training converges fast. For real benchmark numbers, switch to real IMDB.
- `src/train.py` and `src/predict.py` are CLI scripts with `argparse`, so all
  hyperparameters are configurable without touching the code.
