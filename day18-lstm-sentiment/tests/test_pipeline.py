"""
test_pipeline.py
-----------------
Lightweight sanity tests — no test framework required.
Run with: python tests/test_pipeline.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

from dataset import (
    SentimentDataset,
    Vocab,
    collate_batch,
    generate_synthetic_dataset,
    tokenize,
    train_val_split,
)
from model import LSTMSentimentClassifier


def test_tokenize():
    assert tokenize("Great film! Loved it.") == ["great", "film", "loved", "it"]
    print("test_tokenize: OK")


def test_synthetic_dataset():
    data = generate_synthetic_dataset(n_samples=50, seed=1)
    assert len(data) == 50
    assert all(label in (0, 1) for _, label in data)
    print("test_synthetic_dataset: OK")


def test_vocab_and_dataset():
    data = generate_synthetic_dataset(n_samples=50, seed=1)
    train, val = train_val_split(data, val_frac=0.2, seed=1)
    assert len(train) + len(val) == 50

    vocab = Vocab([t for t, _ in train])
    assert len(vocab) > 2  # pad + unk + real words

    ds = SentimentDataset(train, vocab, max_len=20)
    ids, label = ds[0]
    assert ids.dtype == torch.long
    assert label.dtype == torch.float32
    print("test_vocab_and_dataset: OK")


def test_model_forward_pass():
    data = generate_synthetic_dataset(n_samples=16, seed=1)
    vocab = Vocab([t for t, _ in data])
    ds = SentimentDataset(data, vocab, max_len=20)
    batch = [ds[i] for i in range(8)]
    input_ids, lengths, labels = collate_batch(batch)

    model = LSTMSentimentClassifier(vocab_size=len(vocab), embed_dim=16, hidden_dim=16, num_layers=1)
    logits = model(input_ids, lengths)

    assert logits.shape == (8,)
    assert torch.isfinite(logits).all()
    print("test_model_forward_pass: OK")


if __name__ == "__main__":
    test_tokenize()
    test_synthetic_dataset()
    test_vocab_and_dataset()
    test_model_forward_pass()
    print("\nAll tests passed.")
