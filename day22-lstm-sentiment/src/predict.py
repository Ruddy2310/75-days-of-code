import argparse
import re
from pathlib import Path

import torch

from .dataset import tokenize
from .model import LSTMSentimentClassifier


def predict(text, checkpoint_path="outputs/lstm_sentiment.pt"):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    vocab = checkpoint["vocab"]
    max_len = checkpoint["max_len"]

    model = LSTMSentimentClassifier(len(vocab))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    tokens = tokenize(text)[:max_len]
    ids = [vocab.get(token, vocab["<UNK>"]) for token in tokens] or [vocab["<UNK>"]]

    x = torch.tensor([ids], dtype=torch.long)
    lengths = torch.tensor([len(ids)], dtype=torch.long)

    with torch.no_grad():
        probability = torch.sigmoid(model(x, lengths)).item()

    sentiment = "positive" if probability >= 0.5 else "negative"
    confidence = probability if sentiment == "positive" else 1 - probability
    return sentiment, confidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("text", nargs="+")
    args = parser.parse_args()

    text = " ".join(args.text)
    sentiment, confidence = predict(text)
    print(f"Sentiment: {sentiment}")
    print(f"Confidence: {confidence:.2%}")
