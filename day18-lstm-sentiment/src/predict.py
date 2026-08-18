"""
predict.py
----------
Loads a trained checkpoint and runs sentiment prediction on arbitrary text.

Usage:
    python src/predict.py --checkpoint saved_models/best_model.pt \
        --text "what a fantastic and touching film"
"""

import argparse

import torch

from dataset import PAD_TOKEN, UNK_TOKEN, Vocab, tokenize
from model import LSTMSentimentClassifier


def load_checkpoint(path: str, device: torch.device):
    ckpt = torch.load(path, map_location=device, weights_only=False)

    vocab = Vocab.__new__(Vocab)
    vocab.itos = ckpt["vocab_itos"]
    vocab.stoi = {w: i for i, w in enumerate(vocab.itos)}

    cfg = ckpt["config"]
    model = LSTMSentimentClassifier(
        vocab_size=len(vocab),
        embed_dim=cfg["embed_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, vocab, cfg["max_len"]


def predict(text: str, model, vocab: Vocab, max_len: int, device: torch.device):
    ids = vocab.encode(text)[:max_len]
    if not ids:
        ids = [vocab.stoi[UNK_TOKEN]]
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    lengths = torch.tensor([len(ids)], dtype=torch.long)

    with torch.no_grad():
        logit = model(input_ids, lengths)
        prob = torch.sigmoid(logit).item()

    label = "positive" if prob >= 0.5 else "negative"
    confidence = prob if prob >= 0.5 else 1 - prob
    return label, confidence


def main():
    parser = argparse.ArgumentParser(description="Run sentiment prediction")
    parser.add_argument("--checkpoint", type=str, default="saved_models/best_model.pt")
    parser.add_argument("--text", type=str, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, vocab, max_len = load_checkpoint(args.checkpoint, device)
    label, confidence = predict(args.text, model, vocab, max_len, device)

    print(f'Text:       "{args.text}"')
    print(f"Prediction: {label} ({confidence:.1%} confidence)")


if __name__ == "__main__":
    main()
