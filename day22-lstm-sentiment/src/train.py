import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

from .dataset import (
    TextDataset,
    build_vocab,
    collate_batch,
    load_csv,
    make_fallback_dataset,
)
from .model import LSTMSentimentClassifier


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for x, lengths, y in loader:
            x, lengths, y = x.to(device), lengths.to(device), y.to(device)
            logits = model(x, lengths)
            loss = criterion(logits, y)

            total_loss += loss.item() * len(y)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            correct += (preds == y).sum().item()
            total += len(y)

    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-len", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    if args.csv:
        rows = load_csv(args.csv)
    else:
        rows = make_fallback_dataset(n=1000, seed=args.seed)

    random.shuffle(rows)
    split = int(0.8 * len(rows))
    train_rows, val_rows = rows[:split], rows[split:]

    vocab = build_vocab(train_rows)
    train_ds = TextDataset(train_rows, vocab, args.max_len)
    val_ds = TextDataset(val_rows, vocab, args.max_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMSentimentClassifier(len(vocab)).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    history = []

    print(f"Device: {device}")
    print(f"Training samples: {len(train_ds)}")
    print(f"Validation samples: {len(val_ds)}")
    print(f"Vocabulary size: {len(vocab)}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for x, lengths, y in train_loader:
            x, lengths, y = x.to(device), lengths.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x, lengths)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(y)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            correct += (preds == y).sum().item()
            total += len(y)

        train_loss = running_loss / total
        train_acc = correct / total
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
        })

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train loss {train_loss:.4f} | train acc {train_acc:.3f} | "
            f"val loss {val_loss:.4f} | val acc {val_acc:.3f}"
        )

    torch.save(
        {
            "model_state": model.state_dict(),
            "vocab": vocab,
            "max_len": args.max_len,
        },
        output_dir / "lstm_sentiment.pt",
    )

    (output_dir / "history.json").write_text(json.dumps(history, indent=2))
    print(f"Saved model to {output_dir / 'lstm_sentiment.pt'}")


if __name__ == "__main__":
    main()
