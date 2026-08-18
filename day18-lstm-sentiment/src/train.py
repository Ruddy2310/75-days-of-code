"""
train.py
--------
Trains the LSTM sentiment classifier end-to-end and saves the best
checkpoint (by validation accuracy) to saved_models/.

Usage:
    python src/train.py --epochs 8 --batch-size 64
"""

import argparse
import json
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import (
    SentimentDataset,
    Vocab,
    collate_batch,
    load_raw_dataset,
    train_val_split,
)
from model import LSTMSentimentClassifier


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_epoch(model, loader, optimizer, criterion, device, train: bool):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for input_ids, lengths, labels in loader:
            input_ids, lengths, labels = (
                input_ids.to(device),
                lengths.to(device),
                labels.to(device),
            )
            if train:
                optimizer.zero_grad()

            logits = model(input_ids, lengths)
            loss = criterion(logits, labels)

            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            total_loss += loss.item() * labels.size(0)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser(description="Train the LSTM sentiment classifier")
    parser.add_argument("--n-samples", type=int, default=6000)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-len", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=str, default="saved_models")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = get_device()
    print(f"Using device: {device}")

    raw = load_raw_dataset(n_samples=args.n_samples, seed=args.seed)
    train_samples, val_samples = train_val_split(raw, val_frac=0.15, seed=args.seed)
    print(f"Train: {len(train_samples)}  Val: {len(val_samples)}")

    vocab = Vocab([t for t, _ in train_samples])
    print(f"Vocab size: {len(vocab)}")

    train_ds = SentimentDataset(train_samples, vocab, max_len=args.max_len)
    val_ds = SentimentDataset(val_samples, vocab, max_len=args.max_len)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch
    )

    model = LSTMSentimentClassifier(
        vocab_size=len(vocab),
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    os.makedirs(args.out_dir, exist_ok=True)
    best_val_acc = 0.0
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, optimizer, criterion, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, optimizer, criterion, device, train=False)
        dt = time.time() - t0

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val_loss {val_loss:.4f} acc {val_acc:.4f} | {dt:.1f}s"
        )
        history.append(
            {"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
             "val_loss": val_loss, "val_acc": val_acc}
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "vocab_itos": vocab.itos,
                    "config": {
                        "embed_dim": args.embed_dim,
                        "hidden_dim": args.hidden_dim,
                        "num_layers": args.num_layers,
                        "max_len": args.max_len,
                    },
                },
                os.path.join(args.out_dir, "best_model.pt"),
            )

    with open(os.path.join(args.out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nBest validation accuracy: {best_val_acc:.4f}")
    print(f"Checkpoint saved to {args.out_dir}/best_model.pt")


if __name__ == "__main__":
    main()
