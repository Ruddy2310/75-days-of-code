"""
plot_losses.py
--------------
Reads outputs/loss_log.csv (written by train.py) and plots the
generator / discriminator loss curves over training.

Usage:
    python src/plot_losses.py --log outputs/loss_log.csv --output outputs/loss_curve.png
"""

import argparse
import csv
import os

import matplotlib.pyplot as plt


def main(args: argparse.Namespace) -> None:
    batches, d_losses, g_losses = [], [], []
    with open(args.log, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            batches.append(int(row["batch"]))
            d_losses.append(float(row["loss_D"]))
            g_losses.append(float(row["loss_G"]))

    plt.figure(figsize=(8, 5))
    plt.plot(batches, d_losses, label="Discriminator loss")
    plt.plot(batches, g_losses, label="Generator loss")
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.title("GAN Training Losses")
    plt.legend()
    plt.tight_layout()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.savefig(args.output, dpi=150)
    print(f"Saved loss curve to {args.output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot GAN loss curves")
    parser.add_argument("--log", type=str, default="outputs/loss_log.csv")
    parser.add_argument("--output", type=str, default="outputs/loss_curve.png")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
