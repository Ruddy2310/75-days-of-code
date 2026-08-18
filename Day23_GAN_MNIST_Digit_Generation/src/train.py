"""
train.py
--------
Trains a simple GAN on MNIST.

Usage:
    python src/train.py --epochs 50 --batch-size 128 --latent-dim 100

Every `--sample-interval` batches, a grid of generated digits is saved to
outputs/samples/. A checkpoint is saved at the end of every epoch to
outputs/checkpoints/.

Loss curves (generator vs discriminator) are logged to
outputs/loss_log.csv so they can be plotted later (see README).
"""

import argparse
import csv
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models import Discriminator, Generator
from utils import save_checkpoint, save_sample_grid, set_seed


def get_dataloader(batch_size: int, data_dir: str) -> DataLoader:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),  # -> pixels in [-1, 1]
        ]
    )
    dataset = datasets.MNIST(
        root=data_dir, train=True, download=True, transform=transform
    )
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True
    )


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    print(f"Using device: {device}")

    img_shape = (1, 28, 28)
    dataloader = get_dataloader(args.batch_size, args.data_dir)

    generator = Generator(latent_dim=args.latent_dim, img_shape=img_shape).to(device)
    discriminator = Discriminator(img_shape=img_shape).to(device)

    adversarial_loss = nn.BCELoss()

    opt_g = torch.optim.Adam(
        generator.parameters(), lr=args.lr, betas=(args.b1, args.b2)
    )
    opt_d = torch.optim.Adam(
        discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2)
    )

    # Fixed noise vector so we can watch the *same* digits evolve over training
    fixed_noise = torch.randn(64, args.latent_dim, device=device)

    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, "loss_log.csv")
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "batch", "loss_D", "loss_G"])

    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        for i, (imgs, _) in enumerate(dataloader):
            batch_size = imgs.size(0)
            real_imgs = imgs.to(device)

            valid = torch.ones(batch_size, 1, device=device)
            fake = torch.zeros(batch_size, 1, device=device)

            # ---------------------
            #  Train Generator
            # ---------------------
            opt_g.zero_grad()
            z = torch.randn(batch_size, args.latent_dim, device=device)
            gen_imgs = generator(z)
            # Generator wants the discriminator to call its fakes "real"
            g_loss = adversarial_loss(discriminator(gen_imgs), valid)
            g_loss.backward()
            opt_g.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------
            opt_d.zero_grad()
            real_loss = adversarial_loss(discriminator(real_imgs), valid)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            opt_d.step()

            batches_done = (epoch - 1) * len(dataloader) + i
            if batches_done % args.log_interval == 0:
                elapsed = time.time() - start_time
                print(
                    f"[Epoch {epoch}/{args.epochs}] [Batch {i}/{len(dataloader)}] "
                    f"[D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}] "
                    f"[{elapsed:.1f}s elapsed]"
                )
                with open(log_path, "a", newline="") as f:
                    csv.writer(f).writerow(
                        [epoch, batches_done, d_loss.item(), g_loss.item()]
                    )

            if batches_done % args.sample_interval == 0:
                generator.eval()
                with torch.no_grad():
                    samples = generator(fixed_noise)
                save_sample_grid(
                    samples,
                    os.path.join(
                        args.output_dir, "samples", f"epoch{epoch:03d}_batch{i:04d}.png"
                    ),
                )
                generator.train()

        # End-of-epoch checkpoint
        save_checkpoint(
            generator,
            discriminator,
            epoch,
            os.path.join(args.output_dir, "checkpoints", f"gan_epoch{epoch:03d}.pt"),
        )

    print("Training complete.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a simple GAN on MNIST")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4, help="Adam learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="Adam beta1")
    parser.add_argument("--b2", type=float, default=0.999, help="Adam beta2")
    parser.add_argument("--latent-dim", type=int, default=100)
    parser.add_argument("--sample-interval", type=int, default=200)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--output-dir", type=str, default="./outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if a GPU is available")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
