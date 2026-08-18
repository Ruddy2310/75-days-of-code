"""
generate.py
-----------
Load a trained generator checkpoint and produce new handwritten-digit
images from random noise.

Usage:
    python src/generate.py --checkpoint outputs/checkpoints/gan_epoch050.pt \
        --num-images 64 --output outputs/final_samples.png
"""

import argparse

import torch

from models import Generator
from utils import load_checkpoint, save_sample_grid, set_seed


def main(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )

    generator = Generator(latent_dim=args.latent_dim).to(device)
    epoch = load_checkpoint(generator, None, args.checkpoint, map_location=device)
    generator.eval()
    print(f"Loaded generator from checkpoint at epoch {epoch}")

    z = torch.randn(args.num_images, args.latent_dim, device=device)
    with torch.no_grad():
        images = generator(z)

    save_sample_grid(images, args.output, nrow=int(args.num_images ** 0.5) or 8)
    print(f"Saved {args.num_images} generated digits to {args.output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate MNIST-style digits from a trained GAN")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num-images", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=100)
    parser.add_argument("--output", type=str, default="outputs/final_samples.png")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
