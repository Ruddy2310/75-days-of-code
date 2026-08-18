"""
utils.py
--------
Small helpers used by train.py / generate.py:
- saving a grid of generated sample images to disk
- saving / loading model checkpoints
- setting a global random seed for reproducibility
"""

import os
import random

import numpy as np
import torch
import torchvision.utils as vutils
from PIL import Image


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_sample_grid(images: torch.Tensor, path: str, nrow: int = 8) -> None:
    """
    images: tensor of shape (N, 1, 28, 28) with values in [-1, 1]
    Saves a single PNG grid to `path`.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    images = (images.clamp(-1, 1) + 1) / 2  # rescale [-1,1] -> [0,1]
    grid = vutils.make_grid(images, nrow=nrow, padding=2)
    ndarr = grid.mul(255).byte().permute(1, 2, 0).cpu().numpy()
    Image.fromarray(ndarr).save(path)


def save_checkpoint(generator, discriminator, epoch: int, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "generator_state_dict": generator.state_dict(),
            "discriminator_state_dict": discriminator.state_dict(),
        },
        path,
    )


def load_checkpoint(generator, discriminator, path: str, map_location="cpu") -> int:
    ckpt = torch.load(path, map_location=map_location)
    generator.load_state_dict(ckpt["generator_state_dict"])
    if discriminator is not None and "discriminator_state_dict" in ckpt:
        discriminator.load_state_dict(ckpt["discriminator_state_dict"])
    return ckpt.get("epoch", 0)
