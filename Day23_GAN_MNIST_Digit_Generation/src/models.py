"""
models.py
---------
Generator and Discriminator networks for a simple fully-connected
GAN trained on MNIST (28x28 grayscale handwritten digits).

Architecture choices are intentionally simple (MLPs, not conv layers)
so the project trains quickly on CPU and is easy to read end-to-end.
See README.md for a DCGAN upgrade path.
"""

import torch
import torch.nn as nn


class Generator(nn.Module):
    """
    Maps a random noise vector z ~ N(0, 1) of size `latent_dim`
    to a fake MNIST image of shape (1, 28, 28), pixels in [-1, 1].
    """

    def __init__(self, latent_dim: int = 100, img_shape: tuple = (1, 28, 28)):
        super().__init__()
        self.img_shape = img_shape
        img_size = int(torch.prod(torch.tensor(img_shape)).item())

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, img_size),
            nn.Tanh(),  # output in [-1, 1] to match normalized MNIST
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img


class Discriminator(nn.Module):
    """
    Binary classifier: takes an image (real or fake) and outputs the
    probability that it is real, in [0, 1].
    """

    def __init__(self, img_shape: tuple = (1, 28, 28)):
        super().__init__()
        img_size = int(torch.prod(torch.tensor(img_shape)).item())

        self.model = nn.Sequential(
            nn.Linear(img_size, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity
