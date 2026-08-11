# Day 20 — Deep Learning Basics: NN Experiment #5 (Autoencoder)

Part of my #75DaysOfCode challenge. First unsupervised deep learning architecture.

## What it does
Builds an autoencoder that compresses 64-pixel digit images down to
just 8 numbers (latent space) and reconstructs them - the target is
the input itself, no labels used. Also trains a denoising variant that
maps noisy images back to their clean originals.

## Architecture
64 -> 32 -> 8 (latent) -> 32 -> 64, all Dense layers

## Results
- Plain AE test reconstruction MSE: 0.023
- Denoising AE test MSE (from noisy input): 0.039
- Latent space naturally clusters by digit despite never seeing labels

## Files
- day20_autoencoder.py — main script (both plain + denoising autoencoders)
- reconstruction_comparison.png — original vs reconstructed digits
- denoising_comparison.png — clean vs noisy vs denoised digits
- latent_space_visualization.png — 8D latent space, PCA-projected, colored by digit
- training_curves.png — loss curves for both autoencoders
