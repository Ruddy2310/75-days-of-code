# Day 24 — Deep Learning Basics: NN Experiment #9 (Siamese Network)

Part of my #75DaysOfCode challenge. Similarity learning architecture -
twin networks with shared weights, trained with contrastive loss to
learn whether two inputs are the same class or different.

## What it does
Two identical sub-networks (shared weights) each embed an input into
a feature vector. Contrastive loss pulls embeddings of same-class pairs
together and pushes different-class pairs apart, without ever training
a direct classifier - the network learns a similarity metric instead.
Same idea used in real face/signature verification systems.

## Architecture
Shared twin encoder: 64 (image) -> 32 -> 16 -> 2 (embedding)
Contrastive loss compares embedding distance for same/different pairs

## Results
- Trained 400 epochs, contrastive loss dropped from 0.011 -> 0.0001
- Embedding space shows tight, well-separated clusters for circle vs square
- Same-class pairs collapse to near-zero distance; different-class pairs
  stay near/above the margin (1.0)

## Files
- day24_siamese_network.py — main script (twin network + contrastive loss, pure numpy)
- siamese_training_curve.png — contrastive loss over training
- embedding_space_visualization.png — learned 2D embedding space, colored by class
- similarity_pairs_comparison.png — example pairs with predicted distance and SAME/DIFF label
