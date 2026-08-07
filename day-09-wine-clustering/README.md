# Day 9 — Classic ML Mini-Project #4: Wine Clustering (Unsupervised)

Part of my #75DaysOfCode challenge.

## What it does
Groups wines into clusters based on chemical properties using K-Means,
without using the true cultivar labels — then checks how well the
discovered clusters match reality.

## Results
- Best k (via silhouette score): 3
- Adjusted Rand Index vs true cultivars: 0.90
- Uses sklearn's built-in wine dataset, no download needed

## Files
- day9_kmeans_wine_clustering.py — main script
- elbow_and_clusters.png — elbow method + silhouette + PCA cluster view
- clusters_vs_true_labels.png — cluster assignment vs ground truth
