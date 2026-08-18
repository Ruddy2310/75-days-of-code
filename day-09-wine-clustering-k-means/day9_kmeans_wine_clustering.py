"""
Day 9 - 75 Days of Code Challenge
Classic ML Mini-Project #4: Wine Clustering (Unsupervised Learning)
Author: Rudra (Ruddy2310)

Goal: Group wines into clusters based on their chemical properties using
K-Means, without using the known cultivar labels — then check how well
the discovered clusters line up with the real wine classes.

Dataset: Wine recognition dataset — built into scikit-learn,
no external download required.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, confusion_matrix

RANDOM_STATE = 42

# -----------------------------
# 1. Load & explore the data
# -----------------------------
data = load_wine()
X = pd.DataFrame(data.data, columns=data.feature_names)
y_true = pd.Series(data.target, name="cultivar")  # true labels, used only for evaluation

print("Dataset shape:", X.shape)
print("Number of true classes (cultivars):", y_true.nunique())
print("\nMissing values:", X.isnull().sum().sum())

# -----------------------------
# 2. Scale features
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# 3. Find the right number of clusters (Elbow + Silhouette)
# -----------------------------
inertias = []
sil_scores = []
k_range = range(2, 8)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, labels))

best_k = list(k_range)[int(np.argmax(sil_scores))]
print(f"\nBest k by silhouette score: {best_k}")

# -----------------------------
# 4. Fit final K-Means model
# -----------------------------
kmeans = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10)
cluster_labels = kmeans.fit_predict(X_scaled)

sil = silhouette_score(X_scaled, cluster_labels)
ari = adjusted_rand_score(y_true, cluster_labels)
print(f"Silhouette Score: {sil:.4f}")
print(f"Adjusted Rand Index (vs true cultivars): {ari:.4f}")

# -----------------------------
# 5. Reduce to 2D with PCA for visualization
# -----------------------------
pca = PCA(n_components=2, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X_scaled)
print(f"\nExplained variance by 2 PCA components: {pca.explained_variance_ratio_.sum():.2%}")

# -----------------------------
# 6. Visualizations
# -----------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Elbow + Silhouette plot
ax2 = axes[0].twinx()
axes[0].plot(list(k_range), inertias, marker="o", color="tab:blue", label="Inertia")
ax2.plot(list(k_range), sil_scores, marker="s", color="tab:orange", label="Silhouette")
axes[0].set_xlabel("Number of clusters (k)")
axes[0].set_ylabel("Inertia", color="tab:blue")
ax2.set_ylabel("Silhouette Score", color="tab:orange")
axes[0].set_title("Elbow Method & Silhouette Score")
axes[0].axvline(best_k, color="gray", linestyle="--", alpha=0.6)

# PCA scatter colored by cluster
scatter = axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap="viridis", s=50, edgecolor="k", alpha=0.8)
axes[1].set_xlabel("PCA Component 1")
axes[1].set_ylabel("PCA Component 2")
axes[1].set_title(f"K-Means Clusters (k={best_k}) - PCA Projection")
plt.colorbar(scatter, ax=axes[1], label="Cluster")

plt.tight_layout()
plt.savefig("elbow_and_clusters.png", dpi=150)
plt.close()

# Compare clusters vs true cultivars
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sc1 = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap="viridis", s=50, edgecolor="k")
axes[0].set_title("K-Means Cluster Assignment")
axes[0].set_xlabel("PCA Component 1")
axes[0].set_ylabel("PCA Component 2")

sc2 = axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=y_true, cmap="viridis", s=50, edgecolor="k")
axes[1].set_title("True Wine Cultivars")
axes[1].set_xlabel("PCA Component 1")
axes[1].set_ylabel("PCA Component 2")

plt.tight_layout()
plt.savefig("clusters_vs_true_labels.png", dpi=150)
plt.close()

print("\nSaved plots: elbow_and_clusters.png, clusters_vs_true_labels.png")
print("\nDone. K-Means clustering complete.")
