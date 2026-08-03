"""
Day 14 - 75 Days of Code Challenge
Classic ML Mini-Project #9: Anomaly Detection Comparison
Author: Rudra (Ruddy2310)

Goal: Detect anomalies/outliers in a high-dimensional dataset using
two unsupervised approaches - Isolation Forest (tree-based, efficient
for high dimensions) and Local Outlier Factor (density-based, sensitive
to local clustering). Compare their performance on a credit card fraud
detection scenario.

Dataset: Synthetically generated anomalies mixed with normal data
(no download required, fully reproducible).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve
)
from sklearn.decomposition import PCA

RANDOM_STATE = 42

# -----------------------------
# 1. Generate synthetic anomaly dataset
# -----------------------------
# Create mostly normal data with a few clusters of anomalies
X_normal, _ = make_blobs(n_samples=900, n_features=8, centers=3, 
                          cluster_std=0.5, random_state=RANDOM_STATE)
X_anomalies, _ = make_blobs(n_samples=100, n_features=8, centers=1,
                             cluster_std=2.0, center_box=(-8, 8),
                             random_state=RANDOM_STATE+1)

X = np.vstack([X_normal, X_anomalies])
y_true = np.hstack([np.zeros(900), np.ones(100)])  # 0 = normal, 1 = anomaly

df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
df['is_anomaly'] = y_true

print("Dataset shape:", X.shape)
print("Class distribution:")
print(f"  Normal: {(y_true == 0).sum()} (90%)")
print(f"  Anomalies: {(y_true == 1).sum()} (10%)")

# Scale for LOF (distance-based), keep unscaled for Isolation Forest
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# 2. Train anomaly detection models
# -----------------------------
iso_forest = IsolationForest(contamination=0.1, random_state=RANDOM_STATE)
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)

iso_forest.fit(X)  # fit the model
iso_pred = iso_forest.predict(X)  # 1 = normal, -1 = anomaly (sklearn convention)
iso_scores = iso_forest.score_samples(X)  # lower = more anomalous

lof_pred = lof.fit_predict(X_scaled)  # 1 = normal, -1 = anomaly
lof_scores = lof.negative_outlier_factor_  # higher = more normal

# Convert to 0/1 for consistency with y_true (0 = normal, 1 = anomaly)
iso_pred_binary = (iso_pred == -1).astype(int)
lof_pred_binary = (lof_pred == -1).astype(int)

# Convert scores to anomaly probability (0 = normal, 1 = anomaly)
iso_scores_prob = 1 / (1 + np.exp(iso_scores))  # sigmoid normalization
lof_scores_prob = 1 / (1 + lof_scores)         # same

# -----------------------------
# 3. Evaluate models
# -----------------------------
def evaluate(name, y_true, y_pred, y_scores):
    print(f"\n--- {name} ---")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred):.4f}")
    print(f"F1 Score:  {f1_score(y_true, y_pred):.4f}")
    print(f"ROC-AUC:   {roc_auc_score(y_true, y_scores):.4f}")
    print(classification_report(y_true, y_pred, target_names=["Normal", "Anomaly"]))
    return confusion_matrix(y_true, y_pred)

cm_iso = evaluate("Isolation Forest", y_true, iso_pred_binary, iso_scores_prob)
cm_lof = evaluate("Local Outlier Factor", y_true, lof_pred_binary, lof_scores_prob)

# -----------------------------
# 4. Visualizations
# -----------------------------
# PCA reduction to 2D for visualization
pca = PCA(n_components=2, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Isolation Forest - predictions
scatter1 = axes[0, 0].scatter(X_pca[:, 0], X_pca[:, 1], c=iso_pred_binary,
                               cmap="RdYlBu", s=30, alpha=0.6, edgecolor="k")
axes[0, 0].set_title("Isolation Forest - Detected Anomalies")
axes[0, 0].set_xlabel("PCA Component 1")
axes[0, 0].set_ylabel("PCA Component 2")
plt.colorbar(scatter1, ax=axes[0, 0], label="Anomaly (1) / Normal (0)")

# Isolation Forest - anomaly scores
scatter2 = axes[0, 1].scatter(X_pca[:, 0], X_pca[:, 1], c=iso_scores_prob,
                               cmap="YlOrRd", s=30, alpha=0.6, edgecolor="k")
axes[0, 1].set_title("Isolation Forest - Anomaly Scores")
axes[0, 1].set_xlabel("PCA Component 1")
axes[0, 1].set_ylabel("PCA Component 2")
plt.colorbar(scatter2, ax=axes[0, 1], label="Anomaly Probability")

# LOF - predictions
scatter3 = axes[1, 0].scatter(X_pca[:, 0], X_pca[:, 1], c=lof_pred_binary,
                               cmap="RdYlBu", s=30, alpha=0.6, edgecolor="k")
axes[1, 0].set_title("Local Outlier Factor - Detected Anomalies")
axes[1, 0].set_xlabel("PCA Component 1")
axes[1, 0].set_ylabel("PCA Component 2")
plt.colorbar(scatter3, ax=axes[1, 0], label="Anomaly (1) / Normal (0)")

# LOF - anomaly scores
scatter4 = axes[1, 1].scatter(X_pca[:, 0], X_pca[:, 1], c=lof_scores_prob,
                               cmap="YlOrRd", s=30, alpha=0.6, edgecolor="k")
axes[1, 1].set_title("Local Outlier Factor - Anomaly Scores")
axes[1, 1].set_xlabel("PCA Component 1")
axes[1, 1].set_ylabel("PCA Component 2")
plt.colorbar(scatter4, ax=axes[1, 1], label="Anomaly Probability")

plt.suptitle("Anomaly Detection: Isolation Forest vs LOF (PCA 2D projection)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("anomaly_detection_comparison.png", dpi=150)
plt.close()

# Confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.heatmap(cm_iso, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Normal", "Anomaly"],
            yticklabels=["Normal", "Anomaly"], ax=axes[0])
axes[0].set_title("Isolation Forest - Confusion Matrix")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")

sns.heatmap(cm_lof, annot=True, fmt="d", cmap="Greens",
            xticklabels=["Normal", "Anomaly"],
            yticklabels=["Normal", "Anomaly"], ax=axes[1])
axes[1].set_title("Local Outlier Factor - Confusion Matrix")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Actual")

plt.tight_layout()
plt.savefig("confusion_matrices.png", dpi=150)
plt.close()

# ROC curves
plt.figure(figsize=(7, 6))
iso_fpr, iso_tpr, _ = roc_curve(y_true, iso_scores_prob)
lof_fpr, lof_tpr, _ = roc_curve(y_true, lof_scores_prob)

iso_auc = roc_auc_score(y_true, iso_scores_prob)
lof_auc = roc_auc_score(y_true, lof_scores_prob)

plt.plot(iso_fpr, iso_tpr, label=f"Isolation Forest (AUC = {iso_auc:.3f})")
plt.plot(lof_fpr, lof_tpr, label=f"LOF (AUC = {lof_auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison - Anomaly Detection")
plt.legend()
plt.tight_layout()
plt.savefig("roc_curve_comparison.png", dpi=150)
plt.close()

print("\nSaved plots: anomaly_detection_comparison.png, confusion_matrices.png, roc_curve_comparison.png")
print("\nDone. Isolation Forest vs LOF comparison complete.")
