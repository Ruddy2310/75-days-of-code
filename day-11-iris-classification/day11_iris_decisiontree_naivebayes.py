"""
Day 11 - 75 Days of Code Challenge
Classic ML Mini-Project #6: Iris Flower Classification
Author: Rudra (Ruddy2310)

Goal: Classify iris flowers into 3 species using Decision Tree vs
Gaussian Naive Bayes, compare performance, and visualize decision
boundaries using two features.

Dataset: Iris dataset - built into scikit-learn, no download required.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

RANDOM_STATE = 42

# -----------------------------
# 1. Load & explore the data
# -----------------------------
data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="species")
species_names = data.target_names

print("Dataset shape:", X.shape)
print("Classes:", list(species_names))
print("\nClass distribution:")
print(y.value_counts().sort_index())
print("\nMissing values:", X.isnull().sum().sum())

# -----------------------------
# 2. Train/test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# -----------------------------
# 3. Train models
# -----------------------------
dtree = DecisionTreeClassifier(max_depth=4, random_state=RANDOM_STATE)
dtree.fit(X_train, y_train)

nb = GaussianNB()
nb.fit(X_train, y_train)

# -----------------------------
# 4. Evaluate models
# -----------------------------
def evaluate(name, y_true, y_pred):
    print(f"\n--- {name} ---")
    print(f"Accuracy : {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred, average='macro'):.4f}")
    print(f"Recall   : {recall_score(y_true, y_pred, average='macro'):.4f}")
    print(f"F1 Score : {f1_score(y_true, y_pred, average='macro'):.4f}")
    print(classification_report(y_true, y_pred, target_names=species_names))
    return confusion_matrix(y_true, y_pred)

dtree_pred = dtree.predict(X_test)
cm_dtree = evaluate("Decision Tree", y_test, dtree_pred)

nb_pred = nb.predict(X_test)
cm_nb = evaluate("Gaussian Naive Bayes", y_test, nb_pred)

# Cross-validation for robustness
dtree_cv = cross_val_score(dtree, X, y, cv=5)
nb_cv = cross_val_score(nb, X, y, cv=5)
print(f"\nDecision Tree 5-fold CV accuracy: {dtree_cv.mean():.4f} (+/- {dtree_cv.std():.4f})")
print(f"Naive Bayes 5-fold CV accuracy: {nb_cv.mean():.4f} (+/- {nb_cv.std():.4f})")

# -----------------------------
# 5. Visualizations
# -----------------------------
# Confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.heatmap(cm_dtree, annot=True, fmt="d", cmap="Blues",
            xticklabels=species_names, yticklabels=species_names, ax=axes[0])
axes[0].set_title("Decision Tree - Confusion Matrix")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")

sns.heatmap(cm_nb, annot=True, fmt="d", cmap="Greens",
            xticklabels=species_names, yticklabels=species_names, ax=axes[1])
axes[1].set_title("Naive Bayes - Confusion Matrix")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Actual")

plt.tight_layout()
plt.savefig("confusion_matrices.png", dpi=150)
plt.close()

# Decision tree structure
plt.figure(figsize=(16, 8))
plot_tree(dtree, feature_names=data.feature_names, class_names=species_names,
          filled=True, rounded=True, fontsize=9)
plt.title("Decision Tree Structure (max_depth=4)")
plt.tight_layout()
plt.savefig("decision_tree_structure.png", dpi=150)
plt.close()

# Decision boundary using 2 features (petal length & width - most discriminative)
feat_x, feat_y = "petal length (cm)", "petal width (cm)"
X2 = X[[feat_x, feat_y]].values

dtree2 = DecisionTreeClassifier(max_depth=4, random_state=RANDOM_STATE).fit(X2, y)
nb2 = GaussianNB().fit(X2, y)

x_min, x_max = X2[:, 0].min() - 0.5, X2[:, 0].max() + 0.5
y_min, y_max = X2[:, 1].min() - 0.5, X2[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, model, title in zip(axes, [dtree2, nb2], ["Decision Tree", "Naive Bayes"]):
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3, cmap="viridis")
    scatter = ax.scatter(X2[:, 0], X2[:, 1], c=y, cmap="viridis", edgecolor="k", s=40)
    ax.set_xlabel(feat_x)
    ax.set_ylabel(feat_y)
    ax.set_title(f"{title} - Decision Boundary")

plt.tight_layout()
plt.savefig("decision_boundaries.png", dpi=150)
plt.close()

print("\nSaved plots: confusion_matrices.png, decision_tree_structure.png, decision_boundaries.png")
print("\nDone. Decision Tree vs Naive Bayes comparison complete.")
