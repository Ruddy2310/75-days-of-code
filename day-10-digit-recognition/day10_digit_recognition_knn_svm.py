"""
Day 10 - 75 Days of Code Challenge
Classic ML Mini-Project #5: Handwritten Digit Recognition
Author: Rudra (Ruddy2310)

Goal: Classify handwritten digits (0-9) from 8x8 pixel images using
classic ML models - K-Nearest Neighbors vs Support Vector Machine -
and compare their performance.

Dataset: Digits dataset - built into scikit-learn (1797 8x8 images),
no external download required.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

RANDOM_STATE = 42

# -----------------------------
# 1. Load & explore the data
# -----------------------------
data = load_digits()
X = data.data          # 1797 samples x 64 features (8x8 flattened)
y = data.target        # digit labels 0-9
images = data.images   # original 8x8 images for visualization

print("Dataset shape:", X.shape)
print("Number of classes:", len(np.unique(y)))
print("Class distribution:\n", pd.Series(y).value_counts().sort_index())

# -----------------------------
# 2. Visualize a few sample digits
# -----------------------------
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for i, ax in enumerate(axes.flat):
    ax.imshow(images[i], cmap="gray")
    ax.set_title(f"Label: {y[i]}")
    ax.axis("off")
plt.suptitle("Sample Digits from the Dataset")
plt.tight_layout()
plt.savefig("sample_digits.png", dpi=150)
plt.close()

# -----------------------------
# 3. Train/test split + scaling
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# 4. Train models
# -----------------------------
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

svm = SVC(kernel="rbf", gamma="scale", random_state=RANDOM_STATE)
svm.fit(X_train_scaled, y_train)

# -----------------------------
# 5. Evaluate models
# -----------------------------
def evaluate(name, y_true, y_pred):
    print(f"\n--- {name} ---")
    print(f"Accuracy : {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred, average='macro'):.4f}")
    print(f"Recall   : {recall_score(y_true, y_pred, average='macro'):.4f}")
    print(f"F1 Score : {f1_score(y_true, y_pred, average='macro'):.4f}")
    print(classification_report(y_true, y_pred))
    return confusion_matrix(y_true, y_pred)

knn_pred = knn.predict(X_test_scaled)
cm_knn = evaluate("K-Nearest Neighbors", y_test, knn_pred)

svm_pred = svm.predict(X_test_scaled)
cm_svm = evaluate("Support Vector Machine", y_test, svm_pred)

# 5-fold cross-validation for a more robust comparison
knn_cv = cross_val_score(knn, scaler.fit_transform(X), y, cv=5)
svm_cv = cross_val_score(svm, scaler.fit_transform(X), y, cv=5)
print(f"\nKNN 5-fold CV accuracy: {knn_cv.mean():.4f} (+/- {knn_cv.std():.4f})")
print(f"SVM 5-fold CV accuracy: {svm_cv.mean():.4f} (+/- {svm_cv.std():.4f})")

# -----------------------------
# 6. Visualizations
# -----------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.heatmap(cm_knn, annot=True, fmt="d", cmap="Blues", ax=axes[0])
axes[0].set_title("KNN - Confusion Matrix")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")

sns.heatmap(cm_svm, annot=True, fmt="d", cmap="Greens", ax=axes[1])
axes[1].set_title("SVM - Confusion Matrix")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Actual")

plt.tight_layout()
plt.savefig("confusion_matrices.png", dpi=150)
plt.close()

# Misclassified examples
misclassified_idx = np.where(svm_pred != y_test)[0][:10]
if len(misclassified_idx) > 0:
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    for i, ax in enumerate(axes.flat):
        if i < len(misclassified_idx):
            idx = misclassified_idx[i]
            img = X_test[idx].reshape(8, 8)
            ax.imshow(img, cmap="gray")
            ax.set_title(f"True: {y_test[idx]}, Pred: {svm_pred[idx]}")
        ax.axis("off")
    plt.suptitle("SVM Misclassified Examples")
    plt.tight_layout()
    plt.savefig("misclassified_examples.png", dpi=150)
    plt.close()

print("\nSaved plots: sample_digits.png, confusion_matrices.png, misclassified_examples.png")
print("\nDone. KNN vs SVM comparison complete.")
