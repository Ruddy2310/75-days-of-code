"""
Day 12 - 75 Days of Code Challenge
Classic ML Mini-Project #7: Ensemble Methods Comparison
Author: Rudra (Ruddy2310)

Goal: Compare three classic ensemble learning strategies - Bagging
(Random Forest), Boosting (AdaBoost), and Gradient Boosting - on the
same binary classification problem, to see how each approach to
combining weak learners performs.

Dataset: Synthetically generated with sklearn's make_classification
(no external download required, fully reproducible with a fixed seed).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, roc_auc_score
)

RANDOM_STATE = 42

# -----------------------------
# 1. Generate a synthetic dataset
# -----------------------------
X, y = make_classification(
    n_samples=1000, n_features=20, n_informative=10, n_redundant=5,
    n_clusters_per_class=2, flip_y=0.03, random_state=RANDOM_STATE
)
feature_names = [f"feature_{i}" for i in range(X.shape[1])]
X = pd.DataFrame(X, columns=feature_names)
y = pd.Series(y, name="target")

print("Dataset shape:", X.shape)
print("Class distribution:\n", y.value_counts())

# -----------------------------
# 2. Train/test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# -----------------------------
# 3. Train three ensemble models
# -----------------------------
models = {
    "Random Forest (Bagging)": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
    "AdaBoost (Boosting)": AdaBoostClassifier(n_estimators=200, random_state=RANDOM_STATE),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, random_state=RANDOM_STATE),
}

results = {}
predictions = {}
probabilities = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    predictions[name] = pred
    probabilities[name] = proba

    acc = accuracy_score(y_test, pred)
    prec = precision_score(y_test, pred)
    rec = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    auc = roc_auc_score(y_test, proba)
    cv_scores = cross_val_score(model, X, y, cv=5)

    results[name] = {
        "Accuracy": acc, "Precision": prec, "Recall": rec,
        "F1": f1, "ROC-AUC": auc, "CV Mean": cv_scores.mean(), "CV Std": cv_scores.std()
    }

    print(f"\n--- {name} ---")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {auc:.4f}")
    print(f"5-fold CV: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

results_df = pd.DataFrame(results).T
print("\n=== Summary Comparison ===")
print(results_df.round(4))

# -----------------------------
# 4. Visualizations
# -----------------------------
# ROC curves for all three models
plt.figure(figsize=(7, 6))
for name, proba in probabilities.items():
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc = roc_auc_score(y_test, proba)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison - Ensemble Methods")
plt.legend()
plt.tight_layout()
plt.savefig("roc_curve_comparison.png", dpi=150)
plt.close()

# Confusion matrices side by side
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
cmaps = ["Blues", "Oranges", "Greens"]
for ax, (name, pred), cmap in zip(axes, predictions.items(), cmaps):
    cm = confusion_matrix(y_test, pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, ax=ax)
    ax.set_title(name)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrices.png", dpi=150)
plt.close()

# Feature importance comparison (top 10)
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, (name, model) in zip(axes, models.items()):
    importances = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False).head(10)
    sns.barplot(x=importances.values, y=importances.index, ax=ax, palette="viridis")
    ax.set_title(f"{name}\nTop 10 Feature Importances")
    ax.set_xlabel("Importance")
plt.tight_layout()
plt.savefig("feature_importance_comparison.png", dpi=150)
plt.close()

# Bar chart comparing accuracy/F1/AUC across models
metrics_to_plot = ["Accuracy", "F1", "ROC-AUC"]
results_df[metrics_to_plot].plot(kind="bar", figsize=(9, 5), colormap="viridis")
plt.title("Ensemble Method Performance Comparison")
plt.ylabel("Score")
plt.xticks(rotation=15)
plt.ylim(0, 1)
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("performance_comparison.png", dpi=150)
plt.close()

print("\nSaved plots: roc_curve_comparison.png, confusion_matrices.png,")
print("feature_importance_comparison.png, performance_comparison.png")
print("\nDone. Random Forest vs AdaBoost vs Gradient Boosting comparison complete.")
