"""
Day 8 - 75 Days of Code Challenge
Classic ML Mini-Project #3: Breast Cancer Classification
Author: Rudra (Ruddy2310)

Goal: Predict whether a tumor is malignant or benign using classic
supervised ML models (Logistic Regression vs Random Forest), and
compare their performance using standard classification metrics.

Dataset: Breast Cancer Wisconsin (Diagnostic) — built into scikit-learn,
no external download required.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, roc_auc_score
)

RANDOM_STATE = 42

# -----------------------------
# 1. Load & explore the data
# -----------------------------
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="target")  # 0 = malignant, 1 = benign

print("Dataset shape:", X.shape)
print("\nClass distribution:")
print(y.value_counts().rename({0: "malignant", 1: "benign"}))
print("\nMissing values:", X.isnull().sum().sum())

# -----------------------------
# 2. Train/test split + scaling
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# 3. Train models
# -----------------------------
log_reg = LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)
log_reg.fit(X_train_scaled, y_train)

rf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)
rf.fit(X_train, y_train)  # tree models don't need scaling

# -----------------------------
# 4. Evaluate models
# -----------------------------
def evaluate(name, y_true, y_pred, y_proba):
    print(f"\n--- {name} ---")
    print(f"Accuracy : {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall   : {recall_score(y_true, y_pred):.4f}")
    print(f"F1 Score : {f1_score(y_true, y_pred):.4f}")
    print(f"ROC-AUC  : {roc_auc_score(y_true, y_proba):.4f}")
    print(classification_report(y_true, y_pred, target_names=["malignant", "benign"]))
    return confusion_matrix(y_true, y_pred)

log_pred = log_reg.predict(X_test_scaled)
log_proba = log_reg.predict_proba(X_test_scaled)[:, 1]
cm_log = evaluate("Logistic Regression", y_test, log_pred, log_proba)

rf_pred = rf.predict(X_test)
rf_proba = rf.predict_proba(X_test)[:, 1]
cm_rf = evaluate("Random Forest", y_test, rf_pred, rf_proba)

# -----------------------------
# 5. Visualizations
# -----------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.heatmap(cm_log, annot=True, fmt="d", cmap="Blues",
            xticklabels=["malignant", "benign"],
            yticklabels=["malignant", "benign"], ax=axes[0])
axes[0].set_title("Logistic Regression - Confusion Matrix")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")

sns.heatmap(cm_rf, annot=True, fmt="d", cmap="Greens",
            xticklabels=["malignant", "benign"],
            yticklabels=["malignant", "benign"], ax=axes[1])
axes[1].set_title("Random Forest - Confusion Matrix")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Actual")

plt.tight_layout()
plt.savefig("confusion_matrices.png", dpi=150)
plt.close()

# ROC curves
plt.figure(figsize=(6, 6))
for name, proba in [("Logistic Regression", log_proba), ("Random Forest", rf_proba)]:
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc = roc_auc_score(y_test, proba)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.tight_layout()
plt.savefig("roc_curve.png", dpi=150)
plt.close()

# Feature importance (Random Forest)
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False).head(10)
plt.figure(figsize=(8, 5))
sns.barplot(x=importances.values, y=importances.index, palette="viridis")
plt.title("Top 10 Feature Importances - Random Forest")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
plt.close()

print("\nSaved plots: confusion_matrices.png, roc_curve.png, feature_importance.png")
print("\nDone. Random Forest and Logistic Regression comparison complete.")
