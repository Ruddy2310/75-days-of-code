"""
Day 15 - 75 Days of Code Challenge
Classic ML Mini-Project #10: Hyperparameter Tuning Capstone
Author: Rudra (Ruddy2310)

Goal: Wrap up Phase 1 (Classic ML) with a proper end-to-end workflow -
build a scikit-learn Pipeline (scaler + model), tune it with
GridSearchCV across multiple models and hyperparameters, and compare
tuned performance against untuned defaults, to show why hyperparameter
search matters.

Dataset: Synthetically generated with make_classification
(no download required, fully reproducible with a fixed seed).
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

RANDOM_STATE = 42

# -----------------------------
# 1. Generate dataset
# -----------------------------
X, y = make_classification(
    n_samples=1200, n_features=15, n_informative=8, n_redundant=4,
    n_clusters_per_class=2, flip_y=0.05, random_state=RANDOM_STATE
)
X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
y = pd.Series(y, name="target")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

print("Dataset shape:", X.shape)
print("Class balance:\n", y.value_counts(normalize=True))

# -----------------------------
# 2. Build pipelines + hyperparameter grids for 3 model families
# -----------------------------
pipelines_and_grids = {
    "SVM": (
        Pipeline([("scaler", StandardScaler()), ("clf", SVC(probability=True, random_state=RANDOM_STATE))]),
        {"clf__C": [0.1, 1, 10, 100], "clf__gamma": ["scale", 0.01, 0.1], "clf__kernel": ["rbf", "linear"]}
    ),
    "Random Forest": (
        Pipeline([("scaler", StandardScaler()), ("clf", RandomForestClassifier(random_state=RANDOM_STATE))]),
        {"clf__n_estimators": [100, 200], "clf__max_depth": [None, 10, 20], "clf__min_samples_split": [2, 5]}
    ),
    "Logistic Regression": (
        Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=5000, random_state=RANDOM_STATE))]),
        {"clf__C": [0.01, 0.1, 1, 10], "clf__penalty": ["l2"]}
    ),
}

# -----------------------------
# 3. Run GridSearchCV for each model, compare tuned vs default
# -----------------------------
results = []
best_estimators = {}

for name, (pipeline, param_grid) in pipelines_and_grids.items():
    print(f"\n{'='*50}\nTuning {name}...")

    # Baseline: default hyperparameters, no tuning
    default_pipeline = pipeline
    default_pipeline.fit(X_train, y_train)
    default_pred = default_pipeline.predict(X_test)
    default_acc = accuracy_score(y_test, default_pred)
    default_f1 = f1_score(y_test, default_pred)

    # Tuned: GridSearchCV
    start = time.time()
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring="f1", n_jobs=-1)
    grid_search.fit(X_train, y_train)
    elapsed = time.time() - start

    best_estimators[name] = grid_search.best_estimator_
    tuned_pred = grid_search.best_estimator_.predict(X_test)
    tuned_acc = accuracy_score(y_test, tuned_pred)
    tuned_f1 = f1_score(y_test, tuned_pred)

    print(f"Best params: {grid_search.best_params_}")
    print(f"Grid search time: {elapsed:.1f}s across {len(grid_search.cv_results_['params'])} combinations")
    print(f"Default -> Accuracy: {default_acc:.4f}, F1: {default_f1:.4f}")
    print(f"Tuned   -> Accuracy: {tuned_acc:.4f}, F1: {tuned_f1:.4f}")

    results.append({
        "Model": name, "Default Accuracy": default_acc, "Tuned Accuracy": tuned_acc,
        "Default F1": default_f1, "Tuned F1": tuned_f1,
        "Improvement (F1)": tuned_f1 - default_f1
    })

results_df = pd.DataFrame(results)
print(f"\n{'='*50}\n=== Final Comparison ===")
print(results_df.round(4))

best_overall = results_df.loc[results_df["Tuned F1"].idxmax(), "Model"]
print(f"\nBest overall model: {best_overall}")

# -----------------------------
# 4. Visualizations
# -----------------------------
# Default vs Tuned performance comparison
fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(results_df))
width = 0.35
ax.bar(x - width/2, results_df["Default F1"], width, label="Default", color="lightcoral")
ax.bar(x + width/2, results_df["Tuned F1"], width, label="Tuned (GridSearchCV)", color="seagreen")
ax.set_xticks(x)
ax.set_xticklabels(results_df["Model"])
ax.set_ylabel("F1 Score")
ax.set_title("Default vs Tuned Hyperparameters - F1 Score Comparison")
ax.legend()
ax.set_ylim(0, 1)
plt.tight_layout()
plt.savefig("default_vs_tuned_comparison.png", dpi=150)
plt.close()

# Confusion matrix for best overall model
best_model = best_estimators[best_overall]
best_pred = best_model.predict(X_test)
cm = confusion_matrix(y_test, best_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title(f"Best Model ({best_overall}) - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("best_model_confusion_matrix.png", dpi=150)
plt.close()

# Learning curve for the best model - shows if more data would help
train_sizes, train_scores, val_scores = learning_curve(
    best_model, X, y, cv=5, scoring="f1",
    train_sizes=np.linspace(0.1, 1.0, 8), random_state=RANDOM_STATE
)

plt.figure(figsize=(8, 5))
plt.plot(train_sizes, train_scores.mean(axis=1), "o-", label="Training score")
plt.plot(train_sizes, val_scores.mean(axis=1), "o-", label="Cross-validation score")
plt.fill_between(train_sizes, train_scores.mean(axis=1) - train_scores.std(axis=1),
                  train_scores.mean(axis=1) + train_scores.std(axis=1), alpha=0.15)
plt.fill_between(train_sizes, val_scores.mean(axis=1) - val_scores.std(axis=1),
                  val_scores.mean(axis=1) + val_scores.std(axis=1), alpha=0.15)
plt.xlabel("Training Set Size")
plt.ylabel("F1 Score")
plt.title(f"Learning Curve - {best_overall}")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("learning_curve.png", dpi=150)
plt.close()

print("\nSaved plots: default_vs_tuned_comparison.png, best_model_confusion_matrix.png, learning_curve.png")
print(f"\nDone. Phase 1 (Classic ML) capstone complete - {best_overall} selected as best performer.")
