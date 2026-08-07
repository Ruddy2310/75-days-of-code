"""
Day 13 - 75 Days of Code Challenge
Classic ML Mini-Project #8: Regularized Regression vs SVR
Author: Rudra (Ruddy2310)

Goal: Compare Ridge Regression, Lasso Regression, and Support Vector
Regression (SVR) on the same regression problem, to see how
regularization and kernel-based methods differ in performance and
in how they handle feature importance.

Dataset: sklearn's built-in diabetes dataset re-used here specifically
to compare against Day 7's plain Linear Regression / Random Forest
baseline, now with regularized and kernel methods. No download required.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

RANDOM_STATE = 42

# -----------------------------
# 1. Load & explore the data
# -----------------------------
data = load_diabetes()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="disease_progression")

print("Dataset shape:", X.shape)
print("Target range:", y.min(), "-", y.max())
print("Missing values:", X.isnull().sum().sum())

# -----------------------------
# 2. Train/test split + scaling
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# 3. Train models
# -----------------------------
models = {
    "Ridge Regression": Ridge(alpha=1.0, random_state=RANDOM_STATE),
    "Lasso Regression": Lasso(alpha=0.5, random_state=RANDOM_STATE),
    "SVR (RBF kernel)": SVR(kernel="rbf", C=100, gamma="scale"),
}

results = {}
predictions = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    pred = model.predict(X_test_scaled)
    predictions[name] = pred

    rmse = np.sqrt(mean_squared_error(y_test, pred))
    mae = mean_absolute_error(y_test, pred)
    r2 = r2_score(y_test, pred)

    X_all_scaled = scaler.fit_transform(X)
    cv_scores = cross_val_score(model, X_all_scaled, y, cv=5, scoring="r2")

    results[name] = {
        "RMSE": rmse, "MAE": mae, "R2": r2,
        "CV R2 Mean": cv_scores.mean(), "CV R2 Std": cv_scores.std()
    }

    print(f"\n--- {name} ---")
    print(f"RMSE     : {rmse:.2f}")
    print(f"MAE      : {mae:.2f}")
    print(f"R2 Score : {r2:.4f}")
    print(f"5-fold CV R2: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

results_df = pd.DataFrame(results).T
print("\n=== Summary Comparison ===")
print(results_df.round(4))

# -----------------------------
# 4. Visualizations
# -----------------------------
# Predicted vs Actual for all 3 models
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, (name, pred) in zip(axes, predictions.items()):
    ax.scatter(y_test, pred, alpha=0.6, edgecolor="k")
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(f"{name}\nR2 = {results[name]['R2']:.3f}")
plt.tight_layout()
plt.savefig("predicted_vs_actual.png", dpi=150)
plt.close()

# Ridge vs Lasso coefficients (feature importance / shrinkage comparison)
ridge_coefs = pd.Series(models["Ridge Regression"].coef_, index=X.columns)
lasso_coefs = pd.Series(models["Lasso Regression"].coef_, index=X.columns)
coef_df = pd.DataFrame({"Ridge": ridge_coefs, "Lasso": lasso_coefs})

coef_df.plot(kind="bar", figsize=(10, 5), colormap="viridis")
plt.title("Ridge vs Lasso Coefficients (feature shrinkage)")
plt.ylabel("Coefficient value")
plt.axhline(0, color="black", linewidth=0.8)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("ridge_vs_lasso_coefficients.png", dpi=150)
plt.close()

# Performance comparison bar chart
results_df[["R2", "CV R2 Mean"]].plot(kind="bar", figsize=(8, 5), colormap="viridis")
plt.title("Model Performance Comparison (R2 Score)")
plt.ylabel("R2 Score")
plt.xticks(rotation=15)
plt.legend(["Test R2", "5-fold CV R2"])
plt.tight_layout()
plt.savefig("performance_comparison.png", dpi=150)
plt.close()

print("\nSaved plots: predicted_vs_actual.png, ridge_vs_lasso_coefficients.png, performance_comparison.png")
print("\nDone. Ridge vs Lasso vs SVR comparison complete.")
print(f"\nNote: Lasso zeroed out {(lasso_coefs == 0).sum()} of {len(lasso_coefs)} features (automatic feature selection).")
