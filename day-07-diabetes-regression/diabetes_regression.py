"""
Day 7 - Classic ML Mini-Project #2
Diabetes Progression Prediction (Regression)

Goal: Predict disease progression score using Linear Regression and
Random Forest Regressor, then compare performance.

Note: uses sklearn's built-in diabetes dataset (no download needed,
works fully offline).

Part of: 75-Days-of-Code (Ruddy2310)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ----------------------------------------------------------------------
# 1. Load data
# ----------------------------------------------------------------------
data = load_diabetes(as_frame=True)
df = data.frame
print("Shape:", df.shape)
print(df.head())

# ----------------------------------------------------------------------
# 2. Train / test split
# ----------------------------------------------------------------------
X = df.drop(columns=["target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------------------------------------------------
# 3. Train models
# ----------------------------------------------------------------------
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
lin_preds = lin_reg.predict(X_test)

rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)

# ----------------------------------------------------------------------
# 4. Evaluate
# ----------------------------------------------------------------------
def report(name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"\n=== {name} ===")
    print(f"RMSE: {rmse:.4f}")
    print(f"R^2 : {r2:.4f}")

report("Linear Regression", y_test, lin_preds)
report("Random Forest", y_test, rf_preds)

# ----------------------------------------------------------------------
# 5. Predicted vs Actual plot
# ----------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

axes[0].scatter(y_test, lin_preds, alpha=0.3, s=10)
axes[0].plot([y.min(), y.max()], [y.min(), y.max()], "r--")
axes[0].set_title("Linear Regression")
axes[0].set_xlabel("Actual")
axes[0].set_ylabel("Predicted")

axes[1].scatter(y_test, rf_preds, alpha=0.3, s=10, color="green")
axes[1].plot([y.min(), y.max()], [y.min(), y.max()], "r--")
axes[1].set_title("Random Forest")
axes[1].set_xlabel("Actual")
axes[1].set_ylabel("Predicted")

plt.tight_layout()
plt.savefig("predicted_vs_actual.png", dpi=150)
print("\nSaved predicted_vs_actual.png")

# ----------------------------------------------------------------------
# 6. Feature importance (Random Forest)
# ----------------------------------------------------------------------
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(
    ascending=False
)

plt.figure(figsize=(6, 4))
importances.plot(kind="bar", color="teal")
plt.title("Random Forest - Feature Importance")
plt.ylabel("Importance")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
print("Saved feature_importance.png")

print("\nFeature importance ranking:\n", importances)
