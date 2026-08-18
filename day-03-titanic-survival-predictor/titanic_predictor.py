"""
Day 3 - 75 Days of Code Challenge
Titanic Survival Predictor
Author: Ruddy2310

Goal: Predict passenger survival using Logistic Regression & Random Forest.
Covers: data cleaning, feature engineering, model training, evaluation.
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ---------------------------------------------------------
# 1. Load Data
# ---------------------------------------------------------
print("Loading Titanic dataset...")
df = sns.load_dataset("titanic")
print(f"Shape: {df.shape}")
print(df.head())

# ---------------------------------------------------------
# 2. Data Cleaning
# ---------------------------------------------------------
df = df[["survived", "pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]].copy()

df["age"] = df["age"].fillna(df["age"].median())
df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])

# ---------------------------------------------------------
# 3. Feature Engineering
# ---------------------------------------------------------
df["family_size"] = df["sibsp"] + df["parch"] + 1
df["is_alone"] = (df["family_size"] == 1).astype(int)

le_sex = LabelEncoder()
le_embarked = LabelEncoder()
df["sex"] = le_sex.fit_transform(df["sex"])          # male=1, female=0
df["embarked"] = le_embarked.fit_transform(df["embarked"])

features = ["pclass", "sex", "age", "fare", "family_size", "is_alone", "embarked"]
X = df[features]
y = df["survived"]

# ---------------------------------------------------------
# 4. Train/Test Split + Scaling
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------
# 5. Train Models
# ---------------------------------------------------------
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_scaled, y_train)
log_preds = log_reg.predict(X_test_scaled)

rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)  # RF doesn't need scaling
rf_preds = rf.predict(X_test)

# ---------------------------------------------------------
# 6. Evaluation
# ---------------------------------------------------------
print("\n===== Logistic Regression =====")
print(f"Accuracy: {accuracy_score(y_test, log_preds):.4f}")
print(classification_report(y_test, log_preds))

print("\n===== Random Forest =====")
print(f"Accuracy: {accuracy_score(y_test, rf_preds):.4f}")
print(classification_report(y_test, rf_preds))

# ---------------------------------------------------------
# 7. Feature Importance (Random Forest)
# ---------------------------------------------------------
importance = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
print("\nFeature Importance (Random Forest):")
print(importance)

plt.figure(figsize=(8, 5))
sns.barplot(x=importance.values, y=importance.index, hue=importance.index, palette="viridis", legend=False)
plt.title("Feature Importance - Random Forest")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
print("\nSaved: feature_importance.png")

# ---------------------------------------------------------
# 8. Confusion Matrix (Random Forest)
# ---------------------------------------------------------
plt.figure(figsize=(5, 4))
cm = confusion_matrix(y_test, rf_preds)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Died", "Survived"], yticklabels=["Died", "Survived"])
plt.title("Confusion Matrix - Random Forest")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
print("Saved: confusion_matrix.png")

print("\nDone. Day 3 complete.")
