"""
Day 6 - 75 Days of Code | Classic ML Mini-Project #1
Titanic Survival Prediction

Goal: Predict whether a passenger survived the Titanic disaster using
classic supervised ML (Logistic Regression + Random Forest), with a full
mini end-to-end pipeline: load -> EDA -> clean -> encode -> train -> evaluate.

Author: Rudra (Ruddy2310)
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# -----------------------------
# 1. Load Data
# -----------------------------
print("=" * 60)
print("STEP 1: Loading Titanic dataset")
print("=" * 60)

df = sns.load_dataset("titanic")
print(f"Shape: {df.shape}")
print(df.head())

# -----------------------------
# 2. Quick EDA
# -----------------------------
print("\n" + "=" * 60)
print("STEP 2: Exploratory Data Analysis")
print("=" * 60)

print("\nMissing values:\n", df.isnull().sum())
print("\nSurvival rate overall:", round(df["survived"].mean(), 3))
print("\nSurvival rate by sex:\n", df.groupby("sex")["survived"].mean())
print("\nSurvival rate by class:\n", df.groupby("pclass")["survived"].mean())

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

sns.countplot(data=df, x="survived", ax=axes[0])
axes[0].set_title("Survival Count (0 = No, 1 = Yes)")

sns.barplot(data=df, x="sex", y="survived", ax=axes[1])
axes[1].set_title("Survival Rate by Sex")

sns.barplot(data=df, x="pclass", y="survived", ax=axes[2])
axes[2].set_title("Survival Rate by Passenger Class")

plt.tight_layout()
plt.savefig("titanic_eda.png", dpi=120)
print("\nSaved EDA plot -> titanic_eda.png")

# -----------------------------
# 3. Data Cleaning & Feature Engineering
# -----------------------------
print("\n" + "=" * 60)
print("STEP 3: Cleaning & Feature Engineering")
print("=" * 60)

data = df.copy()

# Keep only relevant columns for a clean classic ML pipeline
data = data[["survived", "pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]]

# Handle missing values
data["age"] = data["age"].fillna(data["age"].median())
data["embarked"] = data["embarked"].fillna(data["embarked"].mode()[0])

# Feature engineering: family size
data["family_size"] = data["sibsp"] + data["parch"] + 1

# Encode categoricals
le_sex = LabelEncoder()
data["sex"] = le_sex.fit_transform(data["sex"])  # male=1, female=0

le_embarked = LabelEncoder()
data["embarked"] = le_embarked.fit_transform(data["embarked"])

print("Cleaned data sample:\n", data.head())
print("\nMissing values after cleaning:\n", data.isnull().sum().sum(), "total")

# -----------------------------
# 4. Train / Test Split
# -----------------------------
print("\n" + "=" * 60)
print("STEP 4: Train/Test Split")
print("=" * 60)

X = data.drop("survived", axis=1)
y = data["survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

# -----------------------------
# 5. Train Models
# -----------------------------
print("\n" + "=" * 60)
print("STEP 5: Training Models")
print("=" * 60)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
}

results = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    results[name] = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

    print(f"\n--- {name} ---")
    print(f"Accuracy : {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall   : {rec:.3f}")
    print(f"F1 Score : {f1:.3f}")

# -----------------------------
# 6. Compare & Confusion Matrix (best model)
# -----------------------------
print("\n" + "=" * 60)
print("STEP 6: Model Comparison")
print("=" * 60)

results_df = pd.DataFrame(results).T.sort_values("f1", ascending=False)
print(results_df)

best_model_name = results_df.index[0]
best_model = models[best_model_name]
best_preds = best_model.predict(X_test_scaled)

print(f"\nBest model: {best_model_name}")
print("\nClassification Report:\n", classification_report(y_test, best_preds))

cm = confusion_matrix(y_test, best_preds)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Did not survive", "Survived"],
            yticklabels=["Did not survive", "Survived"])
plt.title(f"Confusion Matrix - {best_model_name}")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("titanic_confusion_matrix.png", dpi=120)
print("\nSaved confusion matrix -> titanic_confusion_matrix.png")

# -----------------------------
# 7. Feature Importance (Random Forest)
# -----------------------------
if "Random Forest" in models:
    rf = models["Random Forest"]
    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\nFeature Importances (Random Forest):\n", importances)

    plt.figure(figsize=(6, 4))
    sns.barplot(x=importances.values, y=importances.index)
    plt.title("Feature Importance - Random Forest")
    plt.tight_layout()
    plt.savefig("titanic_feature_importance.png", dpi=120)
    print("Saved feature importance plot -> titanic_feature_importance.png")

print("\n" + "=" * 60)
print("DONE. Day 6 mini-project complete.")
print("=" * 60)
