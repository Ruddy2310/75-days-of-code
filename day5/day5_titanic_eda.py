"""
Day 5 - 75 Days of Code
Titanic Dataset EDA: Missing Values, Outliers & Insight-Driven Visualizations
Author: Rudra
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------
df = sns.load_dataset('titanic')  # built-in, no download needed

print("Shape:", df.shape)
print("\n--- Info ---")
print(df.info())
print("\n--- Describe ---")
print(df.describe())

# ---------------------------------------------------------
# 2. MISSING VALUES
# ---------------------------------------------------------
print("\n--- Missing Values ---")
missing = df.isnull().sum()
print(missing[missing > 0])

# Handle missing values
df['age'] = df['age'].fillna(df['age'].median())          # numeric -> median (robust to outliers)
df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])  # categorical -> mode
df.drop(columns=['deck'], inplace=True)                   # too many missing, drop column

print("\nMissing values after cleaning:")
print(df.isnull().sum().sum())

# ---------------------------------------------------------
# 3. OUTLIER DETECTION (IQR method on 'fare' and 'age')
# ---------------------------------------------------------
def detect_outliers_iqr(data, col):
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = data[(data[col] < lower) | (data[col] > upper)]
    print(f"{col}: {len(outliers)} outliers detected (bounds: {lower:.2f} to {upper:.2f})")
    return outliers

detect_outliers_iqr(df, 'fare')
detect_outliers_iqr(df, 'age')

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.boxplot(y=df['fare'], ax=axes[0], color='skyblue')
axes[0].set_title('Fare - Outlier Check')
sns.boxplot(y=df['age'], ax=axes[1], color='salmon')
axes[1].set_title('Age - Outlier Check')
plt.tight_layout()
plt.savefig('outliers_boxplot.png', dpi=120)
plt.show()

# ---------------------------------------------------------
# 4. CORRELATION HEATMAP
# ---------------------------------------------------------
plt.figure(figsize=(8, 6))
numeric_df = df.select_dtypes(include=[np.number])
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap - Titanic Numeric Features')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=120)
plt.show()

# ---------------------------------------------------------
# 5. INSIGHT-DRIVEN PLOTS
# ---------------------------------------------------------

# Insight 1: Survival rate by passenger class and gender
plt.figure(figsize=(8, 5))
sns.barplot(data=df, x='class', y='survived', hue='sex')
plt.title('Survival Rate by Class and Gender')
plt.ylabel('Survival Rate')
plt.tight_layout()
plt.savefig('survival_by_class_gender.png', dpi=120)
plt.show()

# Insight 2: Age distribution of survivors vs non-survivors
plt.figure(figsize=(8, 5))
sns.kdeplot(data=df[df['survived'] == 1], x='age', label='Survived', fill=True)
sns.kdeplot(data=df[df['survived'] == 0], x='age', label='Did Not Survive', fill=True)
plt.title('Age Distribution: Survived vs Not Survived')
plt.legend()
plt.tight_layout()
plt.savefig('age_distribution_survival.png', dpi=120)
plt.show()

# Insight 3: Fare vs survival
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x='survived', y='fare')
plt.title('Fare Distribution by Survival')
plt.xlabel('Survived (0 = No, 1 = Yes)')
plt.tight_layout()
plt.savefig('fare_by_survival.png', dpi=120)
plt.show()

# ---------------------------------------------------------
# 6. KEY FINDINGS (for README/summary)
# ---------------------------------------------------------
findings = """
KEY FINDINGS - Titanic EDA (Day 5)
------------------------------------
1. Women had a significantly higher survival rate than men across all classes,
   especially in 1st and 2nd class (~90%+ survival for women in 1st class).
2. Younger passengers (children) had a noticeably higher survival rate,
   consistent with 'women and children first' evacuation priority.
3. Passengers who paid higher fares (likely 1st class) had better survival odds -
   fare and survival show a positive correlation.
4. 'Age' and 'fare' both contained outliers, handled via IQR detection.
   Missing 'age' values were imputed with median, 'embarked' with mode,
   and 'deck' was dropped due to excessive missingness.
"""
print(findings)

with open('findings.md', 'w') as f:
    f.write(findings)

print("\nDone! Plots saved as PNG, findings saved to findings.md")
