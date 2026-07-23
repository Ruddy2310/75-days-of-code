"""
Day 1 - Exploratory Data Analysis (EDA) Mini Script
Loads a dataset, cleans it, and generates basic summary stats + a plot.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(path: str) -> pd.DataFrame:
    """Load CSV data into a DataFrame."""
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicates and handle missing values."""
    df = df.drop_duplicates()
    df = df.fillna(df.median(numeric_only=True))
    return df


def summarize(df: pd.DataFrame) -> None:
    """Print basic dataset summary."""
    print("Shape:", df.shape)
    print("\nColumn info:")
    print(df.info())
    print("\nSummary statistics:")
    print(df.describe())


def plot_correlation(df: pd.DataFrame, output_path: str = "correlation_heatmap.png") -> None:
    """Plot and save a correlation heatmap for numeric columns."""
    numeric_df = df.select_dtypes(include="number")
    plt.figure(figsize=(8, 6))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved heatmap to {output_path}")


if __name__ == "__main__":
    # Example: using seaborn's built-in dataset for demo purposes
    df = sns.load_dataset("tips")
    df = clean_data(df)
    summarize(df)
    plot_correlation(df)
