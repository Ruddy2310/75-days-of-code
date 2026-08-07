"""
Day 2 - Data Visualization Toolkit
A set of reusable plotting functions for quick data exploration.
Each function takes a DataFrame + column name(s) and saves a chart.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")


def bar_chart(df: pd.DataFrame, x: str, y: str, output_path: str = "bar_chart.png") -> None:
    """Bar chart comparing y across categories in x."""
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df, x=x, y=y, hue=x, legend=False, palette="viridis")
    plt.title(f"{y} by {x}")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved {output_path}")


def line_chart(df: pd.DataFrame, x: str, y: str, output_path: str = "line_chart.png") -> None:
    """Line chart showing trend of y over x."""
    plt.figure(figsize=(8, 5))
    sorted_df = df.sort_values(by=x)
    sns.lineplot(data=sorted_df, x=x, y=y, marker="o")
    plt.title(f"{y} over {x}")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved {output_path}")


def scatter_plot(df: pd.DataFrame, x: str, y: str, hue: str = None, output_path: str = "scatter_plot.png") -> None:
    """Scatter plot to explore relationship between two numeric columns."""
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x=x, y=y, hue=hue, palette="deep")
    plt.title(f"{y} vs {x}")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved {output_path}")


def box_plot(df: pd.DataFrame, x: str, y: str, output_path: str = "box_plot.png") -> None:
    """Box plot to compare distribution of y across categories in x."""
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x=x, y=y, hue=x, legend=False, palette="pastel")
    plt.title(f"Distribution of {y} by {x}")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved {output_path}")


if __name__ == "__main__":
    # Demo using seaborn's built-in "tips" dataset
    df = sns.load_dataset("tips")

    bar_chart(df, x="day", y="total_bill")
    line_chart(df, x="size", y="tip")
    scatter_plot(df, x="total_bill", y="tip", hue="time")
    box_plot(df, x="day", y="total_bill")
