# Day 2 — Data Visualization Toolkit

Part of my **#75DaysOfGitHub** challenge.

## What it does
A set of 4 reusable plotting functions built on matplotlib + seaborn:
- `bar_chart()` — compare a value across categories
- `line_chart()` — show trends over an ordered variable
- `scatter_plot()` — explore relationships between two numeric variables (with optional color grouping)
- `box_plot()` — compare distributions/spread across categories

Each function saves its chart as a PNG, so it can be dropped into any future project without rewriting plotting code from scratch.

## Why
Good visualization is how you actually *understand* data (and communicate findings to others). Instead of writing one-off plotting code every time, this toolkit is a reusable base I can import into future projects.

## How to run
```bash
pip install pandas matplotlib seaborn
python data_viz_toolkit.py
```

## Output
4 PNG files: `bar_chart.png`, `line_chart.png`, `scatter_plot.png`, `box_plot.png`

## Tech
Python, pandas, matplotlib, seaborn

---
Day 2 of 75 🚀
