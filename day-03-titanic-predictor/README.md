# Day 3 - Titanic Survival Predictor 🚢

Part of my [#75DaysOfCode](https://github.com/Ruddy2310/75-days-of-code) challenge.

## What it does
Predicts passenger survival on the Titanic using two ML models, with full
data cleaning + feature engineering pipeline.

## Pipeline
- **Data cleaning**: handled missing `age` (median) and `embarked` (mode)
- **Feature engineering**: `family_size`, `is_alone`, label-encoded categoricals
- **Models**: Logistic Regression (scaled) & Random Forest (200 trees)
- **Evaluation**: accuracy, precision/recall, confusion matrix, feature importance

## Results
| Model | Accuracy |
|---|---|
| Logistic Regression | ~80% |
| Random Forest | ~81% |

Top predictive features: `fare`, `sex`, `age`

## Files
- `titanic_predictor.py` — full pipeline
- `feature_importance.png` — RF feature importance chart
- `confusion_matrix.png` — RF confusion matrix

## Run it
```bash
pip install pandas scikit-learn seaborn matplotlib
python titanic_predictor.py
```

## Tech stack
`pandas` `scikit-learn` `seaborn` `matplotlib`
