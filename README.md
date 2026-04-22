# Ames Housing — Kaggle House Prices

End-to-end regression project predicting house sale prices in Ames, Iowa.

**Competition:** [House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)

## Project structure

```
ames-housing/
├── data/           # Kaggle data (not committed — download from competition page)
├── notebooks/
│   └── 01-explore.ipynb
└── README.md
```

## Approach

*Coming soon — project in progress.*

## Results

| Model | Validation MAE | Validation R² |
|---|---|---|
| Baseline (numeric-only, mean-impute, Random Forest defaults) | $18,105 | 0.89 |
| Pipeline (numeric median-impute + categorical one-hot) | $17,653 | 0.89 |
| Pipeline (all features) — 5-fold CV | $18,399 ± $1,635 | 0.89 |
| Tuned (RandomizedSearchCV, 30 iters) — 5-fold CV | **$18,004** | 0.86 |

**Best model:** Tuned Random Forest pipeline with `n_estimators=500`, all other defaults.
**Honest benchmark (CV mean):** $18,004 — approximately 10% average error on a target with mean ~$180k.

## Tech stack

- Python, pandas, NumPy
- scikit-learn
- Jupyter