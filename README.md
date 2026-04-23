# Ames Housing — Kaggle House Prices

End-to-end regression project predicting house sale prices in Ames, Iowa.

**Competition:** [House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)

## Project structure

```
ames-housing/
├── data/                 # Kaggle data (not committed — download from competition page)
├── notebooks/
│   ├── 01-explore.ipynb       # EDA: distributions, missing values, correlations
│   ├── 02-baseline.ipynb      # Baseline Random Forest (numeric features only)
│   ├── 03-pipeline.ipynb      # Pipeline with imputation + one-hot encoding
│   └── 04-tuning.ipynb        # RandomizedSearchCV + Kaggle submission
├── submission.csv        # Final Kaggle submission (RMSLE 0.15992)
└── README.md
```

## Approach

The project followed a six-stage workflow:

1. **Exploration** — Inspected data shape, target distribution (right-skewed), missing values (19 columns with NaNs; `PoolQC`, `MiscFeature`, `Alley` the worst offenders), and numeric-feature correlations with `SalePrice` (`OverallQual`, `GrLivArea`, `GarageCars` top the list).

2. **Baseline model** — Random Forest with numeric features only, mean-imputed. Established a floor to beat: MAE ~$18,400.

3. **Preprocessing pipeline** — Built a `ColumnTransformer` wrapping `SimpleImputer` (median for numeric, constant `"missing"` for categorical) and `OneHotEncoder(handle_unknown="ignore")`. Wrapped the whole thing in a `Pipeline` with the regressor so preprocessing and modelling fit in one call — no data leakage.

4. **Honest evaluation** — Replaced single train/test split with 5-fold cross-validation. Revealed that the single-split score was optimistic; true CV mean MAE was $18,399 ± $1,635.

5. **Hyperparameter tuning** — `RandomizedSearchCV` with 30 iterations across `n_estimators`, `max_depth`, `max_features`, `min_samples_split`, `min_samples_leaf`. Best CV MAE: $18,004.

6. **Kaggle submission** — Applied the tuned pipeline to `test.csv` and submitted. Public RMSLE: 0.15992.

## Results

| Model | Validation MAE | Validation R² |
|---|---|---|
| Baseline (numeric-only, mean-impute, Random Forest defaults) | $18,105 | 0.89 |
| Pipeline (numeric median-impute + categorical one-hot) | $17,653 | 0.89 |
| Pipeline (all features) — 5-fold CV | $18,399 ± $1,635 | 0.89 |
| Tuned (RandomizedSearchCV, 30 iters) — 5-fold CV | **$18,004** | 0.86 |

**Best model:** Tuned Random Forest pipeline with `n_estimators=500`, all other defaults.
**Honest benchmark (CV mean):** $18,004 — approximately 10% average error on a target with mean ~$180k.


## Kaggle Submission

- **Public Score (RMSLE):** 0.15992
- **Model:** Random Forest (tuned) inside a preprocessing Pipeline
- **Submission:** [View on Kaggle](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/submissions)

## Key Learnings

- **Cross-validation > single-split validation.** A single train/test split gave optimistic MAE ($17,653); 5-fold CV revealed the honest number ($18,399).
- **Default hyperparameters are strong.** Random Forest's defaults were nearly optimal — tuning via RandomizedSearchCV only improved CV MAE by ~$395.
- **Pipelines prevent data leakage.** Fitting the imputer and encoder on training folds only (automatic inside Pipeline) is crucial for honest evaluation.
- **Categorical features mattered less than expected.** On this dataset with Random Forest, numeric features carried most of the signal.

## Next Steps

- [ ] Feature engineering (total square footage, house age, quality interactions)
- [ ] Try gradient boosting (`HistGradientBoostingRegressor`, XGBoost)
- [ ] Log-transform the target to address right-skew
- [ ] Stacked ensemble combining RF + GB


## Tech stack

- Python, pandas, NumPy
- scikit-learn
- Jupyter