# Forest Fires Analysis

Predicting and classifying forest fire risk using the UCI Forest Fires dataset (Cortez & Morais, 2007), sourced from Montesinho Natural Park, Portugal.

## Dataset

- **517 observations**, 12 predictors, 1 target (`area`, hectares burned)
- Predictors include spatial coordinates, month/day, and the Canadian Forest Fire Weather Index components (FFMC, DMC, DC, ISI) plus weather variables (temp, RH, wind, rain)
- 47.78% of observations have `area` = 0; the rest are right-skewed (max 1090.84 ha)
- Source: [UCI ML Repository, ID 162](https://archive.ics.uci.edu/dataset/162/forest+fires)

## Approach

1. **Exploratory analysis**: examined structure, summary statistics and distributions; `area` is zero-inflated and right-skewed, and all predictors have weak correlations with it (|r| ≤ 0.10).
2. **Regression**: fit four OLS variants on log(1+area) (quadratic terms, temp×RH interaction, log transforms). Training adjusted R² was 0.019–0.030, and test R² was negative for all four (-0.027 to -0.054).
3. **Diagnostics**: residuals were right-skewed with a floor from the zero-area cases, and the Q-Q plot showed a heavy right tail. Cook's distance flagged 23 observations; refitting without them changed some coefficients (e.g. temp -0.240 to -0.154) and moved test R² from -0.027 to -0.0005, but the model still has no real predictive power.
4. **Regularization**: Ridge and Lasso gave test R² of 0.0065 and 0.0116 (MSE 2.184 and 2.172), slightly better than OLS but close to predicting the mean. Lasso kept 5 of 34 features (month_dec, DMC, month_may, month_sep, X); two of these are rare months (9 and 2 observations).
5. **Classification**: the target was area > 0.5 ha (260 vs 257 cases), not whether a fire occurred. Logistic regression with balanced class weights reached 58.7% accuracy, precision 0.58, recall 0.62 and F1 0.60 on 104 test rows.
6. **Multicollinearity check**: VIF exceeded 10 for month_sep (44.8), month_aug (36.4) and DC (25.4).

## Key Findings

- Linear regression has no useful predictive power for burned area (negative test R² for every OLS variant).
- The logistic classifier is only modestly better than chance (58.7% vs a 50% baseline) for fires above 0.5 ha.
- No single weather or fire-index variable carries strong signal; most of the data comes from August and September (356 of 517 rows), and several months have very few observations.
- The dataset has no topography, vegetation or urban-proximity variables, so those factors could not be assessed.

## Recommendation

Do not use the regression models to forecast burned area. The logistic classifier can serve as a screening aid for fires above 0.5 ha, with a probability threshold tuned to the cost of false alarms versus missed fires (at the default threshold it missed 20 of 52 such fires). A promising next step is a two-stage model: classify first, then apply a Gamma GLM to estimate area only for positive cases. Terrain, vegetation and settlement data, plus more years of records, would likely improve both stages.

## Files

- `C06M08Lab.ipynb`: full analysis notebook (EDA, regression, classification, diagnostics)

## Tools

Python, pandas, numpy, matplotlib, seaborn, statsmodels, scikit-learn

