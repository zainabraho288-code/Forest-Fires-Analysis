# Forest-Fires-Analysis
# Forest Fires Analysis

Predicting and classifying forest fire risk using the UCI Forest Fires dataset (Cortez & Morais, 2007), sourced from Montesinho Natural Park, Portugal.

## Dataset

- **517 observations**, 12 predictors, 1 target (`area`, hectares burned)
- Predictors include spatial coordinates, month/day, and the Canadian Forest Fire Weather Index components (FFMC, DMC, DC, ISI) plus weather variables (temp, RH, wind, rain)
- Source: [UCI ML Repository, ID 162](https://archive.ics.uci.edu/dataset/162/forest+fires)

## Approach

1. **Exploratory analysis** — examined structure, summary statistics, and distributions; identified that `area` is heavily zero-inflated and right-skewed
2. **Regression** — fit four OLS variants (quadratic terms, interactions, log transformations); best model (log-transformed DC and rain) reached adjusted R² of 0.125, test R² of 0.18
3. **Diagnostics** — residual plots showed heteroscedasticity and non-normality; Cook's distance flagged influential points, but removal didn't improve fit
4. **Regularization** — Ridge and Lasso produced similar test R² (~0.17); Lasso reduced most coefficients to near zero, leaving temperature, DMC, and ISI as the main positive drivers
5. **Classification** — reframed as fire occurrence prediction using logistic regression with balanced class weights; achieved 80% accuracy but low precision (0.29) and recall (0.36), F1-score of 0.32
6. **Multicollinearity check** — VIF values under 10 across all predictors

## Key Findings

- Linear regression has poor explanatory power for burned area given how zero-inflated and skewed the target is
- Classification (fire vs. no fire) is a more tractable framing than direct area regression
- Weather-index variables (temperature, DMC, ISI) carry the most predictive signal among available features

## Recommendation

Rather than forecasting burned area directly, use the logistic classifier with a probability threshold tuned to the operational cost of false alarms vs. missed fires. A promising next step is a two-stage model: classify fire occurrence first, then apply a Gamma GLM to estimate area only for positive cases. Incorporating higher-resolution topographic and fuel data would likely improve both stages.

## Files

- `C06M08Lab.ipynb` — full analysis notebook (EDA, regression, classification, diagnostics)

## Tools

Python, pandas, numpy, matplotlib, seaborn, statsmodels, scikit-learn

