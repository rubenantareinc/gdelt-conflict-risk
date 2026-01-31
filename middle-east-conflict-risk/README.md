# Middle East Conflict Risk (cross-sectional OLS)

This project is an **exploratory, cross-sectional** analysis of country-level indicators and their association with **future conflict events** in selected Middle East contexts.
It uses a **manual Ordinary Least Squares (OLS)** implementation (closed-form solution + coefficient inference) to keep the modeling transparent.

> * Important: this repo is **not** a time-series / city-month GDELT pipeline. It is included as a separate, related modeling artifact.*

## What's inside

- `data/MiddleEast.csv` — the working dataset used in the analysis  
- `legacy/palantirMiddleEast.py` — the original script (kept unchanged)  
- `src/run_analysis.py` — a reproducible runner that saves outputs to `./outputs` and plots to `./figures_generated`  
- `reports/Stats_Palantir.pdf` — write-up / notes  
- `figures/` — previously generated plots (from your earlier runs)

## Method (high level)

- Predict target: `Future_Conflict_Events`
- Predictors include: current conflict intensity, refugee flows, macroeconomic and labor indicators, military expenditure, and a simple news negativity signal (`Negative_News_Count`).
- OLS is fit via the closed-form solution, with t-tests and p-values computed from residual variance.

## Reproduce

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python src/run_analysis.py
```

Outputs:
- `outputs/ols_manual_results.csv`
- `outputs/ols_manual_results.txt`
- `figures_generated/*.png`

## Limitations

- Cross-sectional design (no time-aware validation)
- Correlation ≠ causation
- `Negative_News_Count` is treated as an input feature; deriving it from raw text is outside the current scope of this repo
