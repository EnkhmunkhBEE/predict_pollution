# Ulaanbaatar Air Pollution Prediction — v2

## What this version does
- Reads the `English` sheet from the Data.mn Excel file.
- Uses PM10, PM2.5 and SO2 monthly observations.
- Builds features from past observations only (forward-fill only; no two-sided interpolation).
- Compares Ridge, Random Forest, XGBoost, Seasonal Naive, 3-year seasonal mean and 2022–2025 same-month mean.
- Uses rolling-origin one-month-ahead backtesting on the last 18 observed months.
- Reports MAE, RMSE and R². Metrics are exported in both mg/m³ and µg/m³.
- Selects the lowest-MAE method separately for each pollutant and produces a 12-month forecast.
- Saves fitted ML models as `.joblib` files.

## Setup

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
```

## Run from this project folder

```bash
python src/train_predict.py
```

Or specify paths:

```bash
python src/train_predict.py --input data/raw/air-pollution-concentration.xlsx --out results --models-dir models
```

## Outputs

- `results/analysis_summary.json` — data summary, model metrics and selected models.
- `results/forecast_12_months.csv` — selected forecast per pollutant, original mg/m³ and converted µg/m³.
- `results/forecast_scenarios.csv` — all model/scenario forecasts.
- `results/forecast_ridge_model.csv` — Ridge-only forecast for comparison.
- `results/backtest_PM10.csv`, `backtest_PM25.csv`, `backtest_SO2.csv` — actual vs one-step predictions for the test window.
- `results/clean_monthly_data.csv` — cleaned monthly source data.
- `models/*.joblib` — fitted Ridge, Random Forest and XGBoost pipelines.

## Important interpretation notes
- The source dataset contains monthly aggregate values in mg/m³. A conversion to µg/m³ is included (multiply by 1000).
- The last 18 observed months are used for rolling-origin validation. For each test month, the model is fitted only on earlier months.
- This is a small monthly dataset. Results should be interpreted cautiously; do not claim that any model is universally best based on one validation window.
- The script does not use weather data, station-level data, or future meteorological forecasts.
- If a pollutant has a missing latest month, the model forecasts from the dataset's overall final month and carries forward the last known value for lag construction. Check the `observed` and `missing` counts in `analysis_summary.json`.
- The selected method is chosen by lowest MAE on the same rolling backtest window. For a more rigorous thesis, keep a separate validation window for model selection and a final untouched test window for final reporting.
