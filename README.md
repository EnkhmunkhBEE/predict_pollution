# Ulaanbaatar Air Pollution Prediction — v2

This project forecasts monthly PM10, PM2.5 and SO2 concentrations from the Data.mn workbook. The maintained entry point is `src/train_predict.py`; it is the single pipeline that creates the metrics, forecasts, intervals, audit files and figures.

## What the pipeline does

- Reads the `English` worksheet from Excel, or a CSV with `month`, `PM10`, `PM2.5` and `SO2` columns.
- Keeps nonnegative values and forward-fills past values only when constructing lag features.
- Compares raw Ridge, log-target Ridge, Random Forest, XGBoost (when installed), seasonal naive and three-year seasonal mean models.
- Uses up to 60 rolling-origin validation points by default. The earliest third selects the model; later origins calibrate prediction intervals after a 12-month target-window embargo, so selection and calibration target months do not overlap. Set a different limit with `--max-validation-origins`.
- Keeps the latest 12 calendar months for each pollutant out of tuning, model selection and interval calibration. It reports one-step and fixed-origin recursive holdout performance separately.
- Reports performance and model-selection stability independently for PM10, PM2.5 and SO2. Stability includes per-origin MAE, mean and standard deviation of MAE, winner frequency and average rank.
- Calibrates empirical intervals separately by forecast horizon. If forecasting must bridge elapsed months between the last observation and the next future calendar month, the interval uses calibration errors at the corresponding extended horizon so bridge forecast error is included.
- Builds CSV, JSON and PNG outputs from the same in-memory run.

## Setup

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

From the project folder:

```bash
python src/train_predict.py --input-unit mg/m3
```

The input unit can also be set to `ug/m3`. The default `auto` mode is only a scale heuristic; confirm the workbook's unit metadata where possible. For example:

```bash
python src/train_predict.py --input data/raw/air-pollution-concentration.xlsx --out results --models-dir models --input-unit mg/m3 --max-validation-origins 60
```

## Main outputs

- `results/analysis_summary.json` — data audit, methodology, selected methods, validation/calibration split, interval diagnostics, holdout scores and model-stability summaries for each pollutant.
- `results/model_metrics.csv` — validation, calibration and holdout metrics.
- `results/model_stability.csv` — per-origin scores and ranks for each method and pollutant.
- `results/forecast_12_months.csv` — selected 12-month forecast by pollutant.
- `results/forecast_scenarios.csv` — forecasts from every candidate method.
- `results/forecast_intervals.csv` — empirical interval, calibration horizon and sample diagnostics for each forecast month. Missing intervals are left blank when that horizon has no calibration residuals.
- `results/recursive_validation_PM10.csv`, `recursive_validation_PM25.csv`, `recursive_validation_SO2.csv` — model-selection predictions by origin and horizon.
- `results/interval_calibration_PM10.csv`, `interval_calibration_PM25.csv`, `interval_calibration_SO2.csv` — separate calibration predictions and absolute errors.
- `results/recursive_holdout_*.csv`, `one_step_holdout_*.csv` — final holdout predictions.
- `results/clean_monthly_data.csv` — cleaned, monthly source series in µg/m³.
- `results/forecast_PM10.png`, `forecast_PM25.png`, `forecast_SO2.png`, `model_comparison.png`, `horizon_error_comparison.png` — figures generated in the same run.
- `models/*.joblib` — final fitted ML pipelines.

## Interpretation

The interval columns are empirical calibration intervals, not guaranteed future 95% coverage. Rolling-origin residuals may be dependent, and sparse horizons are explicitly flagged in the JSON and CSV. The final holdout is for reporting only and does not affect model choice or intervals. These series are short monthly aggregates; stability results should be read as evidence across the evaluated origins, not a guarantee of future performance. The models do not use weather or station-level predictors.


Legacy outputs from the previous pipeline are preserved under `results/archive_pre_refresh/` and are not part of the current run's output manifest.
