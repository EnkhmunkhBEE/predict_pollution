"""Leakage-aware monthly air-pollution modelling, rolling backtest and 12-month forecast.

Input: monthly Data.mn Excel/CSV with columns month, PM10, PM2.5, SO2.
Source concentrations are kept in their original mg/m3 units. Additional *_ug_m3
columns/files are exported for easier interpretation (1 mg/m3 = 1000 ug/m3).
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler

TARGETS = ["PM10", "PM2.5", "SO2"]
LAGS = (1, 2, 3, 6, 12)
HORIZON = 12
TEST_OBSERVATIONS = 18
RANDOM_STATE = 42


def read_source(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
    else:
        frame = pd.read_excel(path, sheet_name="English")
    if "month" not in frame.columns:
        raise ValueError("Input must contain a 'month' column (English sheet).")
    frame["month"] = pd.to_datetime(frame["month"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    frame = frame.dropna(subset=["month"]).sort_values("month")
    frame = frame.drop_duplicates("month", keep="last").set_index("month").asfreq("MS")
    for target in TARGETS:
        if target not in frame.columns:
            frame[target] = np.nan
        frame[target] = pd.to_numeric(frame[target], errors="coerce")
        # Negative concentrations are invalid; do not silently turn them into observations.
        frame.loc[frame[target] < 0, target] = np.nan
    return frame[TARGETS]


def make_feature_row(history: pd.Series, target_date: pd.Timestamp) -> dict[str, float]:
    """Build a forecast feature vector using only values available before target_date."""
    hist = history.loc[history.index < target_date].asfreq("MS")
    # Forward-fill only: no backward interpolation and no future information.
    past = hist.ffill()
    if past.notna().sum() == 0:
        raise ValueError(f"No historical observations available before {target_date.date()}")
    row: dict[str, float] = {
        "trend_years": float((target_date - history.index.min()).days / 365.25),
        "month_sin": float(np.sin(2 * np.pi * target_date.month / 12)),
        "month_cos": float(np.cos(2 * np.pi * target_date.month / 12)),
    }
    for lag in LAGS:
        lag_date = target_date - pd.DateOffset(months=lag)
        candidates = past.loc[past.index <= lag_date].dropna()
        row[f"lag_{lag}"] = float(candidates.iloc[-1]) if len(candidates) else np.nan
    # Trailing averages are also strictly historical.
    for window in (3, 6, 12):
        trailing = past.loc[past.index < target_date].dropna().tail(window)
        row[f"mean_{window}"] = float(trailing.mean()) if len(trailing) else np.nan
    return row


def make_supervised(series: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """Vectorized, leakage-safe supervised rows; every feature is shifted into the past."""
    idx = series.index
    X = pd.DataFrame(index=idx)
    X["trend_years"] = (idx - idx.min()).days / 365.25
    X["month_sin"] = np.sin(2 * np.pi * idx.month / 12)
    X["month_cos"] = np.cos(2 * np.pi * idx.month / 12)
    for lag in LAGS:
        X[f"lag_{lag}"] = series.shift(lag).ffill()
    past_only = series.shift(1).ffill()
    for window in (3, 6, 12):
        X[f"mean_{window}"] = past_only.rolling(window=window, min_periods=window).mean()
    y = series.copy()
    valid = y.notna() & X.notna().all(axis=1)
    return X.loc[valid].astype(float), y.loc[valid].astype(float)


def build_models() -> dict[str, object]:
    models: dict[str, object] = {
        "ridge": TransformedTargetRegressor(
            regressor=Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("model", Ridge(alpha=10.0)),
            ]),
            func=np.log1p,
            inverse_func=np.expm1,
        ),
        "random_forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(
                n_estimators=120, max_depth=5, min_samples_leaf=2,
                max_features=0.9, random_state=RANDOM_STATE, n_jobs=1,
            )),
        ]),
    }
    try:
        from xgboost import XGBRegressor
        models["xgboost"] = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", XGBRegressor(
                n_estimators=120, max_depth=2, learning_rate=0.03,
                subsample=0.85, colsample_bytree=0.9, reg_lambda=5.0,
                objective="reg:squarederror", random_state=RANDOM_STATE,
                n_jobs=1, tree_method="hist", verbosity=0,
            )),
        ])
    except ImportError:
        warnings.warn("xgboost is not installed; XGBoost will be skipped. Install requirements.txt.")
    return models


def seasonal_naive(history: pd.Series, date: pd.Timestamp) -> float:
    prior_date = date - pd.DateOffset(years=1)
    value = history.get(prior_date, np.nan)
    if pd.notna(value):
        return float(value)
    same_month = history.loc[(history.index < date) & (history.index.month == date.month)].dropna()
    if len(same_month):
        return float(same_month.iloc[-1])
    return float(history.dropna().iloc[-1])


def three_year_mean(history: pd.Series, date: pd.Timestamp) -> float:
    recent = history.loc[
        (history.index < date)
        & (history.index >= date - pd.DateOffset(years=3))
        & (history.index.month == date.month)
    ].dropna()
    return float(recent.mean()) if len(recent) else seasonal_naive(history, date)


def period_mean(history: pd.Series, date: pd.Timestamp) -> float:
    recent = history.loc[
        (history.index < date)
        & (history.index >= pd.Timestamp("2022-01-01"))
        & (history.index <= pd.Timestamp("2025-12-01"))
        & (history.index.month == date.month)
    ].dropna()
    return float(recent.mean()) if len(recent) else three_year_mean(history, date)


def metric_pack(actual: list[float], predicted: list[float]) -> dict[str, float | int | None]:
    a, p = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    if len(a) == 0:
        return {"n": 0, "mae_mg_m3": None, "rmse_mg_m3": None, "r2": None,
                "mae_ug_m3": None, "rmse_ug_m3": None}
    mae = float(mean_absolute_error(a, p))
    rmse = float(np.sqrt(mean_squared_error(a, p)))
    r2 = float(r2_score(a, p)) if len(a) >= 2 else None
    return {"n": int(len(a)), "mae_mg_m3": mae, "rmse_mg_m3": rmse, "r2": r2,
            "mae_ug_m3": mae * 1000, "rmse_ug_m3": rmse * 1000}


def forecast_model(model: object, history: pd.Series, months: pd.DatetimeIndex) -> list[float]:
    working = history.copy()
    preds: list[float] = []
    for date in months:
        row = pd.DataFrame([make_feature_row(working, date)])
        value = max(0.0, float(model.predict(row)[0]))
        preds.append(value)
        working.loc[date] = value
    return preds


def backtest_target(series: pd.Series) -> tuple[dict, dict[str, list[float]], list[pd.Timestamp], list[float]]:
    observed_dates = series.dropna().index
    test_dates = list(observed_dates[-TEST_OBSERVATIONS:])
    if len(observed_dates) < 48:
        raise ValueError(f"Not enough observations for {series.name}: {len(observed_dates)} (need at least 48).")
    methods = ["ridge", "random_forest", "xgboost", "seasonal_naive", "three_year_seasonal_mean", "mean_2022_2025_same_month"]
    predictions = {name: [] for name in methods}
    actual: list[float] = []

    for date in test_dates:
        history = series.loc[series.index < date]
        X_train, y_train = make_supervised(history)
        X_test = pd.DataFrame([make_feature_row(history, date)])
        if len(X_train) < 24:
            raise ValueError(f"Insufficient training rows before {date.date()} for {series.name}.")
        actual.append(float(series.loc[date]))
        models = build_models()
        for name in ("ridge", "random_forest", "xgboost"):
            if name not in models:
                continue
            model = models[name]
            model.fit(X_train, y_train)
            predictions[name].append(max(0.0, float(model.predict(X_test)[0])))
        predictions["seasonal_naive"].append(max(0.0, seasonal_naive(history, date)))
        predictions["three_year_seasonal_mean"].append(max(0.0, three_year_mean(history, date)))
        predictions["mean_2022_2025_same_month"].append(max(0.0, period_mean(history, date)))

    predictions = {k: v for k, v in predictions.items() if len(v) == len(actual)}
    metrics = {name: metric_pack(actual, pred) for name, pred in predictions.items()}
    return metrics, predictions, test_dates, actual


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/raw/air-pollution-concentration.xlsx"))
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    args.models_dir.mkdir(parents=True, exist_ok=True)

    df = read_source(args.input)
    df.to_csv(args.out / "clean_monthly_data.csv", encoding="utf-8-sig", index_label="month")
    summary: dict = {
        "source_file": str(args.input), "unit": "mg/m3", "conversion": "ug/m3 = mg/m3 * 1000",
        "rows_monthly": int(len(df)), "start_month": str(df.index.min().date()),
        "end_month": str(df.index.max().date()),
        "missing": {c: int(df[c].isna().sum()) for c in TARGETS},
        "observed": {c: int(df[c].notna().sum()) for c in TARGETS},
        "weather_features_used": False,
        "validation": f"Rolling-origin one-step-ahead backtest on last {TEST_OBSERVATIONS} observed months; training uses only dates before each test month.",
        "feature_policy": "Forward-fill past values only; no backward fill or two-sided interpolation.",
    }

    all_metrics: dict = {}
    selected: dict = {}
    scenario_forecasts: dict[str, dict[str, list[float]]] = {}
    final_month = df.index.max()
    future_months = pd.date_range(final_month + pd.offsets.MonthBegin(1), periods=HORIZON, freq="MS")
    forecast_rows = []
    ridge_rows = []
    scenario_rows = []

    for target in TARGETS:
        series = df[target].astype(float)
        metrics, backtest_preds, test_dates, actual = backtest_target(series)
        all_metrics[target] = {
            "validation_start": str(test_dates[0].date()),
            "validation_end": str(test_dates[-1].date()),
            "actual_unit": "mg/m3",
            "metrics": metrics,
        }

        # Select by lowest MAE in the same rolling-origin backtest.
        best_method = min(metrics, key=lambda name: float(metrics[name]["mae_mg_m3"]))
        selected[target] = best_method

        # Refit ML models on all available supervised observations for future forecasts.
        X_all, y_all = make_supervised(series)
        trained_models = build_models()
        fitted: dict[str, object] = {}
        for name in ("ridge", "random_forest", "xgboost"):
            if name in trained_models:
                trained_models[name].fit(X_all, y_all)
                fitted[name] = trained_models[name]
                joblib.dump(trained_models[name], args.models_dir / f"{target.replace('.', '')}_{name}.joblib")

        forecasts: dict[str, list[float]] = {}
        for name, model in fitted.items():
            forecasts[name] = forecast_model(model, series, future_months)
        forecasts["seasonal_naive"] = [max(0.0, seasonal_naive(series, d)) for d in future_months]
        forecasts["three_year_seasonal_mean"] = [max(0.0, three_year_mean(series, d)) for d in future_months]
        forecasts["mean_2022_2025_same_month"] = [max(0.0, period_mean(series, d)) for d in future_months]
        scenario_forecasts[target] = forecasts

        best_values = forecasts[best_method]
        ridge_values = forecasts.get("ridge", [np.nan] * HORIZON)
        for i, date in enumerate(future_months):
            forecast_rows.append({"month": date, target: best_values[i], f"{target}_ug_m3": best_values[i] * 1000,
                                  f"{target}_selected_method": best_method})
            ridge_rows.append({"month": date, target: ridge_values[i], f"{target}_ug_m3": ridge_values[i] * 1000})
            scenario_row = next((r for r in scenario_rows if r["month"] == date), None)
            if scenario_row is None:
                scenario_row = {"month": date}
                scenario_rows.append(scenario_row)
            for method, values in forecasts.items():
                scenario_row[f"{target}_{method}"] = values[i]
                scenario_row[f"{target}_{method}_ug_m3"] = values[i] * 1000

        # Save one-step backtest predictions for reproducibility and plotting.
        backtest_frame = pd.DataFrame({"actual_mg_m3": actual, "actual_ug_m3": np.asarray(actual) * 1000}, index=test_dates)
        for method, values in backtest_preds.items():
            backtest_frame[f"{method}_mg_m3"] = values
            backtest_frame[f"{method}_ug_m3"] = np.asarray(values) * 1000
        backtest_frame.index.name = "month"
        backtest_frame.to_csv(args.out / f"backtest_{target.replace('.', '')}.csv", encoding="utf-8-sig")

    # Merge target forecast columns into one row per month.
    best_frame = pd.DataFrame({"month": future_months})
    ridge_frame = pd.DataFrame({"month": future_months})
    for target in TARGETS:
        best_method = selected[target]
        vals = scenario_forecasts[target][best_method]
        best_frame[target] = vals
        best_frame[f"{target}_ug_m3"] = np.asarray(vals) * 1000
        best_frame[f"{target}_selected_method"] = best_method
        rvals = scenario_forecasts[target].get("ridge", [np.nan] * HORIZON)
        ridge_frame[target] = rvals
        ridge_frame[f"{target}_ug_m3"] = np.asarray(rvals) * 1000

    scenario_frame = pd.DataFrame(scenario_rows).sort_values("month")
    best_frame.to_csv(args.out / "forecast_12_months.csv", index=False, encoding="utf-8-sig", float_format="%.8f")
    ridge_frame.to_csv(args.out / "forecast_ridge_model.csv", index=False, encoding="utf-8-sig", float_format="%.8f")
    scenario_frame.to_csv(args.out / "forecast_scenarios.csv", index=False, encoding="utf-8-sig", float_format="%.8f")
    scenario_frame.to_csv(args.out / "forecast_scenarios_mg_m3.csv", index=False, encoding="utf-8-sig", float_format="%.8f")

    summary["model_metrics"] = all_metrics
    summary["selected_forecast_methods"] = selected
    summary["forecast_horizon_months"] = HORIZON
    summary["forecast_start"] = str(future_months[0].date())
    summary["forecast_end"] = str(future_months[-1].date())
    summary["forecast_note"] = (
        "Forecasts are model estimates, not observed measurements. Future values are recursive for ML models. "
        "No weather or station-level features are included. Forecast CSV includes original mg/m3 and converted ug/m3. "
        "Selection is based on lowest MAE in the same rolling-origin backtest; this may be optimistic because the same window is used for selection."
    )
    (args.out / "analysis_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({"selected_forecast_methods": selected,
                      "metrics": {t: all_metrics[t]["metrics"] for t in TARGETS},
                      "forecast_file": str(args.out / "forecast_12_months.csv"),
                      "forecast_start": str(future_months[0].date()),
                      "forecast_end": str(future_months[-1].date())}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
