"""Monthly air-pollution forecasting with leakage-safe rolling-origin evaluation.

Expected input columns: month, PM10, PM2.5, SO2. Default Excel sheet: English.
Run from project root:
    python src/train_predict_improved.py --input-unit mg/m3

All model outputs are expressed in micrograms per cubic metre (ug/m3).
Use --input-unit auto only as a cautious scale heuristic; verify the source
workbook's unit metadata and preferably pass mg/m3 or ug/m3 explicitly.
"""
from __future__ import annotations

import argparse
import json
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TARGETS = ["PM10", "PM2.5", "SO2"]
HORIZON = 12
FINAL_HOLDOUT_MONTHS = 12
MIN_HISTORY = 72
MIN_TRAIN_ROWS = 30
LAGS = (1, 2, 3, 6, 12)
RANDOM_STATE = 42
UNIT_FACTOR = 1000.0
BASELINES = ("seasonal_naive", "three_year_seasonal_mean")


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if value is pd.NaT:
        return None
    return value


def read_source(path: Path, input_unit: str) -> tuple[pd.DataFrame, dict[str, str], dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Өгөгдлийн файл олдсонгүй: {path.resolve()}")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        raw = pd.read_csv(path)
    elif suffix in (".xlsx", ".xls"):
        try:
            raw = pd.read_excel(path, sheet_name="English")
        except ValueError as exc:
            raise ValueError("Excel файлд 'English' нэртэй sheet олдсонгүй. Кодын SHEET тохиргоог шалгана уу.") from exc
    else:
        raise ValueError("Оролтын файл .xlsx, .xls эсвэл .csv байх ёстой.")

    raw.columns = [str(c).strip() for c in raw.columns]
    required = ["month", *TARGETS]
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise ValueError(f"Дараах багана дутуу байна: {missing}. Шаардлагатай багана: {required}")

    original_rows = len(raw)
    raw["month"] = pd.to_datetime(raw["month"], errors="coerce")
    invalid_dates = int(raw["month"].isna().sum())
    raw = raw.dropna(subset=["month"]).copy()
    raw["month"] = raw["month"].dt.to_period("M").dt.to_timestamp()
    duplicate_rows = int(raw.duplicated("month", keep="last").sum())
    raw = raw.sort_values("month").drop_duplicates("month", keep="last").set_index("month")
    raw = raw.asfreq("MS")

    clean = pd.DataFrame(index=raw.index)
    units: dict[str, str] = {}
    unit_audit: dict[str, Any] = {}
    for col in TARGETS:
        numeric = pd.to_numeric(raw[col], errors="coerce")
        nonnumeric = int(raw[col].notna().sum() - numeric.notna().sum())
        negative = int((numeric < 0).sum())
        numeric = numeric.mask(numeric < 0)
        positive = numeric[numeric > 0].dropna()
        if input_unit == "auto":
            if positive.empty:
                raise ValueError(f"{col}: эерэг тоон утга олдсонгүй; нэгжийг автоматаар таамаглах боломжгүй.")
            med = float(positive.median())
            if med <= 1.0:
                detected = "mg/m3"
            elif med >= 5.0:
                detected = "ug/m3"
            else:
                raise ValueError(
                    f"{col}: медиан={med:g}. Энэ хэмжээнээс нэгжийг найдвартай ялгах боломжгүй. "
                    "Excel-ийн нэгжийг шалгаад --input-unit mg/m3 эсвэл ug/m3 гэж заана уу."
                )
        else:
            detected = input_unit
        factor = UNIT_FACTOR if detected == "mg/m3" else 1.0
        clean[col] = numeric.astype(float) * factor
        units[col] = detected
        unit_audit[col] = {
            "source_unit": detected, "output_unit": "ug/m3", "conversion_factor": factor,
            "negative_values_set_missing": negative, "nonnumeric_values_set_missing": nonnumeric,
            "missing_values_after_month_reindex": int(clean[col].isna().sum()),
            "positive_median_in_source_unit": float(positive.median()) if len(positive) else None,
            "unit_auto_detection_is_heuristic": input_unit == "auto",
        }
    audit = {
        "source_rows": int(original_rows), "invalid_month_rows_dropped": invalid_dates,
        "duplicate_month_rows_dropped_keep_last": duplicate_rows,
        "monthly_index_start": clean.index.min(), "monthly_index_end": clean.index.max(),
        "monthly_rows_after_reindex": len(clean), "unit_audit": unit_audit,
    }
    return clean.astype(float), units, audit


def feature_row(history: pd.Series, date: pd.Timestamp) -> dict[str, float]:
    """Build a target-month feature row using data strictly before that month."""
    # All callers already provide a monthly-start index. Avoid rebuilding a
    # DatetimeIndex with asfreq inside this per-row feature function.
    hist = history.loc[history.index < date].ffill().dropna()
    first = history.dropna().index.min()
    row: dict[str, float] = {
        "trend_years": float((date - first).days / 365.25),
        "month_sin": float(np.sin(2 * np.pi * date.month / 12)),
        "month_cos": float(np.cos(2 * np.pi * date.month / 12)),
    }
    for lag in LAGS:
        cutoff = date - pd.DateOffset(months=lag)
        eligible = hist.loc[hist.index <= cutoff]
        row[f"lag_{lag}"] = float(eligible.iloc[-1]) if not eligible.empty else np.nan
    for window in (3, 6, 12):
        vals = hist.tail(window)
        row[f"mean_{window}"] = float(vals.mean()) if len(vals) == window else np.nan
    return row


def make_supervised(series: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """Vectorized monthly feature construction; every feature uses past data only."""
    s = series.asfreq("MS").astype(float)
    if s.dropna().empty:
        return pd.DataFrame(), pd.Series(dtype=float)

    first_observed = s.dropna().index.min()
    filled = s.ffill()
    features = pd.DataFrame(index=s.index)
    features["trend_years"] = (s.index - first_observed).days / 365.25
    features["month_sin"] = np.sin(2 * np.pi * s.index.month / 12)
    features["month_cos"] = np.cos(2 * np.pi * s.index.month / 12)

    # shift(k) means the feature for month t only sees values through t-k.
    for lag in LAGS:
        features[f"lag_{lag}"] = filled.shift(lag)
    prior_values = filled.shift(1)
    for window in (3, 6, 12):
        features[f"mean_{window}"] = prior_values.rolling(window=window, min_periods=window).mean()

    valid = s.notna() & np.isfinite(features.to_numpy(dtype=float)).all(axis=1)
    X = features.loc[valid].copy()
    y = s.loc[valid].copy()
    return X, y


class SafeLogTargetRegressor(RegressorMixin, BaseEstimator):
    """Ridge-style regressor with a bounded log1p inverse transformation.

    The upper prediction bound is learned from the training fold only, so the
    cap does not inspect validation or holdout targets. This prevents expm1
    overflow and limits recursive forecast explosion.
    """

    def __init__(self, regressor: Any, max_multiplier: float = 5.0):
        self.regressor = regressor
        self.max_multiplier = max_multiplier

    def fit(self, X: Any, y: Any):
        y_array = np.asarray(y, dtype=float).reshape(-1)
        if not np.isfinite(y_array).all():
            raise ValueError("Log загварын сургалтын target дотор finite бус утга байна.")
        if (y_array < 0).any():
            raise ValueError("log1p загварын target сөрөг утгатай байж болохгүй.")
        if self.max_multiplier < 1:
            raise ValueError("max_multiplier нь 1-ээс багагүй байх ёстой.")
        self.regressor_ = clone(self.regressor)
        self.regressor_.fit(X, np.log1p(y_array))
        self.max_target_ = float(np.max(y_array))
        self.max_log_prediction_ = float(np.log1p(max(self.max_target_ * self.max_multiplier,
                                                       self.max_target_)))
        return self

    def predict(self, X: Any) -> np.ndarray:
        raw_log = np.asarray(self.regressor_.predict(X), dtype=float).reshape(-1)
        # Lower bound enforces non-negative concentration; upper bound is
        # learned from this fit's training data, never from future observations.
        safe_log = np.clip(raw_log, 0.0, self.max_log_prediction_)
        return np.expm1(safe_log)


def build_models() -> dict[str, Any]:
    common = Pipeline([("imputer", SimpleImputer(strategy="median")),
                       ("scale", StandardScaler()), ("model", Ridge())])
    models: dict[str, Any] = {
        "ridge_raw": common,
        "ridge_log": SafeLogTargetRegressor(regressor=clone(common), max_multiplier=5.0),
        "random_forest": Pipeline([("imputer", SimpleImputer(strategy="median")),
                                   ("model", RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=1))]),
    }
    try:
        from xgboost import XGBRegressor
        models["xgboost"] = Pipeline([("imputer", SimpleImputer(strategy="median")),
                                     ("model", XGBRegressor(objective="reg:squarederror", random_state=RANDOM_STATE,
                                                            n_jobs=1, tree_method="hist", verbosity=0))])
    except Exception as exc:
        warnings.warn(f"XGBoost импортлогдсонгүй ({exc!r}); XGBoost загварыг алгасав.")
    return models


def tune_models(history: pd.Series) -> tuple[dict[str, Any], dict[str, Any]]:
    X, y = make_supervised(history)
    if len(X) < MIN_TRAIN_ROWS:
        raise ValueError(f"Загварын tuning-д {len(X)} мөр байна. {MIN_TRAIN_ROWS}-аас дээш шаардлагатай.")
    n_splits = min(3, max(2, len(X) // 20))
    splitter = TimeSeriesSplit(n_splits=n_splits)
    grids: dict[str, dict[str, list[Any]]] = {
        "ridge_raw": {"model__alpha": [0.1, 10.0, 50.0]},
        "ridge_log": {"regressor__model__alpha": [1.0, 10.0]},
        "random_forest": {"model__n_estimators": [30], "model__max_depth": [3],
                          "model__min_samples_leaf": [2]},
        "xgboost": {"model__n_estimators": [30], "model__max_depth": [2],
                    "model__learning_rate": [0.05], "model__min_child_weight": [1],
                    "model__reg_lambda": [5.0]},
    }
    tuned: dict[str, Any] = {}
    info: dict[str, Any] = {}
    for name, estimator in build_models().items():
        search = GridSearchCV(estimator, grids[name], scoring="neg_mean_absolute_error", cv=splitter,
                              n_jobs=1, refit=True, error_score="raise")
        search.fit(X, y)
        tuned[name] = clone(search.best_estimator_)
        info[name] = {"best_params": search.best_params_, "tuning_cv_mae_ug_m3": float(-search.best_score_),
                      "time_series_splits": n_splits, "tuning_rows": len(X)}
    return tuned, info


def seasonal_naive(history: pd.Series, date: pd.Timestamp) -> float:
    prior = history.get(date - pd.DateOffset(years=1), np.nan)
    if pd.notna(prior):
        return float(prior)
    same_month = history.loc[(history.index < date) & (history.index.month == date.month)].dropna()
    return float(same_month.iloc[-1] if len(same_month) else history.dropna().iloc[-1])


def three_year_mean(history: pd.Series, date: pd.Timestamp) -> float:
    recent = history.loc[(history.index < date) &
                         (history.index >= date - pd.DateOffset(years=3)) &
                         (history.index.month == date.month)].dropna()
    return float(recent.mean()) if len(recent) else seasonal_naive(history, date)


def predict_one(model: Any, history: pd.Series, date: pd.Timestamp) -> float:
    X = pd.DataFrame([feature_row(history, date)])
    pred = float(np.asarray(model.predict(X)).reshape(-1)[0])
    if not np.isfinite(pred):
        raise ValueError(f"{date:%Y-%m}: загвар finite бус утга таамаглалаа.")
    return max(0.0, pred)


def recursive_forecast(method: str, template: Any | None, history: pd.Series,
                        dates: pd.DatetimeIndex) -> list[float]:
    """Fit once at the origin, then feed each prediction into the next forecast step."""
    working = history.copy().asfreq("MS")
    fitted = None
    if method not in BASELINES:
        X, y = make_supervised(working)
        if len(X) < MIN_TRAIN_ROWS:
            raise ValueError(f"{method}: recursive origin-д сургалтын мөр хүрэлцэхгүй ({len(X)}).")
        fitted = clone(template).fit(X, y)
    out: list[float] = []
    for date in dates:
        if method == "seasonal_naive":
            value = max(0.0, seasonal_naive(working, date))
        elif method == "three_year_seasonal_mean":
            value = max(0.0, three_year_mean(working, date))
        else:
            assert fitted is not None
            value = predict_one(fitted, working, date)
        out.append(value)
        working.loc[date] = value
    return out


def one_step_rolling(series: pd.Series, dates: list[pd.Timestamp],
                     templates: dict[str, Any]) -> tuple[dict[str, list[float]], list[float], list[pd.Timestamp]]:
    methods = list(templates) + list(BASELINES)
    predictions = {m: [] for m in methods}
    actuals: list[float] = []
    used_dates: list[pd.Timestamp] = []
    for date in dates:
        history = series.loc[series.index < date].asfreq("MS")
        X, y = make_supervised(history)
        if len(X) < MIN_TRAIN_ROWS:
            continue
        actual = series.get(date, np.nan)
        if pd.isna(actual):
            continue
        actuals.append(float(actual))
        used_dates.append(date)
        x_test = pd.DataFrame([feature_row(history, date)])
        for name, template in templates.items():
            model = clone(template).fit(X, y)
            value = float(np.asarray(model.predict(x_test)).reshape(-1)[0])
            predictions[name].append(max(0.0, value))
        predictions["seasonal_naive"].append(max(0.0, seasonal_naive(history, date)))
        predictions["three_year_seasonal_mean"].append(max(0.0, three_year_mean(history, date)))
    return predictions, actuals, used_dates


def metric_pack(actual: list[float] | np.ndarray, predicted: list[float] | np.ndarray) -> dict[str, Any]:
    a, p = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    mask = np.isfinite(a) & np.isfinite(p)
    a, p = a[mask], p[mask]
    if not len(a):
        return {"n": 0, "mae": None, "rmse": None, "r2": None, "mape_pct": None, "smape_pct": None, "wape_pct": None}
    err = p - a
    med = float(np.median(np.abs(a)))
    mape_mask = np.abs(a) > max(1e-9, med * 0.01)
    mape = float(np.mean(np.abs(err[mape_mask] / a[mape_mask])) * 100) if mape_mask.any() else None
    sm_denom = np.abs(a) + np.abs(p)
    sm_mask = sm_denom > 1e-9
    smape = float(np.mean(2 * np.abs(err[sm_mask]) / sm_denom[sm_mask]) * 100) if sm_mask.any() else None
    wape = float(np.sum(np.abs(err)) / np.sum(np.abs(a)) * 100) if np.sum(np.abs(a)) > 1e-9 else None
    return {"n": int(len(a)), "mae": float(mean_absolute_error(a, p)),
            "rmse": float(np.sqrt(mean_squared_error(a, p))),
            "r2": float(r2_score(a, p)) if len(a) >= 2 else None,
            "mape_pct": mape, "mape_excluded_near_zero_n": int((~mape_mask).sum()),
            "smape_pct": smape, "wape_pct": wape}


def choose_validation_origins(dev: pd.Series, horizon: int = HORIZON,
                              max_origins: int = 36) -> list[pd.Timestamp]:
    observed_dates = list(dev.dropna().index)
    candidates: list[pd.Timestamp] = []
    for origin in observed_dates:
        if len(dev.loc[dev.index <= origin].dropna()) < MIN_TRAIN_ROWS + max(LAGS):
            continue
        future = pd.date_range(origin + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")
        actual_count = int(dev.reindex(future).notna().sum())
        if actual_count >= max(8, horizon - 2):
            candidates.append(origin)
    return candidates[-max_origins:]


def recursive_backtest(series: pd.Series, origins: list[pd.Timestamp],
                       templates: dict[str, Any], method_names: list[str] | None = None,
                       horizon: int = HORIZON
                       ) -> dict[str, dict[int, list[tuple[pd.Timestamp, float, float]]]]:
    """Collect (target date, actual, prediction) by method and forecast horizon."""
    methods = method_names if method_names is not None else list(templates) + list(BASELINES)
    results = {m: {h: [] for h in range(1, horizon + 1)} for m in methods}
    for origin in origins:
        history = series.loc[series.index <= origin].asfreq("MS")
        future = pd.date_range(origin + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")
        actuals = series.reindex(future)
        for method in methods:
            preds = recursive_forecast(method, templates.get(method), history, future)
            for h, (date, pred) in enumerate(zip(future, preds), start=1):
                actual = actuals.loc[date]
                if pd.notna(actual):
                    results[method][h].append((date, float(actual), float(pred)))
    return results


def origin_stability_table(results: dict[str, dict[int, list[tuple[pd.Timestamp, float, float]]]]) -> pd.DataFrame:
    """Compute one MAE per method and rolling origin, then rank methods per origin."""
    grouped: dict[str, dict[pd.Timestamp, list[float]]] = defaultdict(lambda: defaultdict(list))
    for method, horizon_map in results.items():
        for horizon, rows in horizon_map.items():
            for target_date, actual, predicted in rows:
                origin = target_date - pd.DateOffset(months=horizon)
                grouped[method][origin].append(abs(float(predicted) - float(actual)))
    records: list[dict[str, Any]] = []
    for method, origin_map in grouped.items():
        for origin, errors in origin_map.items():
            records.append({"method": method, "origin": origin,
                            "n_predictions": len(errors), "origin_mae_ug_m3": float(np.mean(errors))})
    frame = pd.DataFrame(records)
    if not frame.empty:
        frame["origin_rank"] = frame.groupby("origin")["origin_mae_ug_m3"].rank(method="min", ascending=True)
    return frame


def flatten_recursive(results: dict[int, list[tuple[pd.Timestamp, float, float]]]) -> tuple[list[float], list[float]]:
    actuals: list[float] = []
    preds: list[float] = []
    for horizon_rows in results.values():
        for _, actual, pred in horizon_rows:
            actuals.append(actual); preds.append(pred)
    return actuals, preds


def conformal_radius(abs_errors: np.ndarray, alpha: float = 0.05) -> float | None:
    """Finite-sample split-conformal order statistic; requires calibration residuals."""
    errors = np.asarray(abs_errors, dtype=float)
    errors = errors[np.isfinite(errors)]
    if len(errors) == 0:
        return None
    rank = int(np.ceil((len(errors) + 1) * (1 - alpha)))
    rank = min(max(rank, 1), len(errors))
    return float(np.sort(errors)[rank - 1])


def so2_diagnostics(series: pd.Series) -> dict[str, Any]:
    s = series.dropna()
    monthly = s.groupby(s.index.month).agg(["count", "mean", "median", "std"]).reindex(range(1, 13))
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = float(q3 - q1)
    low, high = float(q1 - 1.5 * iqr), float(q3 + 1.5 * iqr)
    outliers = s[(s < low) | (s > high)]
    med = float(s.median())
    mad = float(np.median(np.abs(s - med)))
    robust_z = (0.6745 * (s - med) / mad) if mad > 0 else pd.Series(0.0, index=s.index)
    mad_outliers = s[np.abs(robust_z) > 3.5]
    return {
        "seasonality_by_calendar_month": [
            {"month_number": int(i), "count": int(row["count"]) if pd.notna(row["count"]) else 0,
             "mean_ug_m3": float(row["mean"]) if pd.notna(row["mean"]) else None,
             "median_ug_m3": float(row["median"]) if pd.notna(row["median"]) else None,
             "std_ug_m3": float(row["std"]) if pd.notna(row["std"]) else None}
            for i, row in monthly.iterrows()],
        "iqr_outliers": [{"month": d.strftime("%Y-%m"), "value_ug_m3": float(v)} for d, v in outliers.items()],
        "mad_robust_outliers": [{"month": d.strftime("%Y-%m"), "value_ug_m3": float(v)} for d, v in mad_outliers.items()],
        "outliers_removed": False,
        "note": "Outlier гэж тэмдэглэсэн утгуудыг устгаагүй; эх өгөгдөлтэй нь хадгалж үнэлгээнд оруулсан.",
    }


def save_forecast_plot(target: str, history: pd.Series, future: pd.DatetimeIndex,
                       forecast: np.ndarray, lower: np.ndarray, upper: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 5.5))
    obs = history.dropna()
    ax.plot(obs.index, obs.values, label="Бодит хэмжилт", linewidth=1.5)
    ax.plot(future, forecast, marker="o", label="12 сарын recursive прогноз", linewidth=2)
    ax.fill_between(future, lower, upper, alpha=0.18, label="Horizon-specific empirical interval")
    ax.axvline(future[0], linestyle="--", linewidth=1, label="Прогноз эхлэх")
    ax.set_title(f"{target}: бодит өгөгдөл ба 12 сарын прогноз")
    ax.set_xlabel("Сар"); ax.set_ylabel("Концентраци (µg/m³)"); ax.grid(alpha=0.25); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/raw/air-pollution-concentration.xlsx"))
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--input-unit", choices=("auto", "mg/m3", "ug/m3"), default="auto",
                        help="Эх Excel-ийн нэгж. Судалгааны эх сурвалжаар баталгаажуулж mg/m3 эсвэл ug/m3 гэж өгөхийг зөвлөе.")
    parser.add_argument("--max-origins", type=int, default=36,
                        help="Rolling-origin validation-д ашиглах хамгийн их origin-ийн тоо.")
    parser.add_argument("--selection-origins", type=int, default=6,
                        help="Загвар сонгоход ашиглах эхний origin-ийн тоо.")
    parser.add_argument("--min-calibration-errors", type=int, default=20,
                        help="Horizon бүрийн calibration алдааны зөвлөмжит доод тоо.")
    args = parser.parse_args()
    if args.max_origins < args.selection_origins + 1:
        parser.error("--max-origins нь --selection-origins-оос их байх ёстой.")
    if args.selection_origins < 3:
        parser.error("--selection-origins дор хаяж 3 байна.")
    args.out.mkdir(parents=True, exist_ok=True); args.models_dir.mkdir(parents=True, exist_ok=True)

    df, source_units, audit = read_source(args.input, args.input_unit)
    df.to_csv(args.out / "clean_monthly_data.csv", encoding="utf-8-sig", index_label="month")
    summary: dict[str, Any] = {
        "source_file": str(args.input), "source_units": source_units, "model_and_output_unit": "ug/m3",
        "monthly_data_audit": audit,
        "methodology": {
            "one_step_validation": "Expanding-window rolling-origin: every prediction uses only observations before its target month.",
            "recursive_validation": "At each origin, fit once using data through the origin; recursively forecast the next 12 calendar months.",
            "model_selection": "Select model by mean origin-level MAE across the dedicated earlier selection origins; each origin contributes equally. Final holdout is never used for selection.",
            "tuning": "GridSearchCV + TimeSeriesSplit on the pre-validation period only.",
            "final_holdout": "Latest 12 calendar months are excluded from tuning and model selection; evaluate one-step and fixed-origin 12-step recursive forecasts separately.",
            "prediction_intervals": "Horizon-specific absolute-error quantiles. Model-selection origins and later calibration origins are separated; each horizon uses only its own recursive forecast errors. Calibration count, empirical coverage and interval width are reported per horizon.",
            "interval_caveat": "The configured minimum is a practical diagnostic threshold, not a guarantee. Rolling-origin errors can be serially dependent; calibration coverage is descriptive and does not guarantee future 95% coverage. Sparse samples are flagged and intervals are labeled empirical only.",
            "forecast_date_policy": "Publish 12 months after the later of the last observed month and the current calendar month; recursively bridge any elapsed but unobserved months without presenting them as future forecasts.",
            "SO2": "Inspect month-of-year seasonality and IQR/MAD outliers; compare raw Ridge, log1p Ridge and baselines; do not remove flagged outliers automatically.",
            "units": "All calculations/outputs are ug/m3; 1 mg/m3 = 1000 ug/m3. Auto unit detection is a scale heuristic, not proof of source units.",
            "MAPE": "MAPE excludes near-zero actual values; interpret MAE, RMSE, sMAPE and WAPE alongside it.",
        },
        "weather_features_used": False,
    }

    metric_rows: list[dict[str, Any]] = []
    forecast_rows: list[dict[str, Any]] = []
    interval_rows: list[dict[str, Any]] = []
    scenario_rows: list[dict[str, Any]] = []
    bridge_rows: list[dict[str, Any]] = []
    target_summaries: dict[str, Any] = {}
    all_validation_plot: dict[str, pd.DataFrame] = {}
    all_holdout_plot: dict[str, pd.DataFrame] = {}
    selected_methods: dict[str, str] = {}
    stability_rows: list[dict[str, Any]] = []
    all_series: dict[str, pd.Series] = {}

    for target in TARGETS:
        print(f"[{target}] эхэллээ", flush=True)
        series = df[target].astype(float).asfreq("MS")
        all_series[target] = series
        observed = series.dropna()
        if len(observed) < MIN_HISTORY:
            raise ValueError(f"{target}: {len(observed)} ажиглалт байна. Leakage-safe 12 сарын validation/holdout-д дор хаяж {MIN_HISTORY} сар шаардлагатай.")

        # Determine elapsed bridge months before calibration so their uncertainty
        # is included in the effective calibration horizon.
        next_data_month = observed.index.max() + pd.offsets.MonthBegin(1)
        next_calendar_month = pd.Timestamp.now().to_period("M").to_timestamp() + pd.offsets.MonthBegin(1)
        forecast_start = max(next_data_month, next_calendar_month)
        full_future = pd.date_range(forecast_start, periods=HORIZON, freq="MS")
        bridge_dates = pd.date_range(next_data_month, forecast_start - pd.offsets.MonthBegin(1), freq="MS")
        bridge_month_count = int(len(bridge_dates))
        calibration_horizon = HORIZON + bridge_month_count

        # Hold out the latest 12 calendar months, regardless of occasional missing observations.
        holdout_start = observed.index.max() - pd.DateOffset(months=FINAL_HOLDOUT_MONTHS - 1)
        holdout_start = holdout_start.to_period("M").to_timestamp()
        holdout_dates = pd.date_range(holdout_start, observed.index.max(), freq="MS")
        dev = series.loc[series.index < holdout_start].asfreq("MS")
        origins = choose_validation_origins(dev, horizon=calibration_horizon,
                                            max_origins=args.max_origins)
        selection_count = min(args.selection_origins, max(3, len(origins) - 1))
        if len(origins) < selection_count + 1:
            raise ValueError(
                f"{target}: {len(origins)} боломжит origin байна. "
                f"Загвар сонголт + calibration-д дор хаяж {selection_count + 1} шаардлагатай. "
                "--max-origins-ийг нэмэх эсвэл өгөгдлийн тасалдлыг шалгана уу."
            )
        # Chronological split: earlier origins select the model; later origins
        # are reserved exclusively for interval calibration.
        selection_origins = origins[:selection_count]
        calibration_origins = origins[selection_count:]
        first_origin = selection_origins[0]
        tuning_history = dev.loc[dev.index <= first_origin]
        # Tuning history ends at first origin. The origin itself is used only as the training endpoint.
        templates, tuning_info = tune_models(tuning_history)
        print(f"[{target}] tuning дууслаа; models={len(templates)}", flush=True)
        methods = list(templates) + list(BASELINES)

        # Model selection: multiple earlier rolling origins. The primary score is
        # mean origin-level MAE, so each origin contributes equally despite gaps.
        recursive_val = recursive_backtest(dev, selection_origins, templates, horizon=HORIZON)
        recursive_val_metrics: dict[str, dict[str, Any]] = {}
        for method in methods:
            actuals, preds = flatten_recursive(recursive_val[method])
            recursive_val_metrics[method] = metric_pack(actuals, preds)
            metric_rows.append({"target": target, "method": method, "evaluation": "validation_recursive_all_horizons",
                                "horizon": "1-12", **recursive_val_metrics[method]})
            for h in range(1, HORIZON + 1):
                rows = recursive_val[method][h]
                m = metric_pack([r[1] for r in rows], [r[2] for r in rows])
                metric_rows.append({"target": target, "method": method, "evaluation": "validation_recursive_by_horizon",
                                    "horizon": h, **m})

        origin_table = origin_stability_table(recursive_val)
        if not origin_table.empty:
            origin_table.insert(0, "target", target)
            stability_rows.extend(origin_table.to_dict("records"))
            stability_summary = origin_table.groupby("method").agg(
                origin_count=("origin", "nunique"),
                mean_origin_mae_ug_m3=("origin_mae_ug_m3", "mean"),
                std_origin_mae_ug_m3=("origin_mae_ug_m3", "std"),
                mean_origin_rank=("origin_rank", "mean"),
                best_rank_count=("origin_rank", lambda x: int((x == 1).sum())),
            ).reset_index()
            stability_summary["origin_mae_cv"] = (
                stability_summary["std_origin_mae_ug_m3"] /
                stability_summary["mean_origin_mae_ug_m3"].replace(0, np.nan)
            )
            stability_summary["target"] = target
            for row in stability_summary.to_dict("records"):
                metric_rows.append({"target": target, "method": row["method"],
                                    "evaluation": "validation_origin_stability", "horizon": "1-12",
                                    "n": row["origin_count"], "mae": row["mean_origin_mae_ug_m3"],
                                    "rmse": None, "r2": None, "mape_pct": None,
                                    "smape_pct": None, "wape_pct": None,
                                    "origin_mae_std_ug_m3": row["std_origin_mae_ug_m3"],
                                    "mean_origin_rank": row["mean_origin_rank"],
                                    "best_rank_count": row["best_rank_count"],
                                    "origin_mae_cv": row["origin_mae_cv"]})
            score_map = dict(zip(stability_summary["method"], stability_summary["mean_origin_mae_ug_m3"]))
            stability_by_method = {r["method"]: r for r in stability_summary.to_dict("records")}
        else:
            score_map = {m: recursive_val_metrics[m]["mae"] for m in methods}
            stability_by_method = {}
        best_method = min(methods, key=lambda m: float(score_map[m]) if score_map[m] is not None else float("inf"))
        selected_methods[target] = best_method
        print(f"[{target}] selection дууслаа; selected={best_method}", flush=True)

        # One-step rolling validation, reported separately (not used as the primary selector).
        one_step_val_dates = list(pd.date_range(max(origins[0] + pd.offsets.MonthBegin(1), dev.index.max() - pd.DateOffset(months=5)), dev.index.max(), freq="MS"))
        one_val_preds, one_val_actuals, one_val_dates_used = one_step_rolling(dev, one_step_val_dates, templates)
        for method in methods:
            metric_rows.append({"target": target, "method": method, "evaluation": "validation_one_step",
                                "horizon": 1, **metric_pack(one_val_actuals, one_val_preds[method])})

        # Final one-step holdout: refit at each month using only earlier observations.
        one_holdout_preds, one_holdout_actuals, one_holdout_dates_used = one_step_rolling(series, list(holdout_dates), templates)
        print(f"[{target}] one-step holdout дууслаа", flush=True)
        for method in methods:
            metric_rows.append({"target": target, "method": method, "evaluation": "final_holdout_one_step",
                                "horizon": 1, **metric_pack(one_holdout_actuals, one_holdout_preds[method])})

        # Final recursive holdout: one fit at the last pre-holdout origin, then 12 recursive steps.
        final_origin = holdout_start - pd.offsets.MonthBegin(1)
        final_history = series.loc[series.index <= final_origin].asfreq("MS")
        final_future = pd.date_range(holdout_start, periods=HORIZON, freq="MS")
        final_recursive_preds: dict[str, list[float]] = {}
        final_actuals = series.reindex(final_future)
        for method in methods:
            final_recursive_preds[method] = recursive_forecast(method, templates.get(method), final_history, final_future)
            valid_dates = final_actuals.dropna().index
            actuals = [float(final_actuals.loc[d]) for d in valid_dates]
            # Use exact month-to-position mapping; do not assume every month has identical day count.
            pos_map = {d: i for i, d in enumerate(final_future)}
            preds = [final_recursive_preds[method][pos_map[d]] for d in valid_dates]
            metric_rows.append({"target": target, "method": method, "evaluation": "final_holdout_recursive_all_horizons",
                                "horizon": "1-12", **metric_pack(actuals, preds)})
            for h, date in enumerate(final_future, start=1):
                actual = final_actuals.loc[date]
                if pd.notna(actual):
                    metric_rows.append({"target": target, "method": method, "evaluation": "final_holdout_recursive_by_horizon",
                                        "horizon": h, **metric_pack([float(actual)], [final_recursive_preds[method][h-1]])})

        # Independent-in-time calibration origins, not used for model selection.
        # Each horizon is calibrated only from errors at that exact horizon.
        calibration_val = recursive_backtest(
            dev, calibration_origins, templates,
            method_names=[best_method], horizon=calibration_horizon
        ) if calibration_origins else {best_method: {h: [] for h in range(1, calibration_horizon + 1)}}
        print(f"[{target}] calibration дууслаа", flush=True)
        interval_radii: list[float] = []
        interval_calibration: list[dict[str, Any]] = []
        for h in range(1, HORIZON + 1):
            # A published forecast after k bridge months has effective horizon
            # bridge_month_count + k from the last observed data point.
            calibration_h = bridge_month_count + h
            rows = calibration_val[best_method][calibration_h]
            errors = np.array([abs(pred - actual) for _, actual, pred in rows], dtype=float)
            radius = conformal_radius(errors, alpha=0.05)
            interval_radii.append(float(radius) if radius is not None else np.nan)
            coverage = float(np.mean(errors <= radius)) if radius is not None and len(errors) else None
            width = float(2 * radius) if radius is not None else None
            enough = len(errors) >= args.min_calibration_errors
            info = {"horizon": h, "published_forecast_horizon": h,
                    "bridge_month_count": bridge_month_count,
                    "effective_calibration_horizon": calibration_h,
                    "calibration_residual_count": len(errors), "radius_ug_m3": radius,
                    "calibration_coverage_pct": coverage * 100 if coverage is not None else None,
                    "average_interval_width_ug_m3": width,
                    "95pct_finite_sample_rank": int(np.ceil((len(errors) + 1) * 0.95)) if len(errors) else None,
                    "minimum_recommended_calibration_count": args.min_calibration_errors,
                    "calibration_sample_sufficient": bool(enough),
                    "interval_status": "empirical_only_sample_small" if not enough else "empirical_calibration_check_required",
                    "sample_warning": f"{args.min_calibration_errors}-оос цөөн calibration алдаа: 95%-ийн coverage баталгаатай гэж үзэхгүй." if not enough else None}
            interval_calibration.append(info)
            metric_rows.append({"target": target, "method": best_method, "evaluation": "interval_validation_calibration",
                                "horizon": h, "effective_calibration_horizon": calibration_h,
                                "n": len(errors), "mae": None, "rmse": None, "r2": None,
                                "mape_pct": None, "smape_pct": None, "wape_pct": None,
                                "calibration_coverage_pct": info["calibration_coverage_pct"],
                                "average_interval_width_ug_m3": width})

        # Forecast the next 12 genuinely future calendar months. If the workbook
        # ends before the current month, recursively bridge the unobserved elapsed
        # months first; do not publish stale forecasts for months already passed.
        recursive_dates = bridge_dates.append(full_future) if len(bridge_dates) else full_future
        full_models: dict[str, Any] = {}
        X_full, y_full = make_supervised(series)
        for name, template in templates.items():
            fitted = clone(template).fit(X_full, y_full)
            full_models[name] = fitted
            joblib.dump(fitted, args.models_dir / f"{target.replace('.', '')}_{name}.joblib")
        future_scenarios: dict[str, list[float]] = {}
        for method in methods:
            if method in BASELINES:
                all_preds = recursive_forecast(method, None, series, recursive_dates)
            else:
                all_preds = recursive_forecast(method, full_models[method], series, recursive_dates)
            future_scenarios[method] = all_preds[bridge_month_count:]
            for bridge_i, bridge_date in enumerate(bridge_dates):
                bridge_rows.append({"month": bridge_date.strftime("%Y-%m-%d"), "target": target,
                                    "method": method, "bridge_forecast_ug_m3": float(all_preds[bridge_i]),
                                    "selected_method": method == best_method,
                                    "note": "Өнгөрсөн боловч хэмжилтгүй сарын дотоод recursive bridge; үндсэн 12 сарын forecast-д оруулаагүй."})
        chosen = np.asarray(future_scenarios[best_method], dtype=float)
        print(f"[{target}] future forecast дууслаа", flush=True)
        radii = np.asarray(interval_radii, dtype=float)
        lower = np.maximum(0.0, chosen - radii)
        upper = chosen + radii
        for i, date in enumerate(full_future):
            forecast_rows.append({"month": date.strftime("%Y-%m-%d"), "target": target,
                                  "forecast_ug_m3": float(chosen[i]), "selected_method": best_method,
                                  "forecast_horizon": i + 1})
            interval_rows.append({"month": date.strftime("%Y-%m-%d"), "target": target,
                                  "horizon": i + 1,
                                  "effective_calibration_horizon": interval_calibration[i]["effective_calibration_horizon"],
                                  "forecast_ug_m3": float(chosen[i]),
                                  "lower_95_empirical_ug_m3": float(lower[i]),
                                  "upper_95_empirical_ug_m3": float(upper[i]),
                                  "interval_radius_ug_m3": float(radii[i]),
                                  "calibration_residual_count": interval_calibration[i]["calibration_residual_count"],
                                  "calibration_coverage_pct": interval_calibration[i]["calibration_coverage_pct"],
                                  "average_interval_width_ug_m3": interval_calibration[i]["average_interval_width_ug_m3"],
                                  "interval_status": interval_calibration[i]["interval_status"],
                                  "coverage_claim": "empirical calibration only; not guaranteed future 95% coverage"})
            for method in methods:
                scenario_rows.append({"month": date.strftime("%Y-%m-%d"), "target": target,
                                      "method": method, "forecast_ug_m3": float(future_scenarios[method][i]),
                                      "selected_method": method == best_method})

        # Save validation and final holdout prediction records for auditability.
        val_records = []
        for method in methods:
            for h, rows in recursive_val[method].items():
                for date, actual, pred in rows:
                    val_records.append({"method": method, "origin_horizon": h, "month": date,
                                        "actual_ug_m3": actual, "predicted_ug_m3": pred,
                                        "absolute_error_ug_m3": abs(pred - actual)})
        val_frame = pd.DataFrame(val_records)
        val_frame.to_csv(args.out / f"recursive_validation_{target.replace('.', '')}.csv", index=False, encoding="utf-8-sig")
        calibration_records = []
        for h, rows in calibration_val[best_method].items():
            for date, actual, pred in rows:
                calibration_records.append({"method": best_method, "horizon": h, "month": date,
                                            "actual_ug_m3": actual, "predicted_ug_m3": pred,
                                            "absolute_error_ug_m3": abs(pred - actual)})
        pd.DataFrame(calibration_records).to_csv(
            args.out / f"interval_calibration_{target.replace('.', '')}.csv",
            index=False, encoding="utf-8-sig")
        holdout_records = []
        for method in methods:
            for h, date in enumerate(final_future, start=1):
                actual = final_actuals.loc[date]
                holdout_records.append({"method": method, "horizon": h, "month": date,
                                        "actual_ug_m3": float(actual) if pd.notna(actual) else None,
                                        "predicted_ug_m3": final_recursive_preds[method][h-1]})
        holdout_frame = pd.DataFrame(holdout_records)
        holdout_frame.to_csv(args.out / f"recursive_holdout_{target.replace('.', '')}.csv", index=False, encoding="utf-8-sig")
        one_step_frame = pd.DataFrame({"month": one_holdout_dates_used,
                                       "actual_ug_m3": one_holdout_actuals})
        for method in methods:
            one_step_frame[f"pred_{method}"] = one_holdout_preds[method]
        one_step_frame.to_csv(args.out / f"one_step_holdout_{target.replace('.', '')}.csv", index=False, encoding="utf-8-sig")
        all_validation_plot[target] = val_frame
        all_holdout_plot[target] = holdout_frame

        so2_info = so2_diagnostics(series) if target == "SO2" else None
        # October 2026 status: only report an actual comparison if the workbook contains that month.
        oct_date = pd.Timestamp("2026-10-01")
        oct_actual = series.get(oct_date, np.nan)
        oct_forecast = None
        oct_status = "actual_missing"
        if pd.notna(oct_actual):
            if oct_date in final_future:
                oct_pos = list(final_future).index(oct_date)
                oct_forecast = final_recursive_preds[best_method][oct_pos]
                oct_status = "compared_with_recursive_final_holdout_prediction"
            elif oct_date in full_future:
                oct_pos = list(full_future).index(oct_date)
                oct_forecast = float(chosen[oct_pos])
                oct_status = "actual_is_in_future_forecast_range_in_this_run_check_data_dates"
            else:
                oct_status = "actual_present_but_not_in_selected_validation_or_holdout_window"
        target_summaries[target] = {
            "observed_month_count": int(len(observed)), "first_observed_month": observed.index.min(),
            "last_observed_month": observed.index.max(), "forecast_bridge_months_skipped_from_output": bridge_month_count,
            "forecast_start": forecast_start, "validation_origins": origins,
            "validation_origin_count": len(origins),
            "selection_origin_count_requested": args.selection_origins,
            "max_validation_origins_requested": args.max_origins,
            "model_selection_origins": selection_origins,
            "model_selection_origin_count": len(selection_origins),
            "interval_calibration_origins": calibration_origins,
            "interval_calibration_origin_count": len(calibration_origins),
            "bridge_month_count": bridge_month_count,
            "calibration_horizon_months": calibration_horizon,
            "model_stability_by_method": stability_by_method,
            "final_holdout_start": holdout_start,
            "final_holdout_end": observed.index.max(),
            "selected_method_by_mean_origin_mae": best_method,
            "selected_method_by_recursive_validation_mae": best_method,
            "tuning": tuning_info, "validation_recursive_metrics": recursive_val_metrics,
            "validation_one_step_metrics": {m: metric_pack(one_val_actuals, one_val_preds[m]) for m in methods},
            "final_holdout_one_step_metrics": {m: metric_pack(one_holdout_actuals, one_holdout_preds[m]) for m in methods},
            "final_holdout_recursive_metrics": {
                m: metric_pack([float(v) for v in final_actuals.dropna().values],
                               [final_recursive_preds[m][i] for i, d in enumerate(final_future) if pd.notna(final_actuals.loc[d])])
                for m in methods},
            "prediction_interval_by_horizon": interval_calibration,
            "SO2_diagnostics": so2_info,
            "october_2026_check": {"actual_ug_m3": float(oct_actual) if pd.notna(oct_actual) else None,
                                   "prediction_ug_m3": oct_forecast,
                                   "absolute_error_ug_m3": abs(float(oct_actual) - oct_forecast) if pd.notna(oct_actual) and oct_forecast is not None else None,
                                   "status": oct_status},
        }

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(args.out / "model_metrics.csv", index=False, encoding="utf-8-sig")
    forecasts = pd.DataFrame(forecast_rows)
    forecasts.to_csv(args.out / "forecast_12_months.csv", index=False, encoding="utf-8-sig")
    scenarios_df = pd.DataFrame(scenario_rows)
    scenarios_df.to_csv(args.out / "forecast_scenarios.csv", index=False, encoding="utf-8-sig")
    intervals_df = pd.DataFrame(interval_rows)
    intervals_df.to_csv(args.out / "forecast_intervals.csv", index=False, encoding="utf-8-sig")
    stability_df = pd.DataFrame(stability_rows)
    stability_df.to_csv(args.out / "model_stability_by_origin.csv", index=False, encoding="utf-8-sig")
    bridge_df = pd.DataFrame(bridge_rows)
    bridge_df.to_csv(args.out / "bridge_forecasts.csv", index=False, encoding="utf-8-sig")

    # Target forecast plots are generated from the same forecast/interval tables
    # that were exported to CSV, avoiding a second independent calculation.
    for target in TARGETS:
        f = forecasts[forecasts["target"] == target].sort_values("month")
        iv = intervals_df[intervals_df["target"] == target].sort_values("month")
        save_forecast_plot(
            target, all_series[target], pd.DatetimeIndex(pd.to_datetime(f["month"])),
            f["forecast_ug_m3"].to_numpy(dtype=float),
            iv["lower_95_empirical_ug_m3"].to_numpy(dtype=float),
            iv["upper_95_empirical_ug_m3"].to_numpy(dtype=float),
            args.out / f"forecast_{target.replace('.', '')}.png",
        )

    # Model comparison uses the same mean origin-level MAE used for selection.
    chart_rows = metrics_df[metrics_df["evaluation"] == "validation_origin_stability"].copy()
    fig, ax = plt.subplots(figsize=(13, 6))
    labels = [f"{r.target}\n{r.method}" for r in chart_rows.itertuples()]
    ax.bar(np.arange(len(chart_rows)), chart_rows["mae"].astype(float),
           yerr=chart_rows.get("origin_mae_std_ug_m3", pd.Series([0] * len(chart_rows))).fillna(0).to_numpy())
    ax.set_xticks(np.arange(len(chart_rows))); ax.set_xticklabels(labels, rotation=65, ha="right", fontsize=8)
    ax.set_ylabel("Mean origin-level MAE (µg/m³)")
    ax.set_title("Загварын тогтвортой байдал — origin MAE mean ± SD")
    ax.grid(axis="y", alpha=0.25); fig.tight_layout()
    fig.savefig(args.out / "model_comparison.png", dpi=160); plt.close(fig)

    # Horizon-wise validation MAE for the selected model per pollutant.
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for target in TARGETS:
        method = selected_methods[target]
        rows = metrics_df[(metrics_df.target == target) &
                          (metrics_df.method == method) &
                          (metrics_df.evaluation == "validation_recursive_by_horizon")]
        ax.plot(rows["horizon"], rows["mae"], marker="o", label=f"{target} ({method})")
    ax.set_xticks(range(1, HORIZON + 1)); ax.set_xlabel("Прогнозын алхам (сар)")
    ax.set_ylabel("MAE (µg/m³)"); ax.set_title("Recursive validation алдаа — horizon тус бүр")
    ax.grid(alpha=0.25); ax.legend(); fig.tight_layout()
    fig.savefig(args.out / "horizon_error_comparison.png", dpi=160); plt.close(fig)

    summary["target_results"] = target_summaries
    summary["forecast_horizon_months"] = HORIZON
    summary["forecast_period_by_target"] = {
        t: {"start": str(pd.to_datetime(forecasts.loc[forecasts.target == t, "month"]).min().date()),
            "end": str(pd.to_datetime(forecasts.loc[forecasts.target == t, "month"]).max().date())}
        for t in TARGETS}
    summary["selected_forecast_methods"] = selected_methods
    summary["output_files"] = [
        "clean_monthly_data.csv", "model_metrics.csv", "forecast_12_months.csv",
        "forecast_scenarios.csv", "forecast_intervals.csv", "model_stability_by_origin.csv",
        "bridge_forecasts.csv", "interval_calibration_PM10.csv", "interval_calibration_PM25.csv",
        "interval_calibration_SO2.csv", "analysis_summary.json",
        "forecast_PM10.png", "forecast_PM25.png", "forecast_SO2.png", "model_comparison.png",
        "horizon_error_comparison.png", "recursive_validation_PM10.csv", "recursive_validation_PM25.csv",
        "recursive_validation_SO2.csv", "recursive_holdout_PM10.csv", "recursive_holdout_PM25.csv",
        "recursive_holdout_SO2.csv", "one_step_holdout_PM10.csv", "one_step_holdout_PM25.csv",
        "one_step_holdout_SO2.csv", "models/*.joblib",
    ]
    (args.out / "analysis_summary.json").write_text(
        json.dumps(json_safe(summary), ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({"status": "completed", "python_version": __import__("sys").version.split()[0],
                      "selected_forecast_methods": selected_methods,
                      "forecast_period_by_target": summary["forecast_period_by_target"],
                      "results_dir": str(args.out.resolve()), "models_dir": str(args.models_dir.resolve())},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
