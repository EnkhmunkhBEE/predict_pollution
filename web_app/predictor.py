
from pathlib import Path
import sys
import json

import numpy as np
import pandas as pd


# ==========================================
# PROJECT PATH
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

SRC_DIR = BASE_DIR / "src"
RESULTS_DIR = BASE_DIR / "results"

HISTORICAL_FILE = RESULTS_DIR / "clean_monthly_data.csv"
SUMMARY_FILE = RESULTS_DIR / "analysis_summary.json"


# ==========================================
# CORE TRAINER
# ==========================================

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import train_predict_improved_weather as trainer


# ==========================================
# VARIABLES
# ==========================================

TARGETS = ["PM10", "PM2.5", "SO2"]

WEATHER_COLUMNS = {
    "temperature": "temperature",
    "max_wind_speed": "max wind speed",
    "precipitation": "precipitation"
}


# ==========================================
# LOAD HISTORICAL DATA
# ==========================================

def load_data():

    if not HISTORICAL_FILE.exists():
        raise FileNotFoundError(
            f"Өгөгдлийн файл олдсонгүй: {HISTORICAL_FILE}"
        )

    df = pd.read_csv(HISTORICAL_FILE)

    if "month" not in df.columns:
        raise ValueError(
            "clean_monthly_data.csv файлд month багана байхгүй байна."
        )

    df["month"] = pd.to_datetime(
        df["month"],
        errors="coerce"
    )

    df = df.dropna(subset=["month"])

    df["month"] = (
        df["month"]
        .dt.to_period("M")
        .dt.to_timestamp()
    )

    df = (
        df.sort_values("month")
        .drop_duplicates("month", keep="last")
        .set_index("month")
        .asfreq("MS")
    )

    required_columns = TARGETS + list(
        WEATHER_COLUMNS.values()
    )

    for column in required_columns:

        if column not in df.columns:
            df[column] = np.nan

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # Түүхэн цаг уурын өгөгдөл
    weather = {
        column: df[column].astype(float)
        for column in WEATHER_COLUMNS.values()
        if df[column].notna().any()
    }

    # Бүх түүхэн өгөгдлийг хадгална.
    # Ингэснээр бодит цаг уурын утгууд ч харагдана.
    return df, weather


# ==========================================
# LOAD SELECTED MODELS
# ==========================================

def load_selected_methods():

    if not SUMMARY_FILE.exists():
        return {}

    with open(
        SUMMARY_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        summary = json.load(file)

    return summary.get(
        "selected_forecast_methods",
        {}
    )


# ==========================================
# TRAIN AND FORECAST
# ==========================================

def train_and_forecast(
    historical,
    weather,
    end_date
):

    historical_end = historical.index.max()

    if end_date <= historical_end:
        return {}, {}, {}

    # Сүүлийн бодит өгөгдлөөс эхлэн
    # хэрэглэгчийн сонгосон дуусах сар хүртэл прогнозлоно.
    forecast_dates = pd.date_range(
        start=historical_end + pd.offsets.MonthBegin(1),
        end=end_date,
        freq="MS"
    )

    selected_methods = load_selected_methods()

    pollution_forecasts = {}
    weather_forecasts = {}
    selected_models = {}

    # --------------------------------------
    # WEATHER FORECAST
    # --------------------------------------

    for output_name, source_name in WEATHER_COLUMNS.items():

        if source_name not in weather:
            continue

        history = weather[source_name]

        weather_forecasts[source_name] = {
            date: trainer.forecast_weather(
                history,
                date,
                source_name
            )
            for date in forecast_dates
        }

    # --------------------------------------
    # POLLUTION MODEL TRAINING
    # --------------------------------------

    for target in TARGETS:

        print(f"{target}: Core Trainer ажиллаж байна...")

        series = historical[target].astype(float).asfreq("MS")

        # Core Trainer-аар загваруудыг сургаж,
        # tuning мэдээллийг авна.
        templates, tuning_info = trainer.tune_models(
            series,
            weather
        )

        # Өмнөх validation-аар сонгосон загварыг ашиглана.
        method = selected_methods.get(target)

        # Сонгосон загвар олдохгүй бол tuning MAE хамгийн
        # бага загварыг сонгоно.
        if (
            method not in templates
            and method not in trainer.BASELINES
        ):

            if not tuning_info:
                raise ValueError(
                    f"{target}: тохирох загвар олдсонгүй."
                )

            method = min(
                tuning_info,
                key=lambda name:
                    tuning_info[name]["tuning_cv_mae_ug_m3"]
            )

        selected_models[target] = method

        # Хэрэглэгчийн сонгосон хугацаа хүртэл
        # recursive forecast хийнэ.
        predictions = trainer.recursive_forecast(
            method=method,
            template=templates.get(method),
            history=series,
            dates=forecast_dates,
            weather=weather
        )

        pollution_forecasts[target] = {
            date: float(value)
            for date, value in zip(
                forecast_dates,
                predictions
            )
        }

    return (
        pollution_forecasts,
        weather_forecasts,
        selected_models
    )


# ==========================================
# GET ACTUAL OR FORECAST VALUE
# ==========================================

def get_value(
    month,
    column,
    historical,
    forecasts
):

    # 1. Бодит хэмжилт байгаа эсэхийг шалгана.
    if column in historical.columns and month in historical.index:

        actual = historical.at[month, column]

        if pd.notna(actual):

            return {
                "value": round(float(actual), 2),
                "source": "actual"
            }

    # 2. Ирээдүйн сарын прогноз.
    # Өнгөрсөн боловч хэмжилтгүй сарыг backcast гэж
    # харуулахгүй.
    current_month = (
        pd.Timestamp.today()
        .to_period("M")
        .to_timestamp()
    )

    if month > current_month and month in forecasts:

        return {
            "value": round(
                float(forecasts[month]),
                2
            ),
            "source": "forecast"
        }

    # 3. Бодит болон прогнозын утга байхгүй.
    return {
        "value": None,
        "source": "unavailable"
    }


# ==========================================
# CALCULATE SUMMARY
# ==========================================

def calculate_summary(rows):

    columns = [
        "PM10",
        "PM2.5",
        "SO2",
        "temperature",
        "max_wind_speed",
        "precipitation"
    ]

    summary = {}

    for column in columns:

        values = [
            row[column]["value"]
            for row in rows
            if row[column]["value"] is not None
        ]

        if not values:

            summary[column] = None

        elif column == "precipitation":

            # Хур тунадасны нийлбэр
            summary[column] = round(
                sum(values),
                2
            )

        else:

            # Бусад үзүүлэлтийн дундаж
            summary[column] = round(
                sum(values) / len(values),
                2
            )

    return summary


# ==========================================
# MAIN PREDICTION FUNCTION
# ==========================================

def predict_by_date(start_date, end_date):

    start = pd.to_datetime(
        start_date,
        errors="coerce"
    )

    end = pd.to_datetime(
        end_date,
        errors="coerce"
    )

    if pd.isna(start) or pd.isna(end):

        raise ValueError(
            "Огнооны формат буруу байна."
        )

    if start > end:

        raise ValueError(
            "Эхлэх огноо дуусах огнооноос хойш байж болохгүй."
        )

    # Сарын эхний өдөр болгоно.
    start = start.to_period("M").to_timestamp()
    end = end.to_period("M").to_timestamp()

    # Бүх түүхэн өгөгдөл
    historical, weather = load_data()

    historical_end = historical.index.max()

    # Core Trainer ашиглан прогноз гаргана.
    (
        pollution_forecasts,
        weather_forecasts,
        selected_models
    ) = train_and_forecast(
        historical,
        weather,
        end
    )

    # Сонгосон хугацааны сарууд
    months = pd.date_range(
        start=start,
        end=end,
        freq="MS"
    )

    rows = []

    for month in months:

        row = {
            "month": month.strftime("%Y-%m")
        }

        # PM10, PM2.5, SO2
        for column in TARGETS:

            row[column] = get_value(
                month,
                column,
                historical,
                pollution_forecasts.get(column, {})
            )

        # Температур, салхи, хур тунадас
        for output_name, source_name in WEATHER_COLUMNS.items():

            row[output_name] = get_value(
                month,
                source_name,
                historical,
                weather_forecasts.get(source_name, {})
            )

        rows.append(row)

    # Нэгтгэсэн тооцоолол
    summary = calculate_summary(rows)

    return {
        "success": True,

        "start_date": start.strftime("%Y-%m"),
        "end_date": end.strftime("%Y-%m"),

        "historical_data_end": historical_end.strftime("%Y-%m"),

        "selected_models": selected_models,

        "monthly_results": rows,

        "summary": summary,

        "forecast_note": (
            "Бодит хэмжилттэй сарууд actual, "
            "ирээдүйн сарууд forecast гэж тэмдэглэгдэнэ. "
            "Алс хугацааны прогнозын тодорхой бус байдал нэмэгдэж болно."
        )
    }
