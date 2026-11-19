
import pandas as pd
from pathlib import Path


# Шаардлагатай баганууд
REQUIRED_COLUMNS = [
    "date",
    "temperature",
    "max_wind_speed",
    "precipitation"
]


def read_weather_file(file):
    """
    Excel эсвэл CSV файл уншина.
    """

    filename = file.filename.lower()

    if filename.endswith(".csv"):
        df = pd.read_csv(file)

    elif filename.endswith((".xlsx", ".xls")):
        df = pd.read_excel(file)

    else:
        raise ValueError(
            "Зөвхөн CSV болон Excel файл оруулах боломжтой."
        )

    return df


def validate_weather_data(df):
    """
    Өгөгдлийн багана болон утгуудыг шалгана.
    """

    # Баганын нэрийг цэвэрлэх
    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # Шаардлагатай баганууд байгаа эсэх
    missing_columns = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Дараах баганууд дутуу байна: {missing_columns}"
        )

    # Зөвхөн шаардлагатай багануудыг авах
    df = df[REQUIRED_COLUMNS].copy()

    # Огноо хөрвүүлэх
    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    if df["date"].isna().any():
        raise ValueError(
            "Огнооны формат буруу эсвэл хоосон байна."
        )

    # Тоон утга болгон хөрвүүлэх
    numeric_columns = [
        "temperature",
        "max_wind_speed",
        "precipitation"
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    if df[numeric_columns].isna().any().any():
        raise ValueError(
            "Температур, салхи, хур тунадасны утга "
            "хоосон эсвэл буруу байна."
        )

    # Хязгаарын шалгалт
    if (df["max_wind_speed"] < 0).any():
        raise ValueError(
            "Салхины хурд сөрөг байж болохгүй."
        )

    if (df["precipitation"] < 0).any():
        raise ValueError(
            "Хур тунадас сөрөг байж болохгүй."
        )

    # Давхардсан огноо шалгах
    if df["date"].duplicated().any():
        raise ValueError(
            "Давхардсан огноо байна. Өдөр бүрийн "
            "нэг л бүртгэл оруулна уу."
        )

    # Огноогоор эрэмбэлэх
    df = df.sort_values("date").reset_index(drop=True)

    return df


def calculate_monthly_weather(df):
    """
    Өдөр тутмын цаг уурын өгөгдлийг
    сарын үзүүлэлт болгон нэгтгэнэ.
    """

    df = validate_weather_data(df)

    # Жил, сарын багана үүсгэх
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    # Сарын үзүүлэлтүүдийг тооцох
    monthly = (
        df.groupby(["year", "month"])
        .agg(
            temperature=("temperature", "mean"),
            max_wind_speed=("max_wind_speed", "max"),
            precipitation=("precipitation", "sum"),
            recorded_days=("date", "count")
        )
        .reset_index()
    )

    # Сарын огноо үүсгэх
    monthly["date"] = pd.to_datetime(
        monthly[["year", "month"]].assign(day=1)
    )

    # Багануудын дараалал
    monthly = monthly[
        [
            "date",
            "temperature",
            "max_wind_speed",
            "precipitation",
            "recorded_days"
        ]
    ]

    # Бүх тоог зохих нарийвчлалтай болгох
    monthly["temperature"] = monthly[
        "temperature"
    ].round(2)

    monthly["max_wind_speed"] = monthly[
        "max_wind_speed"
    ].round(2)

    monthly["precipitation"] = monthly[
        "precipitation"
    ].round(2)

    return monthly


def process_weather_file(file):
    """
    Файлыг уншиж, шалгаж, сарын өгөгдөл буцаана.
    """

    daily_data = read_weather_file(file)

    monthly_data = calculate_monthly_weather(
        daily_data
    )

    return daily_data, monthly_data
