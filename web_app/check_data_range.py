
from pathlib import Path
import pandas as pd
import json

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS = BASE_DIR / "results"

def check_csv(filename, date_column):
    path = RESULTS / filename

    print(f"\n{'=' * 50}")
    print(filename)
    print("=" * 50)

    if not path.exists():
        print("Файл олдсонгүй!")
        return

    df = pd.read_csv(path)

    print("Нийт мөр:", len(df))
    print("Баганууд:", list(df.columns))

    if date_column in df.columns:
        dates = pd.to_datetime(df[date_column], errors="coerce")
        print("Эхлэх огноо:", dates.min())
        print("Дуусах огноо:", dates.max())

    print("\nХоосон утгууд:")
    print(df.isna().sum())

    if "target" in df.columns:
        print("\nҮзүүлэлт тус бүр:")
        print(df["target"].value_counts())

    if "method" in df.columns:
        print("\nАшигласан аргууд:")
        print(df["method"].value_counts())

    return df


# Historical data
historical = check_csv(
    "clean_monthly_data.csv",
    "month"
)

# Air pollution forecasts
pollution = check_csv(
    "forecast_12_months.csv",
    "month"
)

# Weather forecasts
weather = check_csv(
    "weather_forecast_12_months.csv",
    "month"
)

# Selected methods
summary_path = RESULTS / "analysis_summary.json"

print("\n========== SELECTED METHODS ==========")

if summary_path.exists():
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    print(
        json.dumps(
            summary.get("selected_forecast_methods", {}),
            ensure_ascii=False,
            indent=4
        )
    )

print("\n========== CHECK COMPLETED ==========")
