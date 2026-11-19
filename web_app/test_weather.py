
import pandas as pd

from weather_processor import calculate_monthly_weather


data = {
    "date": [
        "2026-11-01",
        "2026-11-02",
        "2026-11-03",
        "2026-11-04",
        "2026-11-05"
    ],

    "temperature": [-12, -15, -10, -13, -8],

    "max_wind_speed": [8, 12, 9, 15, 11],

    "precipitation": [0, 0.2, 0, 1.5, 0.5]
}

df = pd.DataFrame(data)

result = calculate_monthly_weather(df)

print("\nСАРЫН ЦАГ УУРЫН ҮЗҮҮЛЭЛТ:")
print(result.to_string(index=False))
