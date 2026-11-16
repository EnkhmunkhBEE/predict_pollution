
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import math

# Хавтас
RESULTS = Path("results")
OUTPUT = RESULTS / "charts"
OUTPUT.mkdir(parents=True, exist_ok=True)

# Прогнозын CSV
df = pd.read_csv(RESULTS / "forecast_12_months.csv")
df["month"] = pd.to_datetime(df["month"])

# График гаргах үзүүлэлтүүд
targets = ["PM10", "PM2.5", "SO2"]

# Өнгө
colors = {
    "PM10": "#3478C8",
    "PM2.5": "#E39B35",
    "SO2": "#31966B"
}

for target in targets:

    data = df[df["target"] == target].copy()
    data = data.sort_values("month")

    if data.empty:
        print(f"{target}: өгөгдөл олдсонгүй!")
        continue

    dates = data["month"]
    values = data["forecast_ug_m3"]

    last_date = dates.iloc[-1]
    last_value = values.iloc[-1]

    # График
    fig, ax = plt.subplots(figsize=(13, 6), dpi=150)

    ax.plot(
        dates,
        values,
        color=colors[target],
        linewidth=2.5,
        marker="o",
        markersize=6,
        label=f"{target} прогноз"
    )

    # Сүүлийн цэг
    ax.scatter(
        last_date,
        last_value,
        color="#E05252",
        s=100,
        zorder=5
    )

    # Сүүлийн утгын тайлбар
    ax.annotate(
        f"{last_date:%Y.%m}\n{target} {last_value:.1f} µg/m³",
        xy=(last_date, last_value),
        xytext=(-15, 25),
        textcoords="offset points",
        ha="right",
        fontsize=11,
        fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor="white",
            edgecolor="#dddddd"
        )
    )

    # Гарчиг
    ax.set_title(
        f"{target}-ийн сарын таамаглал",
        fontsize=18,
        fontweight="bold",
        pad=25
    )

    # Хугацаа
    ax.text(
        0.5, 1.02,
        f"{dates.iloc[0]:%Y.%m}–{last_date:%Y.%m} | µg/m³",
        transform=ax.transAxes,
        ha="center",
        fontsize=11,
        color="gray"
    )

    # Y тэнхлэгийг автоматаар тохируулах
    max_value = max(values.max(), 1)
    y_max = math.ceil(max_value / 15) * 15
    ax.set_ylim(0, y_max)
    ax.set_yticks(range(0, y_max + 1, 15))
    ax.set_ylabel("Концентраци (µg/m³)")

    # X тэнхлэг
    ax.set_xticks(dates)
    ax.xaxis.set_major_formatter(
        mdates.DateFormatter("%Y.%m")
    )

    plt.xticks(rotation=0, fontsize=9)

    # Загвар
    ax.grid(axis="y", alpha=0.25)
    ax.grid(axis="x", alpha=0.12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    # Зураг хадгалах
    filename = OUTPUT / f"{target.replace('.', '')}_monthly_forecast.png"

    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"{target} график хадгаллаа: {filename}")

print("\nБүх график амжилттай үүслээ!")
