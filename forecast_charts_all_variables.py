from pathlib import Path
import argparse
import math
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

POLLUTANT_COLORS = {
    "PM10": "#3478C8",
    "PM2.5": "#E39B35",
    "SO2": "#31966B",
}
WEATHER_ORDER = ["temperature", "max wind speed", "precipitation"]
WEATHER_LABELS = {
    "temperature": ("Агаарын температур", "Температур (°C)"),
    "max wind speed": ("Хамгийн их салхины хурд", "Салхины хурд (эх өгөгдлийн нэгжээр)"),
    "precipitation": ("Хур тунадас", "Хур тунадас (эх өгөгдлийн нэгжээр)"),
}


def style_time_axis(ax):
    ax.set_xlabel("Сар")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y.%m"))
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def load_forecast_csv(path, required, description):
    if not path.exists():
        raise FileNotFoundError(
            f"{description} олдсонгүй: {path}\n"
            "Файлын зөв замыг --input эсвэл --weather параметрээр өгнө үү."
        )
    df = pd.read_csv(path)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{description}-д дараах багана дутуу байна: "
            + ", ".join(sorted(missing))
        )
    df["month"] = pd.to_datetime(df["month"], errors="coerce")
    return df


def plot_pollutants(csv_path, output_dir):
    df = load_forecast_csv(
        csv_path, {"month", "target", "forecast_ug_m3"}, "Бохирдлын прогнозын CSV"
    )
    df["forecast_ug_m3"] = pd.to_numeric(df["forecast_ug_m3"], errors="coerce")
    df = df.dropna(subset=["month", "target", "forecast_ug_m3"])
    output_dir.mkdir(parents=True, exist_ok=True)

    for target in ["PM10", "PM2.5", "SO2"]:
        data = df[df["target"].astype(str).str.upper() == target.upper()].sort_values("month")
        if data.empty:
            print(f"АНХААРУУЛГА: {target} өгөгдөл олдсонгүй.")
            continue

        dates, values = data["month"], data["forecast_ug_m3"]
        fig, ax = plt.subplots(figsize=(13, 6), dpi=150)
        ax.plot(dates, values, color=POLLUTANT_COLORS[target], linewidth=2.5,
                marker="o", markersize=6, label=f"{target} таамаглал")
        ax.fill_between(dates, values, alpha=0.10, color=POLLUTANT_COLORS[target])

        for idx, label, offset in [
            (values.idxmax(), "Хамгийн их", (0, 15)),
            (values.idxmin(), "Хамгийн бага", (0, -28)),
        ]:
            x, y = data.loc[idx, "month"], data.loc[idx, "forecast_ug_m3"]
            ax.annotate(f"{label}: {y:.2f}", xy=(x, y), xytext=offset,
                        textcoords="offset points", ha="center", fontsize=9,
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc"))

        ax.set_title(f"{target}-ийн сарын агууламжийн таамаглал",
                     fontsize=16, fontweight="bold", pad=16)
        ax.set_ylabel("Концентраци (µg/m³)")
        step = max(1, math.ceil(max(float(values.max()), 1) / 8 / 5) * 5)
        ax.set_ylim(bottom=0, top=math.ceil(max(float(values.max()), 1) * 1.18 / step) * step)
        ax.yaxis.set_major_locator(plt.MultipleLocator(step))
        style_time_axis(ax)
        ax.legend(frameon=False)
        fig.tight_layout()
        filename = output_dir / f"{target.replace('.', '')}_forecast.png"
        fig.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Хадгаллаа: {filename}")

    # Гурван бохирдуулагчийг нэг зурагт харьцуулна.
    fig, ax = plt.subplots(figsize=(14, 7), dpi=150)
    for target in ["PM10", "PM2.5", "SO2"]:
        data = df[df["target"].astype(str).str.upper() == target.upper()].sort_values("month")
        if not data.empty:
            ax.plot(data["month"], data["forecast_ug_m3"], marker="o",
                    linewidth=2, label=target, color=POLLUTANT_COLORS[target])
    ax.set_title("PM10, PM2.5, SO₂ үзүүлэлтүүдийн таамаглалын харьцуулалт",
                 fontsize=15, fontweight="bold", pad=16)
    ax.set_ylabel("Концентраци (µg/m³)")
    style_time_axis(ax)
    ax.legend(frameon=False)
    fig.tight_layout()
    path = output_dir / "all_pollutants_comparison.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Хадгаллаа: {path}")


def normalize_weather_name(value):
    text = str(value).strip().lower()
    aliases = {
        "temp": "temperature",
        "temperature": "temperature",
        "air temperature": "temperature",
        "max wind speed": "max wind speed",
        "maximum wind speed": "max wind speed",
        "wind speed": "max wind speed",
        "precipitation": "precipitation",
        "rainfall": "precipitation",
    }
    return aliases.get(text, text)


def plot_weather(csv_path, output_dir):
    df = load_forecast_csv(
        csv_path, {"month", "feature", "forecast_value"}, "Цаг уурын прогнозын CSV"
    )
    df["forecast_value"] = pd.to_numeric(df["forecast_value"], errors="coerce")
    df["feature"] = df["feature"].map(normalize_weather_name)
    df = df.dropna(subset=["month", "feature", "forecast_value"])
    output_dir.mkdir(parents=True, exist_ok=True)

    found = set(df["feature"].unique())
    expected = set(WEATHER_ORDER)
    missing = expected - found
    if missing:
        print("АНХААРУУЛГА: цаг уурын CSV-д дараах хувьсагч олдсонгүй: "
              + ", ".join(sorted(missing)))

    # Хувьсагч тус бүрийн график.
    for feature in WEATHER_ORDER:
        data = df[df["feature"] == feature].sort_values("month")
        if data.empty:
            continue
        title, ylabel = WEATHER_LABELS[feature]
        fig, ax = plt.subplots(figsize=(12, 5), dpi=150)
        ax.plot(data["month"], data["forecast_value"], marker="o", linewidth=2.3)
        ax.set_title(f"{title}-ийн сарын таамаглал", fontsize=15, fontweight="bold")
        ax.set_ylabel(ylabel)
        style_time_axis(ax)
        fig.tight_layout()
        filename = output_dir / f"weather_{feature.replace(' ', '_')}.png"
        fig.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Хадгаллаа: {filename}")

    # Температур, салхи, хур тунадасыг тус тусын Y тэнхлэгтэй 3 дэд графикаар харуулна.
    fig, axes = plt.subplots(3, 1, figsize=(13, 11), dpi=150, sharex=True)
    for ax, feature in zip(axes, WEATHER_ORDER):
        data = df[df["feature"] == feature].sort_values("month")
        title, ylabel = WEATHER_LABELS[feature]
        if data.empty:
            ax.text(0.5, 0.5, f"{title}: өгөгдөл байхгүй",
                    transform=ax.transAxes, ha="center", va="center")
        else:
            ax.plot(data["month"], data["forecast_value"], marker="o", linewidth=2.2)
        ax.set_title(title, loc="left", fontsize=12, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[-1].set_xlabel("Сар")
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y.%m"))
    plt.setp(axes[-1].get_xticklabels(), rotation=35, ha="right")
    fig.suptitle("Цаг уурын гурван үзүүлэлтийн сарын таамаглал",
                 fontsize=16, fontweight="bold")
    fig.tight_layout()
    combined = output_dir / "all_weather_variables.png"
    fig.savefig(combined, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Хадгаллаа: {combined}")


def main():
    parser = argparse.ArgumentParser(
        description="Агаарын бохирдол болон бүх цаг уурын прогнозын график үүсгэх"
    )
    parser.add_argument("--input", default="results/forecast_12_months.csv",
                        help="Агаарын бохирдлын прогнозын CSV")
    parser.add_argument("--weather", default="results/weather_forecast.csv",
                        help="Температур, салхины хурд, хур тунадасны прогнозын CSV")
    parser.add_argument("--out", default="results/charts",
                        help="График хадгалах хавтас")
    parser.add_argument("--skip-weather", action="store_true",
                        help="Цаг уурын график үүсгэхгүй")
    args = parser.parse_args()

    output_dir = Path(args.out)
    plot_pollutants(Path(args.input), output_dir)
    if not args.skip_weather:
        plot_weather(Path(args.weather), output_dir)
    print("\nБүх боломжтой графикууд амжилттай үүслээ!")


if __name__ == "__main__":
    main()
