
from pathlib import Path
import pandas as pd
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent

print("\n========== PROJECT CHECK ==========")

# 1. Results хавтас шалгах
results_dir = BASE_DIR / "results"

print("\n--- RESULTS FILES ---")

if results_dir.exists():
    for file in results_dir.iterdir():
        print(file.name)
else:
    print("results хавтас олдсонгүй.")

# 2. Models хавтас шалгах
models_dir = BASE_DIR / "models"

print("\n--- MODEL FILES ---")

if models_dir.exists():
    for file in models_dir.glob("*.joblib"):
        print(f"\nModel: {file.name}")

        try:
            model = joblib.load(file)

            features = getattr(model, "feature_names_in_", None)

            if features is not None:
                print("Features:")
                print(list(features))
            else:
                print("Feature names шууд олдсонгүй.")

        except Exception as e:
            print("Model унших алдаа:", e)
else:
    print("models хавтас олдсонгүй.")

# 3. Results CSV баганууд шалгах
print("\n--- CSV COLUMNS ---")

if results_dir.exists():
    for file in results_dir.glob("*.csv"):
        try:
            df = pd.read_csv(file)

            print(f"\nFile: {file.name}")
            print("Columns:", list(df.columns))
            print("Rows:", len(df))

        except Exception as e:
            print(file.name, "унших алдаа:", e)

print("\n========== CHECK COMPLETED ==========")
