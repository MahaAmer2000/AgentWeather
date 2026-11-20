
import pandas as pd

from configs.settings import HISTORICAL_DATA_DIR

from src.ingestion.historical import (
    fetch_historical_weather,
    fetch_historical_forecast_features,
    fetch_historical_air_quality,
)

# Two years of historical data
START_DATE = "2024-11-14"
END_DATE = "2026-11-13"

# 1. Fetch historical weather data
weather_df = fetch_historical_weather(
    START_DATE,
    END_DATE
)

# 2. Fetch archived forecast features
forecast_df = fetch_historical_forecast_features(
    START_DATE,
    END_DATE
)

# 3. Fetch historical air-quality data
air_quality_df = fetch_historical_air_quality(
    START_DATE,
    END_DATE
)

# Display individual dataset information
datasets = {
    "WEATHER DATA": weather_df,
    "ARCHIVED FORECAST FEATURES": forecast_df,
    "AIR QUALITY DATA": air_quality_df,
}

for name, df in datasets.items():
    print(f"\n{name}")
    print(df.head())
    print("Shape:", df.shape)
    print("Missing values:\n", df.isnull().sum())
    print("Time range:", df["time"].min(), "to", df["time"].max())

# 4. Check timestamp alignment
print("\nTIMESTAMP ALIGNMENT")

for name, df in datasets.items():
    print(name, ":", df["time"].nunique(), "timestamps")

# 5. Combine all three datasets
# Outer merge preserves timestamps so we can identify missing coverage.
combined_df = pd.merge(
    weather_df,
    forecast_df,
    on="time",
    how="outer"
)

combined_df = pd.merge(
    combined_df,
    air_quality_df,
    on="time",
    how="outer"
)

combined_df = combined_df.sort_values(
    "time"
).reset_index(drop=True)

# 6. Validate combined dataset
print("\nCOMBINED HISTORICAL DATA")
print(combined_df.head())
print("Shape:", combined_df.shape)

print("\nMissing values:")
print(combined_df.isnull().sum())

print("\nTime range:")
print(combined_df["time"].min())
print(combined_df["time"].max())

print("\nDuplicate timestamps:")
print(combined_df["time"].duplicated().sum())

print("\nHourly timestamp gaps:")
print(
    combined_df["time"].diff()
    .value_counts()
    .head()
)

# 7. Validate required variables
required_columns = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "precipitation_probability",
    "uv_index",
    "wind_speed_10m",
    "wind_direction_10m",
    "us_aqi",
]

print("\nREQUIRED VARIABLE CHECK")
for column in required_columns:
    if column in combined_df.columns:
        print(column, ":", "Available")
    else:
        print(column, ":", "MISSING")

# 8. Check important value ranges
for column in ["precipitation_probability", "uv_index", "us_aqi"]:
    if column in combined_df.columns:
        print(f"\n{column} range:")
        print(combined_df[column].min(), combined_df[column].max())

# 9. Overwrite the previous historical dataset
output_path = HISTORICAL_DATA_DIR / "combined_historical.csv"

combined_df.to_csv(output_path, index=False)

# Check missing precipitation probability values
df = pd.read_csv(output_path)

df["time"] = pd.to_datetime(df["time"])

missing = df[df["precipitation_probability"].isna()]

print("Missing rows:")
print(missing[["time", "precipitation_probability"]])

print("\nMissing dates:")
print(missing["time"].dt.date.value_counts().sort_index())

print("\nMissing count:", len(missing))

print("\nHistorical dataset saved successfully:")
print(output_path)
print("Final shape:", combined_df.shape)




