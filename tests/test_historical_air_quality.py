
from src.ingestion.historical import fetch_historical_air_quality

START_DATE = "2026-11-01"
END_DATE = "2026-11-07"

air_quality_df = fetch_historical_air_quality(
    START_DATE,
    END_DATE
)

print("\nHISTORICAL AIR QUALITY DATA")
print(air_quality_df.head())

print("\nShape:")
print(air_quality_df.shape)

print("\nMissing values:")
print(air_quality_df.isnull().sum())

print("\nDuplicate timestamps:")
print(air_quality_df["time"].duplicated().sum())

print("\nTime range:")
print(air_quality_df["time"].min())
print(air_quality_df["time"].max())

print("\nUS AQI range:")
print(
    air_quality_df["us_aqi"].min(),
    air_quality_df["us_aqi"].max()
)
