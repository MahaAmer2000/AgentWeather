
from src.ingestion.open_meteo import fetch_air_quality_forecast

air_quality_df = fetch_air_quality_forecast()

print("Air Quality Forecast:")
print(air_quality_df.head())

print("\nDataset shape:", air_quality_df.shape)

print("\nMissing values:")
print(air_quality_df.isnull().sum())

print("\nForecast time range:")
print(air_quality_df["time"].min())
print(air_quality_df["time"].max())
