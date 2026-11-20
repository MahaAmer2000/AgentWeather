
from src.ingestion.open_meteo import fetch_weather_forecast

weather_df = fetch_weather_forecast()

print("Weather Forecast:")
print(weather_df.head())

print("\nDataset shape:", weather_df.shape)

print("\nMissing values:")
print(weather_df.isnull().sum())

print("\nForecast time range:")
print(weather_df["time"].min())
print(weather_df["time"].max())
