
from src.ingestion.open_meteo import fetch_combined_forecast

combined_df = fetch_combined_forecast()

print("Combined Weather and Air Quality Forecast:")
print(combined_df.head())

print("\nDataset shape:", combined_df.shape)

print("\nColumn names:")
print(combined_df.columns.tolist())

print("\nMissing values:")
print(combined_df.isnull().sum())

print("\nDuplicate timestamps:")
print(combined_df["time"].duplicated().sum())
