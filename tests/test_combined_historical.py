
from configs.settings import HISTORICAL_DATA_DIR

from src.ingestion.historical import (
    fetch_combined_historical_data,
)

START_DATE = "2026-11-01"
END_DATE = "2026-11-07"

# Fetch and combine all historical data
combined_df = fetch_combined_historical_data(
    START_DATE,
    END_DATE
)

# Save combined historical dataset
output_file = HISTORICAL_DATA_DIR / "combined_historical.csv"

combined_df.to_csv(
    output_file,
    index=False
)

print(f"\nDataset saved to: {output_file}")

print("\nFirst five records:")
print(combined_df.head())

print("\nDataset shape:")
print(combined_df.shape)

print("\nColumns:")
print(combined_df.columns.tolist())

print("\nMissing values:")
print(combined_df.isnull().sum())

print("\nDuplicate timestamps:")
print(combined_df["time"].duplicated().sum())

print("\nTime range:")
print(combined_df["time"].min())
print(combined_df["time"].max())
