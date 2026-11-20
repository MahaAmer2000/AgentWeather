
from src.processing.preprocessing import prepare_training_data
import pandas as pd

from src.models.baseline import (
    naive_baseline,
    wind_direction_baseline
)

# Load historical data
df = pd.read_csv("data/historical/combined_historical.csv")

# Prepare datasets
prepared_datasets = prepare_training_data(df)

# Evaluate naive baseline for seven numerical targets
baseline_results = {}

for target in [
    "us_aqi",
    "temperature_2m",
    "precipitation",
    "precipitation_probability",
    "relative_humidity_2m",
    "uv_index",
    "wind_speed_10m"
]:

    target_data = prepared_datasets[target]

    results = naive_baseline(
        target_data["X_test_unscaled"],
        target_data["y_test"],
        target
    )

    baseline_results[target] = results

    print(f"\n{target}:")
    print(results)


# -----------------------------------------
# Evaluate wind direction baseline
# -----------------------------------------

# Prepare original wind direction data
raw_wind = (
    df.assign(time=pd.to_datetime(df["time"]))
      .set_index("time")["wind_direction_10m"]
)

# Get wind-direction test timestamps
wind_data = prepared_datasets["wind_direction_10m"]

test_timestamps = pd.to_datetime(
    wind_data["test_timestamps"]
)

# Current direction = baseline prediction
y_pred = raw_wind.reindex(test_timestamps).to_numpy()

# Direction 24 hours later = actual target
y_true = raw_wind.reindex(
    test_timestamps + pd.Timedelta(hours=24)
).to_numpy()

# Validate timestamp alignment
assert not pd.isna(y_pred).any()
assert not pd.isna(y_true).any()
assert len(y_pred) == len(y_true)

# Evaluate circular errors
wind_results = wind_direction_baseline(y_pred, y_true)

print("\nwind_direction_10m:")
print(wind_results)




