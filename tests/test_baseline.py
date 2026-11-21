
import pandas as pd

from src.processing.preprocessing import prepare_training_data

from src.models.baseline import (
    naive_baseline,
    wind_direction_baseline
)


# --------------------------------------------------
# 1. Load historical data
# --------------------------------------------------

df = pd.read_csv(
    "data/historical/combined_historical.csv"
)


# --------------------------------------------------
# 2. Prepare datasets
# --------------------------------------------------

prepared_datasets = prepare_training_data(df)


# --------------------------------------------------
# 3. Evaluate seven numerical targets
# --------------------------------------------------

TARGET_COLUMNS = [
    "us_aqi",
    "temperature_2m",
    "precipitation",
    "precipitation_probability",
    "relative_humidity_2m",
    "uv_index",
    "wind_speed_10m"
]

results = []

for target in TARGET_COLUMNS:

    print("\n" + "=" * 60)
    print(f"Evaluating baseline for: {target}")
    print("=" * 60)

    target_data = prepared_datasets[target]

    baseline_metrics = naive_baseline(
        target_data["X_val_unscaled"],
        target_data["y_val"],
        target
    )

    results.append({
        "Target": target,
        "MAE": baseline_metrics["MAE"],
        "RMSE": baseline_metrics["RMSE"],
        "R2": baseline_metrics["R2"]
    })

    print(baseline_metrics)


# --------------------------------------------------
# 4. Evaluate wind direction
# --------------------------------------------------

raw_wind = (
    df.assign(time=pd.to_datetime(df["time"]))
      .set_index("time")["wind_direction_10m"]
)

wind_data = prepared_datasets["wind_direction_10m"]

val_timestamps = pd.to_datetime(
    wind_data["val_timestamps"]
)

# Current wind direction = prediction
y_pred = raw_wind.reindex(
    val_timestamps
).to_numpy()

# Actual wind direction 24 hours ahead
y_true = raw_wind.reindex(
    val_timestamps + pd.Timedelta(hours=24)
).to_numpy()

# Validate timestamp alignment
assert not pd.isna(y_pred).any()
assert not pd.isna(y_true).any()
assert len(y_pred) == len(y_true)

wind_metrics = wind_direction_baseline(
    y_pred,
    y_true
)

results.append({
    "Target": "wind_direction_10m",
    "MAE": wind_metrics["MAE"],
    "RMSE": wind_metrics["RMSE"],
    "R2": None
})

print("\nWind Direction Baseline")
print("-----------------------")
print(wind_metrics)


# --------------------------------------------------
# 5. Display final validation results
# --------------------------------------------------

results_df = pd.DataFrame(results)

print("\n" + "=" * 75)
print("FINAL BASELINE VALIDATION RESULTS")
print("=" * 75)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# --------------------------------------------------
# 6. Save validation results
# --------------------------------------------------

results_df.to_csv(
    "results/baseline_validation.csv",
    index=False
)

print("\nBaseline validation results saved successfully.")
