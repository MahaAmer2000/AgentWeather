
import numpy as np
import pandas as pd

from src.processing.preprocessing import (
    prepare_training_data,
    encode_wind_direction_target,
    decode_wind_direction_prediction,
    calculate_wind_direction_errors
)

from src.models.sarimax_model import (
    prepare_sarimax_data,
    train_sarimax_model,
    rolling_sarimax_predictions,
    evaluate_sarimax_model
)


# --------------------------------------------------
# 1. Load historical data
# --------------------------------------------------

df = pd.read_csv(
    "data/historical/combined_historical.csv"
)


# --------------------------------------------------
# 2. Run existing preprocessing pipeline
# --------------------------------------------------

print("\nRunning existing preprocessing pipeline...")

prepared_datasets = prepare_training_data(df)

print("\nPreprocessing completed successfully.")


# --------------------------------------------------
# 3. Define seven scalar targets
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


# --------------------------------------------------
# 4. Train and evaluate SARIMAX for seven targets
# --------------------------------------------------

all_results = {}

for target_column in TARGET_COLUMNS:

    print("\n" + "=" * 60)
    print(f"Training SARIMAX for: {target_column}")
    print("=" * 60)

    # Get prepared dataset
    target_data = prepared_datasets[target_column]

    train_timestamps = pd.DatetimeIndex(
        pd.to_datetime(target_data["train_timestamps"])
    )

    val_timestamps = pd.DatetimeIndex(
        pd.to_datetime(target_data["val_timestamps"])
    )

    # Prepare original hourly target series
    target_series = prepare_sarimax_data(
        df,
        target_column
    )

    # Extract training period
    train = target_series.loc[
        train_timestamps.min():train_timestamps.max()
    ].asfreq("h")

    # Validate training period
    assert len(train) > 0
    assert train.index.is_monotonic_increasing
    assert train.index.is_unique

    print("\nTraining observations:", len(train))
    print("Validation forecast origins:", len(val_timestamps))

    print(
        "Training period:",
        train.index.min(),
        "to",
        train.index.max()
    )

    print(
        "Validation period:",
        val_timestamps.min(),
        "to",
        val_timestamps.max()
    )

    # Train SARIMAX
    print("\nTraining SARIMAX model...")

    sarimax_model = train_sarimax_model(train)

    print("SARIMAX model trained successfully.")

    # Generate rolling 24-hour-ahead predictions
    print("\nGenerating validation predictions...")

    sarimax_predictions = rolling_sarimax_predictions(
        sarimax_model,
        target_series,
        val_timestamps
    )

    # Retrieve actual values 24 hours ahead
    val_target_timestamps = (
        val_timestamps + pd.Timedelta(hours=24)
    )

    y_val_true = target_series.reindex(
        val_target_timestamps
    )

    # Validate alignment
    assert len(sarimax_predictions) == len(y_val_true)
    assert not y_val_true.isna().any()

    # Evaluate model
    results = evaluate_sarimax_model(
        y_val_true.to_numpy(),
        sarimax_predictions
    )

    all_results[target_column] = results

    print(f"\n{target_column} SARIMAX Validation Results:")
    print(results)


# --------------------------------------------------
# 5. Train and evaluate SARIMAX for wind direction
# --------------------------------------------------

print("\n" + "=" * 60)
print("Training SARIMAX for: wind_direction_10m")
print("=" * 60)

wind_data = prepared_datasets["wind_direction_10m"]

train_timestamps = pd.DatetimeIndex(
    pd.to_datetime(wind_data["train_timestamps"])
)

val_timestamps = pd.DatetimeIndex(
    pd.to_datetime(wind_data["val_timestamps"])
)

# Prepare original wind direction series
wind_series = prepare_sarimax_data(
    df,
    "wind_direction_10m"
)

# Encode wind direction using existing preprocessing function
encoded_wind = encode_wind_direction_target(
    wind_series
)

sin_series = pd.Series(
    encoded_wind["wind_direction_sin_target"].to_numpy(),
    index=wind_series.index
)

cos_series = pd.Series(
    encoded_wind["wind_direction_cos_target"].to_numpy(),
    index=wind_series.index
)

# Extract training periods
train_sin = sin_series.loc[
    train_timestamps.min():train_timestamps.max()
].asfreq("h")

train_cos = cos_series.loc[
    train_timestamps.min():train_timestamps.max()
].asfreq("h")

print("\nTraining observations:", len(train_sin))
print("Validation forecast origins:", len(val_timestamps))

# Train sine SARIMAX
print("\nTraining sine SARIMAX model...")

sin_model = train_sarimax_model(train_sin)

print("Sine model trained successfully.")

# Train cosine SARIMAX
print("\nTraining cosine SARIMAX model...")

cos_model = train_sarimax_model(train_cos)

print("Cosine model trained successfully.")

# Generate rolling validation predictions
print("\nGenerating sine predictions...")

sin_predictions = rolling_sarimax_predictions(
    sin_model,
    sin_series,
    val_timestamps
)

print("\nGenerating cosine predictions...")

cos_predictions = rolling_sarimax_predictions(
    cos_model,
    cos_series,
    val_timestamps
)

# Combine sine and cosine predictions
wind_component_predictions = np.column_stack([
    sin_predictions,
    cos_predictions
])

# Decode predictions into degrees
wind_predictions = decode_wind_direction_prediction(
    wind_component_predictions
)

# Retrieve actual wind directions 24 hours ahead
val_target_timestamps = (
    val_timestamps + pd.Timedelta(hours=24)
)

y_val_true = wind_series.reindex(
    val_target_timestamps
).to_numpy()

# Validate predictions and actual values
assert len(wind_predictions) == len(y_val_true)
assert not np.isnan(y_val_true).any()

# Calculate circular errors
wind_errors = calculate_wind_direction_errors(
    y_val_true,
    wind_predictions
)

# Calculate wind-direction metrics
wind_results = {
    "MAE": np.mean(np.abs(wind_errors)),
    "RMSE": np.sqrt(np.mean(wind_errors ** 2))
}

all_results["wind_direction_10m"] = wind_results

print("\nWind Direction SARIMAX Validation Results:")
print(wind_results)


# --------------------------------------------------
# 6. Display consolidated results
# --------------------------------------------------

print("\n" + "=" * 60)
print("FINAL SARIMAX VALIDATION RESULTS")
print("=" * 60)

results_df = pd.DataFrame(all_results).T

print(results_df)

print("\nSARIMAX evaluation completed for all eight targets.")
