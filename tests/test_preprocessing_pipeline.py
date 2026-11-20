import pandas as pd

from configs.settings import HISTORICAL_DATA_DIR
from src.processing.preprocessing import encode_wind_direction_target
import numpy as np

from src.processing.preprocessing import (
    prepare_training_data,
    TARGET_COLUMNS
)

from src.processing.preprocessing import (
    prepare_training_data,
    TARGET_COLUMNS,
    encode_wind_direction_target,
    decode_wind_direction_prediction,
    calculate_wind_direction_errors,
    validate_hourly_data
)

# Load historical dataset
data_path = HISTORICAL_DATA_DIR / "combined_historical.csv"

df = pd.read_csv(data_path)
df["time"] = pd.to_datetime(df["time"])

print("Original dataset:", df.shape)


# Run unified preprocessing pipeline
prepared_datasets = prepare_training_data(df)

print("\nUnified preprocessing completed.")
print("Number of prepared targets:", len(prepared_datasets))

print("\nValidating prepared datasets...")

for column in TARGET_COLUMNS:

    data = prepared_datasets[column]

    # Check that all eight datasets have been created
    assert column in prepared_datasets

    # Check input feature dimensions
    assert data["X_train"].shape[1] == 160
    assert data["X_val"].shape[1] == 160
    assert data["X_test"].shape[1] == 160

    # Check that input features contain no missing values
    assert not data["X_train"].isnull().any().any()
    assert not data["X_val"].isnull().any().any()
    assert not data["X_test"].isnull().any().any()

    # Check that target values contain no missing values
    if column == "wind_direction_10m":

       assert data["y_train"].shape[1] == 2
       assert data["y_val"].shape[1] == 2
       assert data["y_test"].shape[1] == 2

       assert not data["y_train"].isnull().any().any()
       assert not data["y_val"].isnull().any().any()
       assert not data["y_test"].isnull().any().any()

    else:

       assert not data["y_train"].isnull().any()
       assert not data["y_val"].isnull().any()
       assert not data["y_test"].isnull().any()

    # Check that feature and target row counts match
    assert len(data["X_train"]) == len(data["y_train"])
    assert len(data["X_val"]) == len(data["y_val"])
    assert len(data["X_test"]) == len(data["y_test"])

    # Check timestamp alignment
    assert len(data["train_timestamps"]) == len(data["X_train"])
    assert len(data["val_timestamps"]) == len(data["X_val"])
    assert len(data["test_timestamps"]) == len(data["X_test"])

    # Check preprocessing objects
    assert len(data["imputer"].statistics_) == 160
    assert len(data["scaler"].mean_) == 160

    print(
        f"{column}: "
        f"Train {data['X_train'].shape}, "
        f"Validation {data['X_val'].shape}, "
        f"Test {data['X_test'].shape} — Passed"
    )

print("\nAll unified preprocessing pipeline tests passed.")

print("\nTesting wind direction target encoding...")

# Example wind directions
wind_directions = pd.Series([0, 90, 180, 270, 359, 1])

# Encode directions
encoded = encode_wind_direction_target(wind_directions)

# Reconstruct angles from sine and cosine
predicted_angles = (
    np.degrees(
        np.arctan2(
            encoded["wind_direction_sin_target"],
            encoded["wind_direction_cos_target"]
        )
    ) + 360
) % 360

print("Original directions:", wind_directions.tolist())
print("Reconstructed directions:", predicted_angles.round(2).tolist())

# Verify reconstructed angles
assert np.allclose(
    wind_directions,
    predicted_angles,
    atol=1e-6
)

print("Wind direction encoding test passed.")

print("\nTesting wind direction decoding...")

# Original wind directions
wind_directions = pd.Series([0, 90, 180, 270, 359, 1])

# Encode into sine and cosine
encoded = encode_wind_direction_target(wind_directions)

# Convert encoded values into a NumPy array
predictions = encoded.to_numpy()

# Decode back into degrees
decoded = decode_wind_direction_prediction(predictions)

print("Original:", wind_directions.tolist())
print("Decoded:", decoded.round(2).tolist())

# Verify that reconstructed angles match original angles
assert np.allclose(
    wind_directions,
    decoded,
    atol=1e-6
)

# Verify that all angles are within the valid range
assert np.all((decoded >= 0) & (decoded < 360))

print("Wind direction decoding test passed.")

print("\nTesting circular wind direction errors...")

# Actual and predicted directions
y_true = np.array([359, 1, 90, 180])
y_pred = np.array([1, 359, 100, 170])

# Calculate circular errors
errors = calculate_wind_direction_errors(y_true, y_pred)

print("Actual:", y_true)
print("Predicted:", y_pred)
print("Circular errors:", errors)

# Expected shortest angular differences
expected_errors = np.array([2, -2, 10, -10])

assert np.allclose(errors, expected_errors)

print("Circular wind direction error test passed.")

print("\nTesting hourly data validation...")

# Test with the actual historical dataset
validated_df = validate_hourly_data(df)

assert len(validated_df) == len(df)
assert validated_df["time"].is_monotonic_increasing
assert not validated_df["time"].duplicated().any()

print("Historical dataset validation passed.")
print("Records:", len(validated_df))

# Test with deliberately missing hourly record
# Remove one row while keeping all required columns
invalid_df = df.drop(index=2).reset_index(drop=True)

try:
    validate_hourly_data(invalid_df)
    raise AssertionError(
        "Expected missing-hour validation to fail."
    )

except ValueError as error:
    assert "continuous hourly records" in str(error)
    print("Missing-hour detection passed.")

print("All hourly data validation tests passed.")


print("\nTesting duplicate timestamp detection...")

# Duplicate a complete row, including its timestamp
duplicate_df = pd.concat(
    [df, df.iloc[[0]]],
    ignore_index=True
)

try:
    validate_hourly_data(duplicate_df)
    raise AssertionError(
        "Expected duplicate timestamp validation to fail."
    )

except ValueError as error:
    assert "Duplicate timestamps" in str(error)
    print("Duplicate timestamp detection passed.")

print("Duplicate timestamp validation test passed.")

print("\nTesting missing required column detection...")

missing_column_df = df.drop(columns=["pm2_5"])

try:
    validate_hourly_data(missing_column_df)
    raise AssertionError(
        "Expected missing-column validation to fail."
    )

except ValueError as error:
    assert "Missing required columns" in str(error)
    assert "pm2_5" in str(error)

    print("Missing required column detection passed.")

print("Missing required column validation test passed.")