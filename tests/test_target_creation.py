
import pandas as pd

from configs.settings import HISTORICAL_DATA_DIR

from src.processing.feature_engineering import (
    create_time_features,
    encode_wind_direction,
    create_lag_features,
    create_rolling_features,
    create_rate_of_change_features
)

from src.processing.preprocessing import (
    create_target_variables,
    prepare_target_dataset,
    TARGET_COLUMNS
)


# --------------------------------------------------
# 1. Load historical dataset
# --------------------------------------------------

data_path = HISTORICAL_DATA_DIR / "combined_historical.csv"

df = pd.read_csv(data_path)

df["time"] = pd.to_datetime(df["time"])

print("Original shape:", df.shape)


# --------------------------------------------------
# 2. Apply feature engineering
# --------------------------------------------------

df = create_time_features(df)

df = encode_wind_direction(df)

df = create_lag_features(df)

df = create_rolling_features(df)

df = create_rate_of_change_features(df)

print("\nEngineered dataset shape:", df.shape)

assert df.shape == (17520, 162)


# --------------------------------------------------
# 3. Create 24-hour-ahead target variables
# --------------------------------------------------

target_df = create_target_variables(df)

print("Dataset after target creation:", target_df.shape)

assert target_df.shape == (17520, 170)


# --------------------------------------------------
# 4. Verify target alignment
# --------------------------------------------------

for column in TARGET_COLUMNS:

    target_column = f"{column}_target"

    expected_value = df.loc[124, column]

    actual_value = target_df.loc[100, target_column]

    assert actual_value == expected_value, (
        f"Target alignment failed for {column}"
    )

    # Last 24 rows must have missing targets
    assert target_df[target_column].tail(24).isnull().all()

print("\nAll eight target alignments verified.")


# --------------------------------------------------
# 5. Prepare separate datasets for each model
# --------------------------------------------------
print("\nIndividual model datasets:")

for column in TARGET_COLUMNS:

    X, y, timestamps = prepare_target_dataset(target_df, column)

    assert len(X) == len(y) == len(timestamps)
    assert timestamps.is_monotonic_increasing
    assert X.shape[1] == 161

    # No target columns should be present in X
    assert not any(
        name.endswith("_target") for name in X.columns
    )

    # No missing target values
    assert y.notna().all()

    print(
        f"{column}: "
        f"X shape = {X.shape}, "
        f"y shape = {y.shape}, "
        f"missing targets = {y.isnull().sum()}"
    )


print("\nAll target preparation tests passed.")

