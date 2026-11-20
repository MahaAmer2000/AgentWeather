
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
    chronological_split,
    TARGET_COLUMNS
)


# Load historical dataset
data_path = HISTORICAL_DATA_DIR / "combined_historical.csv"

df = pd.read_csv(data_path)
df["time"] = pd.to_datetime(df["time"])

print("Original dataset:", df.shape)


# Apply feature engineering
df = create_time_features(df)
df = encode_wind_direction(df)
df = create_lag_features(df)
df = create_rolling_features(df)
df = create_rate_of_change_features(df)


# Create target variables
target_df = create_target_variables(df)

print("Engineered dataset:", df.shape)
print("Dataset with targets:", target_df.shape)


# Test chronological split for every target
for column in TARGET_COLUMNS:

    X, y, timestamps = prepare_target_dataset(
        target_df, column
    )

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    ) = chronological_split(
        X, y, timestamps
    )

    # Check matching feature and target lengths
    assert len(X_train) == len(y_train)
    assert len(X_val) == len(y_val)
    assert len(X_test) == len(y_test)

    # Check chronological ordering
    assert timestamps.is_monotonic_increasing

    # Check that all subsets contain data
    assert len(X_train) > 0
    assert len(X_val) > 0
    assert len(X_test) > 0

    # Check feature counts
    assert X_train.shape[1] == 161
    assert X_val.shape[1] == 161
    assert X_test.shape[1] == 161

    
    # Calculate the same split boundaries
    total_samples = len(timestamps)

    train_end = int(total_samples * 0.70)
    validation_end = int(total_samples * 0.85)

    # Get timestamps belonging to each subset
    train_times = timestamps.iloc[:train_end]

    val_times = timestamps.iloc[
        train_end + 24:validation_end
    ]

    test_times = timestamps.iloc[
        validation_end + 24:
    ]

    # Calculate the actual future target timestamps
    train_target_times = (
        train_times + pd.Timedelta(hours=24)
    )

    val_target_times = (
        val_times + pd.Timedelta(hours=24)
    )

    # Training targets must occur before validation features
    assert train_target_times.max() < val_times.min()

    # Validation targets must occur before testing features
    assert val_target_times.max() < test_times.min()

    print("24-hour target separation: verified")


    print(f"\nTarget: {column}")
    print("Training:", X_train.shape, y_train.shape)
    print("Validation:", X_val.shape, y_val.shape)
    print("Testing:", X_test.shape, y_test.shape)

print("\nAll chronological split tests passed.")
