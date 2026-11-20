
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
    impute_missing_values,
    scale_features,
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


# Test scaling for all eight targets
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
    ) = chronological_split(X, y, timestamps)

    # First impute missing values
    (
        X_train_imputed,
        X_val_imputed,
        X_test_imputed,
        imputer
    ) = impute_missing_values(
        X_train, X_val, X_test
    )

    # Then scale input features
    (
        X_train_scaled,
        X_val_scaled,
        X_test_scaled,
        scaler
    ) = scale_features(
        X_train_imputed,
        X_val_imputed,
        X_test_imputed
    )

    # Check that no missing values remain
    assert not X_train_scaled.isnull().any().any()
    assert not X_val_scaled.isnull().any().any()
    assert not X_test_scaled.isnull().any().any()

    # Check feature counts
    assert X_train_scaled.shape[1] == 161
    assert X_val_scaled.shape[1] == 161
    assert X_test_scaled.shape[1] == 161

    # Check row counts remain unchanged
    assert len(X_train_scaled) == len(X_train)
    assert len(X_val_scaled) == len(X_val)
    assert len(X_test_scaled) == len(X_test)

    # Verify training features have approximately zero mean
    assert abs(X_train_scaled.mean().mean()) < 1e-6

    # Verify training features have approximately unit variance
    assert abs(X_train_scaled.std(ddof=0).mean() - 1) < 1e-6

    # Verify scaler learned statistics for all features
    assert len(scaler.mean_) == 161
    assert len(scaler.scale_) == 161

    # Verify target values remain unchanged
    assert y_train.notna().all()
    assert y_val.notna().all()
    assert y_test.notna().all()

    print(f"\nTarget: {column}")
    print("Training mean:", X_train_scaled.mean().mean())
    print("Training standard deviation:",
          X_train_scaled.std(ddof=0).mean())
    print("Scaler statistics:", len(scaler.mean_))


print("\nAll scaling tests passed.")
