
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


# Test imputation for all eight targets
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

    (
        X_train_imputed,
        X_val_imputed,
        X_test_imputed,
        imputer
    ) = impute_missing_values(
        X_train,
        X_val,
        X_test
    )

    assert "wind_direction_10m" not in X.columns
    assert "wind_direction_sin" in X.columns
    assert "wind_direction_cos" in X.columns
    

    # Verify imputer learned one median per input feature
    assert len(imputer.statistics_) == X_train.shape[1]

    # Verify learned medians contain no missing values
    assert not pd.isna(imputer.statistics_).any()

    print("Imputer statistics verified:", len(imputer.statistics_))

    # Check that no missing values remain
    assert not X_train_imputed.isnull().any().any()
    assert not X_val_imputed.isnull().any().any()
    assert not X_test_imputed.isnull().any().any()

    # Check that feature counts remain unchanged
    assert X_train_imputed.shape[1] == 160
    assert X_val_imputed.shape[1] == 160
    assert X_test_imputed.shape[1] == 160

    # Check that row counts remain unchanged
    assert len(X_train_imputed) == len(X_train)
    assert len(X_val_imputed) == len(X_val)
    assert len(X_test_imputed) == len(X_test)

    # Check that target values remain unchanged
    assert y_train.notna().all()
    assert y_val.notna().all()
    assert y_test.notna().all()

    print(f"\nTarget: {column}")
    print("Training missing values:",
          X_train_imputed.isnull().sum().sum())
    print("Validation missing values:",
          X_val_imputed.isnull().sum().sum())
    print("Testing missing values:",
          X_test_imputed.isnull().sum().sum())

print("Number of model input features:", X.shape[1])
print("\nAll imputation tests passed.")
