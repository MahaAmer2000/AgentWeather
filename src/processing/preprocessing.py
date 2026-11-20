
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import numpy as np

from src.processing.feature_engineering import (
    create_time_features,
    encode_wind_direction,
    create_lag_features,
    create_rolling_features,
    create_rate_of_change_features
)


# Eight variables our models will predict
TARGET_COLUMNS = [
    "us_aqi",
    "temperature_2m",
    "precipitation",
    "precipitation_probability",
    "relative_humidity_2m",
    "uv_index",
    "wind_speed_10m",
    "wind_direction_10m"
]


def create_target_variables(df):
    """
    Create target variables representing values 24 hours ahead.
    """

    df = df.copy()

    # Ensure chronological order
    df = df.sort_values("time").reset_index(drop=True)

    # Create one future target for each variable
    for column in TARGET_COLUMNS:
        df[f"{column}_target"] = df[column].shift(-24)

    return df



def prepare_target_dataset(df, target_column):
    if target_column not in TARGET_COLUMNS:
        raise ValueError(f"Unknown target: {target_column}")

    df = df.copy()
    df = df.sort_values("time").reset_index(drop=True)

    # Remove the first 24 rows because they lack sufficient history.
    df = df.iloc[24:].copy()

    # Remove rows where this model's target is missing.
    df = df.dropna(subset=[f"{target_column}_target"])

    # Preserve timestamps separately for chronological splitting.
    timestamps = df["time"].copy()

    # Remove time and all target columns from model inputs.
    all_target_columns = [
        f"{column}_target" for column in TARGET_COLUMNS
    ]

    X = df.drop(columns=["time", "wind_direction_10m"] + all_target_columns)
    y = df[f"{target_column}_target"]

    return X, y, timestamps



def chronological_split(
    X,
    y,
    timestamps,
    train_ratio=0.70,
    validation_ratio=0.15,
    gap=24
):
    # Sort all data by timestamp
    sorted_indices = timestamps.argsort()

    X = X.iloc[sorted_indices].reset_index(drop=True)
    y = y.iloc[sorted_indices].reset_index(drop=True)
    timestamps = timestamps.iloc[sorted_indices].reset_index(drop=True)

    # Calculate split boundaries
    total_samples = len(X)

    train_end = int(total_samples * train_ratio)
    validation_end = int(
        total_samples * (train_ratio + validation_ratio)
    )

    # Training data
    X_train = X.iloc[:train_end]
    y_train = y.iloc[:train_end]

    # Validation data (skip the 24-hour gap)
    X_val = X.iloc[train_end + gap:validation_end]
    y_val = y.iloc[train_end + gap:validation_end]

    # Testing data (skip another 24-hour gap)
    X_test = X.iloc[validation_end + gap:]
    y_test = y.iloc[validation_end + gap:]

    return X_train, X_val, X_test, y_train, y_val, y_test


def impute_missing_values(X_train, X_val, X_test):

    # Create median imputer
    imputer = SimpleImputer(strategy="median")

    # Learn median values from training data only
    X_train_imputed = imputer.fit_transform(X_train)

    # Apply the same learned medians to validation and test data
    X_val_imputed = imputer.transform(X_val)
    X_test_imputed = imputer.transform(X_test)

    # Convert NumPy arrays back to DataFrames
    X_train_imputed = pd.DataFrame(
        X_train_imputed,
        columns=X_train.columns,
        index=X_train.index
    )

    X_val_imputed = pd.DataFrame(
        X_val_imputed,
        columns=X_val.columns,
        index=X_val.index
    )

    X_test_imputed = pd.DataFrame(
        X_test_imputed,
        columns=X_test.columns,
        index=X_test.index
    )

    return (
        X_train_imputed,
        X_val_imputed,
        X_test_imputed,
        imputer
    )


def scale_features(X_train, X_val, X_test):

    # Create the scaler
    scaler = StandardScaler()

    # Learn mean and standard deviation from training data only
    X_train_scaled = scaler.fit_transform(X_train)

    # Apply the same scaling parameters to validation and test data
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Convert NumPy arrays back into DataFrames
    X_train_scaled = pd.DataFrame(
        X_train_scaled,
        columns=X_train.columns,
        index=X_train.index
    )

    X_val_scaled = pd.DataFrame(
        X_val_scaled,
        columns=X_val.columns,
        index=X_val.index
    )

    X_test_scaled = pd.DataFrame(
        X_test_scaled,
        columns=X_test.columns,
        index=X_test.index
    )

    return (
        X_train_scaled,
        X_val_scaled,
        X_test_scaled,
        scaler
    )

def encode_wind_direction_target(y):

    # Convert wind direction from degrees to radians
    radians = np.radians(y)

    # Encode direction into sine and cosine components
    y_sin = np.sin(radians)
    y_cos = np.cos(radians)

    # Return both target components
    return pd.DataFrame({
        "wind_direction_sin_target": y_sin,
        "wind_direction_cos_target": y_cos
    })

def decode_wind_direction_prediction(y_pred):

    # Extract sine and cosine predictions
    y_sin = y_pred[:, 0]
    y_cos = y_pred[:, 1]

    # Convert predictions back to angles in degrees
    angles = np.degrees(
        np.arctan2(y_sin, y_cos)
    )

    # Ensure angles are between 0 and 360 degrees
    angles = (angles + 360) % 360

    return angles

def calculate_wind_direction_errors(y_true, y_pred):

    # Calculate the shortest angular difference
    errors = (
        (y_pred - y_true + 180) % 360
    ) - 180

    return errors

def validate_hourly_data(df):

    # Check that timestamp column exists
    if "time" not in df.columns:
        raise ValueError("Missing time column.")

    required_columns = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "precipitation_probability",
    "wind_speed_10m",
    "wind_direction_10m",
    "uv_index",
    "us_aqi",
    "pm2_5",
    "pm10",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone"
    ]

    missing_columns = [
      column for column in required_columns
      if column not in df.columns
    ]

    if missing_columns:
     raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

    # Convert timestamps to datetime
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"])

    # Check for missing timestamps
    if df["time"].isna().any():
        raise ValueError("Invalid or missing timestamps found.")

    # Check for duplicate timestamps
    if df["time"].duplicated().any():
        raise ValueError("Duplicate timestamps found.")

    # Sort chronologically
    df = df.sort_values("time").reset_index(drop=True)

    # Calculate time differences
    time_differences = df["time"].diff().dropna()

    # Verify hourly frequency
    if not (time_differences == pd.Timedelta(hours=1)).all():
        raise ValueError(
            "Dataset does not contain continuous hourly records."
        )

    return df



def prepare_training_data(
    df,
    train_ratio=0.70,
    validation_ratio=0.15,
    gap=24
):

    # Validate and sort hourly input data
    df = validate_hourly_data(df)

    # Apply feature engineering
    df = create_time_features(df)
    df = encode_wind_direction(df)
    df = create_lag_features(df)
    df = create_rolling_features(df)
    df = create_rate_of_change_features(df)

    # Create 24-hour-ahead target variables
    df = create_target_variables(df)

    # Store prepared datasets for all eight targets
    prepared_datasets = {}

    # Process each target independently
    for column in TARGET_COLUMNS:

        # Prepare input features and target values
        X, y, timestamps = prepare_target_dataset(df, column)

        # Transform wind direction into sine and cosine targets
        if column == "wind_direction_10m":
           y = encode_wind_direction_target(y)

        # Ensure timestamps are in chronological order
        sorted_indices = timestamps.argsort()

        timestamps = timestamps.iloc[sorted_indices].reset_index(drop=True)

        # Split data chronologically
        (
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test
        ) = chronological_split(
            X,
            y,
            timestamps,
            train_ratio=train_ratio,
            validation_ratio=validation_ratio,
            gap=gap
        )

        # Calculate matching timestamp boundaries
        total_samples = len(timestamps)

        train_end = int(total_samples * train_ratio)

        validation_end = int(
            total_samples * (train_ratio + validation_ratio)
        )

        train_timestamps = timestamps.iloc[:train_end]
        val_timestamps = timestamps.iloc[
            train_end + gap:validation_end
        ]
        test_timestamps = timestamps.iloc[
            validation_end + gap:
        ]

        # Impute missing input features
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

        # Scale input features
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

        # Store all prepared data and preprocessing objects
        prepared_datasets[column] = {

            # Scaled input features
            "X_train": X_train_scaled,
            "X_val": X_val_scaled,
            "X_test": X_test_scaled,

            # Imputed but unscaled input features
            "X_train_unscaled": X_train_imputed,
            "X_val_unscaled": X_val_imputed,
            "X_test_unscaled": X_test_imputed,

            # Target values
            "y_train": y_train,
            "y_val": y_val,
            "y_test": y_test,

            # Corresponding feature timestamps
            "train_timestamps": train_timestamps,
            "val_timestamps": val_timestamps,
            "test_timestamps": test_timestamps,

            # Fitted preprocessing objects
            "imputer": imputer,
            "scaler": scaler
        }

    return prepared_datasets
