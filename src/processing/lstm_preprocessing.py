
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from src.processing.feature_engineering import (
    create_time_features,
    encode_wind_direction,
    create_lag_features,
    create_rolling_features,
    create_rate_of_change_features
)

from src.processing.preprocessing import (
    TARGET_COLUMNS,
    validate_hourly_data,
    create_target_variables,
    encode_wind_direction_target
)


def create_lstm_sequences(
    X,
    y,
    timestamps,
    target_positions,
    sequence_length=24
):
    """
    Create hourly LSTM sequences for selected target positions.

    Missing targets are skipped without removing their
    corresponding feature observations from the timeline.
    """

    X_sequences = []
    y_sequences = []
    sequence_timestamps = []

    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    timestamps = pd.Series(
        pd.to_datetime(timestamps)
    ).reset_index(drop=True)

    if not (len(X) == len(y) == len(timestamps)):
        raise ValueError(
            "X, y, and timestamps must have equal lengths."
        )

    for i in target_positions:

        start = i - sequence_length + 1

        # A complete sequence must be available.
        if start < 0:
            continue

        # Verify consecutive hourly observations.
        current_timestamps = timestamps.iloc[start:i + 1]

        differences = current_timestamps.diff().iloc[1:]

        if not (
            differences == pd.Timedelta(hours=1)
        ).all():
            continue

        # Retrieve the target at the forecast origin.
        target = np.asarray(
            y.iloc[i],
            dtype=np.float32
        )

        # Skip missing targets, but retain their features.
        if np.isnan(target).any():
            continue

        X_sequence = X.iloc[start:i + 1].values

        X_sequences.append(X_sequence)
        y_sequences.append(target)
        sequence_timestamps.append(timestamps.iloc[i])

    return (
        np.asarray(X_sequences, dtype=np.float32),
        np.asarray(y_sequences, dtype=np.float32),
        pd.Series(sequence_timestamps)
    )


def prepare_lstm_data(
    df,
    target_column,
    train_ratio=0.70,
    validation_ratio=0.15,
    gap=24,
    sequence_length=24
):
    """
    Prepare training, validation, and test sequences
    for one LSTM forecasting target.

    Preserves the complete hourly feature timeline.
    Uses the same valid-target split boundaries as the
    existing preprocessing pipeline.

    Input features and scalar targets are standardized
    using training data only.

    Original target values are preserved for evaluation.
    Wind direction sine/cosine targets are not scaled.
    """

    if target_column not in TARGET_COLUMNS:
        raise ValueError(
            f"Unknown target: {target_column}"
        )

    # --------------------------------------------------
    # 1. Validate and sort historical data
    # --------------------------------------------------

    df = validate_hourly_data(df)

    # --------------------------------------------------
    # 2. Apply existing feature engineering
    # --------------------------------------------------

    df = create_time_features(df)
    df = encode_wind_direction(df)
    df = create_lag_features(df)
    df = create_rolling_features(df)
    df = create_rate_of_change_features(df)

    # --------------------------------------------------
    # 3. Create 24-hour-ahead targets
    # --------------------------------------------------

    df = create_target_variables(df)

    # Remove the initial 24 rows, as in existing pipeline.
    df = df.iloc[24:].reset_index(drop=True)

    timestamps = df["time"].copy()

    # Preserve target values, including missing values.
    y = df[f"{target_column}_target"].copy()

    # Encode wind direction into sine and cosine.
    if target_column == "wind_direction_10m":
        y = encode_wind_direction_target(y)

    # --------------------------------------------------
    # 4. Prepare feature matrix
    # --------------------------------------------------

    all_target_columns = [
        f"{column}_target"
        for column in TARGET_COLUMNS
    ]

    X = df.drop(
        columns=[
            "time",
            "wind_direction_10m"
        ] + all_target_columns
    )

    # --------------------------------------------------
    # 5. Identify valid target positions
    # --------------------------------------------------

    if isinstance(y, pd.DataFrame):
        valid_target_mask = y.notna().all(axis=1)
    else:
        valid_target_mask = y.notna()

    valid_positions = np.flatnonzero(
        valid_target_mask.to_numpy()
    )

    total_valid_samples = len(valid_positions)

    train_end = int(
        total_valid_samples * train_ratio
    )

    validation_end = int(
        total_valid_samples *
        (train_ratio + validation_ratio)
    )

    # Use the same split logic as existing pipeline.
    train_positions = valid_positions[:train_end]

    val_positions = valid_positions[
        train_end + gap:validation_end
    ]

    test_positions = valid_positions[
        validation_end + gap:
    ]

    # --------------------------------------------------
    # 6. Fit imputer using training samples only
    # --------------------------------------------------

    imputer = SimpleImputer(strategy="median")

    imputer.fit(
        X.iloc[train_positions]
    )

    X_imputed = pd.DataFrame(
        imputer.transform(X),
        columns=X.columns,
        index=X.index
    )

    # --------------------------------------------------
    # 7. Fit feature scaler using training samples only
    # --------------------------------------------------

    scaler = StandardScaler()

    scaler.fit(
        X_imputed.iloc[train_positions]
    )

    X_scaled = pd.DataFrame(
        scaler.transform(X_imputed),
        columns=X.columns,
        index=X.index
    )

    # --------------------------------------------------
    # 8. Create LSTM sequences
    # --------------------------------------------------

    X_train, y_train, train_sequence_timestamps = (
        create_lstm_sequences(
            X_scaled,
            y,
            timestamps,
            train_positions,
            sequence_length
        )
    )

    X_val, y_val, val_sequence_timestamps = (
        create_lstm_sequences(
            X_scaled,
            y,
            timestamps,
            val_positions,
            sequence_length
        )
    )

    X_test, y_test, test_sequence_timestamps = (
        create_lstm_sequences(
            X_scaled,
            y,
            timestamps,
            test_positions,
            sequence_length
        )
    )

    # --------------------------------------------------
    # 9. Preserve original target values
    # --------------------------------------------------

    y_train_original = y_train.copy()
    y_val_original = y_val.copy()
    y_test_original = y_test.copy()

    # --------------------------------------------------
    # 10. Scale scalar targets using training data only
    # --------------------------------------------------

    target_scaler = None

    if target_column != "wind_direction_10m":

        target_scaler = StandardScaler()

        y_train = target_scaler.fit_transform(
            y_train.reshape(-1, 1)
        ).astype(np.float32).reshape(-1)

        y_val = target_scaler.transform(
            y_val.reshape(-1, 1)
        ).astype(np.float32).reshape(-1)

        y_test = target_scaler.transform(
            y_test.reshape(-1, 1)
        ).astype(np.float32).reshape(-1)

    # --------------------------------------------------
    # 11. Return prepared datasets
    # --------------------------------------------------

    return {

        # Input sequences
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,

        # Scaled targets (for model training)
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,

        # Original targets (for evaluation)
        "y_train_original": y_train_original,
        "y_val_original": y_val_original,
        "y_test_original": y_test_original,

        # Timestamps
        "train_timestamps": train_sequence_timestamps,
        "val_timestamps": val_sequence_timestamps,
        "test_timestamps": test_sequence_timestamps,

        # Preprocessing objects
        "imputer": imputer,
        "scaler": scaler,
        "target_scaler": target_scaler,

        # Metadata
        "feature_columns": list(X.columns),
        "sequence_length": sequence_length
    }
