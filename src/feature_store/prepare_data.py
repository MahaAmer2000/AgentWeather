
import pandas as pd

from src.processing.feature_engineering import (
    create_time_features,
    encode_wind_direction,
    create_lag_features,
    create_rolling_features,
    create_rate_of_change_features
)

from src.processing.preprocessing import (
    TARGET_COLUMNS,
    create_target_variables,
    validate_hourly_data
)


# --------------------------------------------------
# 1. Load and validate historical observations
# --------------------------------------------------

DATA_PATH = "data/historical/combined_historical.csv"

def prepare_feature_store_data():

    print("=" * 65)
    print("PREPARING HOPSWORKS FEATURE STORE DATA")
    print("=" * 65)

    observations = pd.read_csv(DATA_PATH)

    observations = validate_hourly_data(observations)

    print("\nRaw observations:", observations.shape)


    # --------------------------------------------------
    # 2. Prepare raw observation dataset
    # --------------------------------------------------

    weather_observations = observations.copy()


    # --------------------------------------------------
    # 3. Generate engineered features
    # --------------------------------------------------

    features = observations.copy()

    features = create_time_features(features)
    features = encode_wind_direction(features)
    features = create_lag_features(features)
    features = create_rolling_features(features)
    features = create_rate_of_change_features(features)

    # Remove the first 24 rows because they lack
    # sufficient historical observations for lookback.
    features = features.iloc[24:].copy()

    # Remove raw timestamp and raw wind direction
    # from model input columns.
    # Keep timestamp separately as the primary key.
    feature_columns = [
        column for column in features.columns
        if column not in ["time", "wind_direction_10m"]
    ]

    weather_features = features[
        ["time"] + feature_columns
    ].copy()


    # --------------------------------------------------
    # 4. Generate 24-hour-ahead target variables
    # --------------------------------------------------

    target_data = create_target_variables(observations)

    # Use the same feature-origin timestamps as the
    # engineered feature dataset.
    target_data = target_data.iloc[24:].copy()

    weather_targets = target_data[
        ["time"] + [
            f"{column}_target"
            for column in TARGET_COLUMNS
        ]
    ].copy()

    # Explicitly record when each target is expected.
    weather_targets["target_time"] = (
        weather_targets["time"] + pd.Timedelta(hours=24)
    )


    # --------------------------------------------------
    # 5. Display dataset information
    # --------------------------------------------------

    print("\nDATASET SHAPES")
    print("-" * 65)

    print("Weather observations:", weather_observations.shape)
    print("Weather features:", weather_features.shape)
    print("Weather targets:", weather_targets.shape)

    print("\nFEATURE INFORMATION")
    print("-" * 65)

    print("Number of model features:", len(feature_columns))
    print("Feature columns:")
    print(feature_columns)

    print("\nMISSING VALUES")
    print("-" * 65)

    print("\nRaw observations:")
    print(weather_observations.isna().sum()[
        weather_observations.isna().sum() > 0
    ])

    print("\nEngineered features:")
    print(weather_features.isna().sum()[
        weather_features.isna().sum() > 0
    ])

    print("\nTargets:")
    print(weather_targets.isna().sum()[
        weather_targets.isna().sum() > 0
    ])


    # --------------------------------------------------
    # 6. Verify timestamp alignment
    # --------------------------------------------------

    assert len(weather_features) == len(weather_targets)

    assert (
        weather_features["time"].reset_index(drop=True)
        == weather_targets["time"].reset_index(drop=True)
    ).all()

    assert (
        weather_targets["target_time"]
        - weather_targets["time"]
        == pd.Timedelta(hours=24)
    ).all()

    print("\nTIMESTAMP ALIGNMENT")
    print("-" * 65)

    print("Feature and target timestamps match.")
    print("Every target is exactly 24 hours ahead.")


    # --------------------------------------------------
    # 7. Verify latest forecasting rows
    # --------------------------------------------------

    latest_features = weather_features.tail(24)

    latest_targets = weather_targets.tail(24)

    print("\nLATEST 24 HOURS")
    print("-" * 65)

    print("Latest feature rows:", len(latest_features))

    print(
        "Missing target values in latest 24 hours:",
        latest_targets[
            [f"{column}_target" for column in TARGET_COLUMNS]
        ].isna().sum().sum()
    )

    print("\nLatest feature timestamps:")
    print(latest_features["time"].head(3).to_string(index=False))
    print("...")
    print(latest_features["time"].tail(3).to_string(index=False))

    print("\nCorresponding target timestamps:")
    print(latest_targets["target_time"].tail(3).to_string(index=False))


    # --------------------------------------------------
    # 8. Return prepared datasets
    # --------------------------------------------------

    return (
        weather_observations,
        weather_features,
        weather_targets
    )


if __name__ == "__main__":

    (
        weather_observations,
        weather_features,
        weather_targets
    ) = prepare_feature_store_data()

    print("\n" + "=" * 65)
    print("FEATURE STORE DATA PREPARATION COMPLETED")
    print("=" * 65)
