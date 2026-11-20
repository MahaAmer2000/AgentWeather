

import pandas as pd
import numpy as np

from configs.settings import HISTORICAL_DATA_DIR

from src.processing.feature_engineering import (
    create_time_features,
    encode_wind_direction,
    create_lag_features,
    create_rolling_features,
    create_rate_of_change_features
)


def test_feature_engineering():

    # Load historical dataset
    df = pd.read_csv(
        HISTORICAL_DATA_DIR / "combined_historical.csv"
    )

    print("Original shape:", df.shape)

    # ==========================================
    # 1. Test time-based features
    # ==========================================

    featured_df = create_time_features(df)

    time_features = [
        "hour",
        "day_of_week",
        "day_of_month",
        "month",
        "day_of_year",
        "is_weekend",
        "hour_sin",
        "hour_cos",
        "day_of_week_sin",
        "day_of_week_cos",
        "month_sin",
        "month_cos"
    ]

    assert all(
        column in featured_df.columns
        for column in time_features
    )

    assert featured_df["hour"].between(0, 23).all()
    assert featured_df["month"].between(1, 12).all()

    assert featured_df["hour_sin"].between(-1, 1).all()
    assert featured_df["hour_cos"].between(-1, 1).all()

    print("\nTime features created successfully.")
    print("Shape after time features:", featured_df.shape)

    print(
        featured_df[
            [
                "time",
                "hour",
                "day_of_week",
                "day_of_month",
                "month",
                "is_weekend",
                "hour_sin",
                "hour_cos",
                "month_sin",
                "month_cos"
            ]
        ].head(10)
    )

    # ==========================================
    # 2. Test wind direction encoding
    # ==========================================

    wind_df = encode_wind_direction(featured_df)

    wind_features = [
        "wind_direction_sin",
        "wind_direction_cos"
    ]

    assert all(
        column in wind_df.columns
        for column in wind_features
    )

    assert wind_df["wind_direction_sin"].between(-1, 1).all()
    assert wind_df["wind_direction_cos"].between(-1, 1).all()

    wind_check = (
        wind_df["wind_direction_sin"] ** 2
        + wind_df["wind_direction_cos"] ** 2
    )

    assert np.allclose(wind_check, 1)

    print("\nWind direction encoding completed successfully.")
    print("Shape after wind encoding:", wind_df.shape)

    print(
        wind_df[
            [
                "wind_direction_10m",
                "wind_direction_sin",
                "wind_direction_cos"
            ]
        ].head(10)
    )

    # ==========================================
    # 3. Test lag features
    # ==========================================

    lag_df = create_lag_features(wind_df)

    lag_columns = [
        "temperature_2m",
        "relative_humidity_2m",
        "precipitation",
        "precipitation_probability",
        "wind_speed_10m",
        "uv_index",
        "us_aqi",
        "pm2_5",
        "pm10",
        "carbon_monoxide",
        "nitrogen_dioxide",
        "sulphur_dioxide",
        "ozone",
        "wind_direction_sin",
        "wind_direction_cos"
    ]

    lag_periods = [1, 3, 6, 12, 24]

    expected_lag_features = [
        f"{column}_lag_{lag}"
        for column in lag_columns
        for lag in lag_periods
    ]

    assert all(
        column in lag_df.columns
        for column in expected_lag_features
    )

    assert len(expected_lag_features) == 75

    assert lag_df["temperature_2m_lag_1"].isna().iloc[0]
    assert lag_df["temperature_2m_lag_24"].isna().iloc[:24].all()

    assert np.isclose(
        lag_df["temperature_2m_lag_1"].iloc[1],
        lag_df["temperature_2m"].iloc[0]
    )

    assert np.isclose(
        lag_df["temperature_2m_lag_3"].iloc[3],
        lag_df["temperature_2m"].iloc[0]
    )

    assert np.isclose(
        lag_df["temperature_2m_lag_24"].iloc[24],
        lag_df["temperature_2m"].iloc[0]
    )

    print("\nLag features created successfully.")
    print("Total lag features:", len(expected_lag_features))

    print(
        lag_df[
            [
                "time",
                "temperature_2m",
                "temperature_2m_lag_1",
                "temperature_2m_lag_3",
                "temperature_2m_lag_24",
                "us_aqi",
                "us_aqi_lag_1",
                "us_aqi_lag_24"
            ]
        ].head(30)
    )

    # ==========================================
    # 4. Test rolling-window features
    # ==========================================

    rolling_df = create_rolling_features(lag_df)

    rolling_columns = [
        "temperature_2m",
        "relative_humidity_2m",
        "precipitation",
        "precipitation_probability",
        "wind_speed_10m",
        "uv_index",
        "us_aqi",
        "pm2_5",
        "pm10",
        "carbon_monoxide",
        "nitrogen_dioxide",
        "sulphur_dioxide",
        "ozone",
        "wind_direction_sin",
        "wind_direction_cos"
    ]

    rolling_periods = [3, 6, 24]

    expected_rolling_features = [
        f"{column}_rolling_mean_{window}"
        for column in rolling_columns
        for window in rolling_periods
    ]

    assert all(
        column in rolling_df.columns
        for column in expected_rolling_features
    )

    assert len(expected_rolling_features) == 45

    assert (
        rolling_df["temperature_2m_rolling_mean_3"]
        .isna().iloc[:3].all()
    )

    assert (
        rolling_df["temperature_2m_rolling_mean_24"]
        .isna().iloc[:24].all()
    )

    expected_mean_3 = (
        rolling_df["temperature_2m"].iloc[:3].mean()
    )

    assert np.isclose(
        rolling_df["temperature_2m_rolling_mean_3"].iloc[3],
        expected_mean_3
    )

    expected_mean_24 = (
        rolling_df["temperature_2m"].iloc[:24].mean()
    )

    assert np.isclose(
        rolling_df["temperature_2m_rolling_mean_24"].iloc[24],
        expected_mean_24
    )

    print("\nRolling features created successfully.")
    print("Total rolling features:", len(expected_rolling_features))

    print(
        rolling_df[
            [
                "time",
                "temperature_2m",
                "temperature_2m_rolling_mean_3",
                "temperature_2m_rolling_mean_6",
                "temperature_2m_rolling_mean_24",
                "us_aqi",
                "us_aqi_rolling_mean_3",
                "us_aqi_rolling_mean_24"
            ]
        ].head(30)
    )

    # ==========================================
    # 5. Test rate-of-change features
    # ==========================================

    change_df = create_rate_of_change_features(rolling_df)

    change_columns = [
        "temperature_2m",
        "relative_humidity_2m",
        "precipitation",
        "precipitation_probability",
        "wind_speed_10m",
        "uv_index",
        "us_aqi",
        "pm2_5",
        "pm10",
        "carbon_monoxide",
        "nitrogen_dioxide",
        "sulphur_dioxide",
        "ozone"
    ]

    expected_change_features = [
        f"{column}_change_1h"
        for column in change_columns
    ]

    # Check that all expected features exist
    assert all(
        column in change_df.columns
        for column in expected_change_features
    )

    # Check total number of change features
    assert len(expected_change_features) == 13

    # First row should have no previous observation
    assert change_df["temperature_2m_change_1h"].isna().iloc[0]

    # Verify temperature change at second row
    expected_temperature_change = (
        change_df["temperature_2m"].iloc[1]
        - change_df["temperature_2m"].iloc[0]
    )

    assert np.isclose(
        change_df["temperature_2m_change_1h"].iloc[1],
        expected_temperature_change
    )

    # Verify AQI change at second row
    expected_aqi_change = (
        change_df["us_aqi"].iloc[1]
        - change_df["us_aqi"].iloc[0]
    )

    assert np.isclose(
        change_df["us_aqi_change_1h"].iloc[1],
        expected_aqi_change
    )

    print("\nRate-of-change features created successfully.")
    print("Total change features:", len(expected_change_features))

    print(
        change_df[
            [
                "time",
                "temperature_2m",
                "temperature_2m_change_1h",
                "us_aqi",
                "us_aqi_change_1h",
                "pm2_5",
                "pm2_5_change_1h"
            ]
        ].head(10)
    )

    # ==========================================
    # Final results
    # ==========================================

    print("\nFinal dataset shape:", change_df.shape)

    print("\nAll feature engineering tests passed")


if __name__ == "__main__":
    test_feature_engineering()
