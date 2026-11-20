
import pandas as pd
import numpy as np



def create_time_features(df):
    """
    Extract time-based features from hourly weather data.
    """

    df = df.copy()

    # Ensure timestamp is in datetime format
    df["time"] = pd.to_datetime(df["time"])

    # Extract basic time features
    df["hour"] = df["time"].dt.hour
    df["day_of_week"] = df["time"].dt.dayofweek
    df["day_of_month"] = df["time"].dt.day
    df["month"] = df["time"].dt.month
    df["day_of_year"] = df["time"].dt.dayofyear

    # Identify weekends (Saturday = 5, Sunday = 6)
    df["is_weekend"] = (
        df["day_of_week"] >= 5
    ).astype(int)

    # Cyclical encoding of hour
    df["hour_sin"] = np.sin(
        2 * np.pi * df["hour"] / 24
    )

    df["hour_cos"] = np.cos(
        2 * np.pi * df["hour"] / 24
    )

    # Cyclical encoding of day of week
    df["day_of_week_sin"] = np.sin(
        2 * np.pi * df["day_of_week"] / 7
    )

    df["day_of_week_cos"] = np.cos(
        2 * np.pi * df["day_of_week"] / 7
    )

    # Cyclical encoding of month
    df["month_sin"] = np.sin(
        2 * np.pi * (df["month"] - 1) / 12
    )

    df["month_cos"] = np.cos(
        2 * np.pi * (df["month"] - 1) / 12
    )

    return df


def encode_wind_direction(df):
    """
    Convert wind direction into cyclical sine and cosine features.
    """

    df = df.copy()

    df["wind_direction_sin"] = np.sin(
        np.radians(df["wind_direction_10m"])
    )

    df["wind_direction_cos"] = np.cos(
        np.radians(df["wind_direction_10m"])
    )

    return df


def create_lag_features(df):
    """
    Create lag features from previous hourly weather
    and air-quality observations.
    """

    df = df.copy()

    # Ensure data is sorted chronologically
    df = df.sort_values("time").reset_index(drop=True)

    # Columns for creating lag features
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

    # Number of previous hours to consider
    lag_periods = [1, 3, 6, 12, 24]

    # Create lag features
    for column in lag_columns:
        for lag in lag_periods:
            df[f"{column}_lag_{lag}"] = df[column].shift(lag)

    return df


def create_rolling_features(df):
    """
    Create rolling averages using previous hourly
    weather and air-quality observations.
    """

    df = df.copy()

    # Ensure data is sorted chronologically
    df = df.sort_values("time").reset_index(drop=True)

    # Variables used for rolling averages
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

    # Rolling window sizes
    rolling_periods = [3, 6, 24]

    # Create rolling mean features
    for column in rolling_columns:
        for window in rolling_periods:
            df[f"{column}_rolling_mean_{window}"] = (
                df[column]
                .shift(1)
                .rolling(window=window)
                .mean()
            )

    return df


def create_rate_of_change_features(df):
    """
    Create rate-of-change features by calculating
    the difference between consecutive hourly observations.
    """

    df = df.copy()

    # Ensure chronological ordering
    df = df.sort_values("time").reset_index(drop=True)

    # Variables for calculating changes
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

    # Calculate hourly changes
    for column in change_columns:
        df[f"{column}_change_1h"] = df[column].diff(1)

    return df


