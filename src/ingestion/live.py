

import json
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from configs.settings import (
    LATITUDE,
    LONGITUDE,
    TIMEZONE,
    RAW_DATA_DIR,
)

from src.processing.preprocessing import validate_hourly_data


# --------------------------------------------------
# Configuration
# --------------------------------------------------

PAST_HOURS = 72

WEATHER_URL = "https://api.open-meteo.com/v1/forecast"

AIR_QUALITY_URL = (
    "https://air-quality-api.open-meteo.com/v1/air-quality"
)

WEATHER_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "precipitation_probability",
    "wind_speed_10m",
    "wind_direction_10m",
    "uv_index",
]

AIR_QUALITY_VARIABLES = [
    "us_aqi",
    "pm2_5",
    "pm10",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",
]


# --------------------------------------------------
# 1. Save raw API responses
# --------------------------------------------------

def save_raw_response(data, filename_prefix):

    timestamp = datetime.now(
        ZoneInfo(TIMEZONE)
    ).strftime("%Y%m%d_%H%M%S_%f")

    raw_file = RAW_DATA_DIR / (
        f"{filename_prefix}_{timestamp}.json"
    )

    with open(raw_file, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)

    print(f"Raw API response saved: {raw_file}")

    return raw_file


# --------------------------------------------------
# 2. Convert API timestamps
# --------------------------------------------------

def convert_timestamps(df):

    df = df.copy()

    timestamps = pd.to_datetime(df["time"])

    if timestamps.dt.tz is None:
        df["time"] = timestamps.dt.tz_localize(TIMEZONE)
    else:
        df["time"] = timestamps.dt.tz_convert(TIMEZONE)

    return df


# --------------------------------------------------
# 3. Fetch previous 72 completed weather observations
# --------------------------------------------------

def fetch_live_weather():

    print("\nFetching live weather observations...")

    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "hourly": WEATHER_VARIABLES,
        "past_hours": PAST_HOURS,
        "forecast_days": 1,
        "timezone": TIMEZONE,
    }

    response = requests.get(
        WEATHER_URL,
        params=params,
        timeout=60
    )

    response.raise_for_status()

    raw_data = response.json()

    save_raw_response(
        raw_data,
        "live_weather"
    )

    if "hourly" not in raw_data:
        raise ValueError(
            "Weather API response does not contain hourly data."
        )

    weather_df = pd.DataFrame(
        raw_data["hourly"]
    )

    weather_df = convert_timestamps(weather_df)

    weather_df = weather_df.sort_values(
        "time"
    ).reset_index(drop=True)

    print("Weather data shape:", weather_df.shape)

    return weather_df


# --------------------------------------------------
# 4. Fetch previous 72 completed air-quality observations
# --------------------------------------------------

def fetch_live_air_quality():

    print("\nFetching live air-quality observations...")

    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "hourly": AIR_QUALITY_VARIABLES,
        "past_hours": PAST_HOURS,
        "forecast_days": 1,
        "timezone": TIMEZONE,
    }

    response = requests.get(
        AIR_QUALITY_URL,
        params=params,
        timeout=60
    )

    response.raise_for_status()

    raw_data = response.json()

    save_raw_response(
        raw_data,
        "live_air_quality"
    )

    if "hourly" not in raw_data:
        raise ValueError(
            "Air-quality API response does not contain hourly data."
        )

    air_quality_df = pd.DataFrame(
        raw_data["hourly"]
    )

    air_quality_df = convert_timestamps(
        air_quality_df
    )

    air_quality_df = air_quality_df.sort_values(
        "time"
    ).reset_index(drop=True)

    print("Air-quality data shape:", air_quality_df.shape)

    return air_quality_df


# --------------------------------------------------
# 5. Fetch and combine live observations
# --------------------------------------------------

def fetch_live_observations():

    print("=" * 65)
    print("FETCHING LIVE HOURLY OBSERVATIONS")
    print("=" * 65)

    # Fetch weather and air-quality data.
    weather_df = fetch_live_weather()

    air_quality_df = fetch_live_air_quality()

    # Merge using timestamps.
    combined_df = pd.merge(
        weather_df,
        air_quality_df,
        on="time",
        how="inner",
        validate="one_to_one"
    )

    combined_df = combined_df.sort_values(
        "time"
    ).reset_index(drop=True)

    print("\nCombined observations shape:", combined_df.shape)

    return combined_df


# --------------------------------------------------
# 6. Validate live observations
# --------------------------------------------------

def validate_live_observations(df):

    print("\nValidating live observations...")

    # Validate required columns, timestamps, duplicates,
    # chronological ordering and hourly continuity.
    df = validate_hourly_data(df)

    # Exclude the current incomplete hour.
    current_hour = pd.Timestamp.now(
        tz=TIMEZONE
    ).floor("h")

    expected_start = (
        current_hour - pd.Timedelta(hours=PAST_HOURS)
    )

    expected_end = (
        current_hour - pd.Timedelta(hours=1)
    )

    # Keep only the previous 72 completed hours.
    df = df[
        (df["time"] >= expected_start)
        & (df["time"] <= expected_end)
    ].copy()

    df = df.sort_values(
        "time"
    ).reset_index(drop=True)

    # Exactly 72 observations must be available.
    if len(df) != PAST_HOURS:
        raise ValueError(
            f"Expected {PAST_HOURS} completed hourly records, "
            f"received {len(df)}. "
            f"Expected range: {expected_start} to {expected_end}. "
            f"Received range: "
            f"{df['time'].min()} to {df['time'].max()}"
        )

    # Report missing values without stopping ingestion.
    # Imputation will be handled during feature preparation.
    missing_values = df.isna().sum()

    missing_columns = missing_values[
        missing_values > 0
    ].to_dict()

    if missing_columns:
        print("\nMissing values detected:")
        print(missing_columns)
    else:
        print("\nNo missing values detected.")

    # Verify expected timestamps.
    expected_timestamps = pd.date_range(
        start=expected_start,
        periods=PAST_HOURS,
        freq="h"
    )

    if not df["time"].reset_index(drop=True).equals(
        pd.Series(expected_timestamps, name="time")
    ):
        raise ValueError(
            "Live observations do not match the expected "
            "72 consecutive completed hourly timestamps."
        )

    print("\nLIVE OBSERVATION VALIDATION PASSED")
    print("-" * 65)

    print("Shape:", df.shape)
    print("First timestamp:", df["time"].iloc[0])
    print("Last timestamp:", df["time"].iloc[-1])
    print("Missing values:", df.isna().sum().sum())
    print("Duplicate timestamps:", df["time"].duplicated().sum())

    return df


# --------------------------------------------------
# 7. Main ingestion function
# --------------------------------------------------

def get_live_observations():

    combined_df = fetch_live_observations()

    validated_df = validate_live_observations(
        combined_df
    )

    print("\nLive ingestion completed successfully.")

    return validated_df


# --------------------------------------------------
# 8. Script entry point
# --------------------------------------------------

if __name__ == "__main__":

    observations = get_live_observations()

    print("\nFinal live observations:")
    print(observations.tail())
