
import json
from datetime import datetime
from zoneinfo import ZoneInfo
import requests

import pandas as pd

from configs.settings import (
    LATITUDE,
    LONGITUDE,
    TIMEZONE,
    RAW_DATA_DIR,
)

#RETRIEVES ALL WEATHER VARIABLES EXCEPT TWO
def fetch_historical_weather(start_date, end_date):
    """
    Fetch historical weather data from Open-Meteo.

    Parameters:
        start_date: Start date in YYYY-MM-DD format.
        end_date: End date in YYYY-MM-DD format.

    Returns:
        DataFrame containing hourly historical weather data.
    """

    url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation",
            "wind_speed_10m",
            "wind_direction_10m",
        ],
        "timezone": TIMEZONE,
    }

    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()

    raw_data = response.json()

    # Save the original API response
    timestamp = datetime.now(
        ZoneInfo(TIMEZONE)
    ).strftime("%Y%m%d_%H%M%S_%f")

    raw_file = RAW_DATA_DIR / f"historical_weather_{timestamp}.json"

    with open(raw_file, "w", encoding="utf-8") as file:
        json.dump(raw_data, file, indent=4)

    # Convert hourly response into a DataFrame
    weather_df = pd.DataFrame(raw_data["hourly"])

    # Convert timestamps to timezone-aware datetime values
    weather_df["time"] = pd.to_datetime(
        weather_df["time"]
    ).dt.tz_localize(TIMEZONE)

    weather_df = weather_df.sort_values(
        "time"
    ).reset_index(drop=True)

    print(f"Raw data saved to: {raw_file}")
    print(f"Historical weather shape: {weather_df.shape}")

    return weather_df


#RETRIEVES UV INDEX AND RAIN PROBABILITY
def fetch_historical_forecast_features(start_date, end_date):
    """
    Fetch archived rain probability and UV index forecasts.
    """

    url = "https://historical-forecast-api.open-meteo.com/v1/forecast"

    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": [
            "precipitation_probability",
            "uv_index",
        ],
        "timezone": TIMEZONE,
    }

    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()

    raw_data = response.json()

    # Save original API response
    timestamp = datetime.now(
        ZoneInfo(TIMEZONE)
    ).strftime("%Y%m%d_%H%M%S_%f")

    raw_file = RAW_DATA_DIR / f"historical_forecast_features_{timestamp}.json"

    with open(raw_file, "w", encoding="utf-8") as file:
        json.dump(raw_data, file, indent=4)

    # Convert response into DataFrame
    forecast_df = pd.DataFrame(raw_data["hourly"])

    forecast_df["time"] = pd.to_datetime(
        forecast_df["time"]
    ).dt.tz_localize(TIMEZONE)

    forecast_df = forecast_df.sort_values(
        "time"
    ).reset_index(drop=True)

    print(f"Raw data saved to: {raw_file}")
    print(f"Historical forecast features shape: {forecast_df.shape}")

    return forecast_df

#RETRIEVES AQI AND OTHER VARIABLES
def fetch_historical_air_quality(start_date, end_date):
    """
    Fetch historical air-quality data from Open-Meteo.
    """

    url = "https://air-quality-api.open-meteo.com/v1/air-quality"

    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": [
            "us_aqi",
            "pm2_5",
            "pm10",
            "carbon_monoxide",
            "nitrogen_dioxide",
            "sulphur_dioxide",
            "ozone",
        ],
        "timezone": TIMEZONE,
    }

    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()

    raw_data = response.json()

    # Save original API response
    timestamp = datetime.now(
        ZoneInfo(TIMEZONE)
    ).strftime("%Y%m%d_%H%M%S_%f")

    raw_file = RAW_DATA_DIR / f"historical_air_quality_{timestamp}.json"

    with open(raw_file, "w", encoding="utf-8") as file:
        json.dump(raw_data, file, indent=4)

    # Convert response into DataFrame
    air_quality_df = pd.DataFrame(raw_data["hourly"])

    air_quality_df["time"] = pd.to_datetime(
        air_quality_df["time"]
    ).dt.tz_localize(TIMEZONE)

    air_quality_df = air_quality_df.sort_values(
        "time"
    ).reset_index(drop=True)

    print(f"Raw data saved to: {raw_file}")
    print(f"Historical air-quality shape: {air_quality_df.shape}")

    return air_quality_df



def fetch_combined_historical_data(start_date, end_date):
    """
    Fetch and combine historical weather and air-quality data.
    """

    # Fetch historical weather data
    weather_df = fetch_historical_weather(
        start_date,
        end_date
    )

    # Fetch archived rain probability and UV index
    forecast_df = fetch_historical_forecast_features(
        start_date,
        end_date
    )

    # Fetch historical air-quality data
    air_quality_df = fetch_historical_air_quality(
        start_date,
        end_date
    )

    # Merge weather and archived forecast features
    combined_df = pd.merge(
        weather_df,
        forecast_df,
        on="time",
        how="inner"
    )

    # Merge air-quality data
    combined_df = pd.merge(
        combined_df,
        air_quality_df,
        on="time",
        how="inner"
    )

    # Sort chronologically
    combined_df = combined_df.sort_values(
        "time"
    ).reset_index(drop=True)

    print("\nCombined historical dataset shape:")
    print(combined_df.shape)

    print("\nMissing values:")
    print(combined_df.isnull().sum())

    return combined_df





