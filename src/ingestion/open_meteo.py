
import json
from datetime import datetime
from zoneinfo import ZoneInfo

import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry

from configs.settings import (
    LATITUDE,
    LONGITUDE,
    TIMEZONE,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
)

# Initialize API client with caching and retry support
cache_session = requests_cache.CachedSession(
    ".cache",
    expire_after=3600
)

retry_session = retry(
    cache_session,
    retries=5,
    backoff_factor=0.2
)

openmeteo = openmeteo_requests.Client(
    session=retry_session
)


def fetch_weather_forecast():
    """Fetch the next 24 hours of weather forecast data."""

    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation_probability",
            "precipitation",
            "uv_index",
            "wind_speed_10m",
            "wind_direction_10m",
        ],
        "forecast_days": 2,
        "timezone": TIMEZONE,
    }

    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]

    # Generate a unique filename for this API fetch
    timestamp = datetime.now(
        ZoneInfo(TIMEZONE)
    ).strftime("%Y%m%d_%H%M%S_%f")

    raw_file = RAW_DATA_DIR / f"weather_forecast_{timestamp}.json"

    # Extract hourly forecast data
    hourly = response.Hourly()

    variables = [
        "temperature_2m",
        "relative_humidity_2m",
        "precipitation_probability",
        "precipitation",
        "uv_index",
        "wind_speed_10m",
        "wind_direction_10m",
    ]

    # Generate timezone-aware timestamps
    start_time = (
        pd.to_datetime(hourly.Time(), unit="s", utc=True)
        .tz_convert(TIMEZONE)
    )

    periods = hourly.Variables(0).ValuesAsNumpy().size

    timestamps = pd.date_range(
        start=start_time,
        periods=periods,
        freq=pd.Timedelta(seconds=hourly.Interval()),
    )

    # Build structured raw data from the API response
    raw_data = {
        "latitude": response.Latitude(),
        "longitude": response.Longitude(),
        "hourly": {
            "time": [
                time.isoformat() for time in timestamps
            ]
        },
    }

    for index, variable in enumerate(variables):
        values = hourly.Variables(index).ValuesAsNumpy()

        raw_data["hourly"][variable] = values.tolist()

    # Save raw API data without overwriting previous files
    with open(raw_file, "w", encoding="utf-8") as file:
        json.dump(raw_data, file, indent=4)

    # Build the DataFrame for processing
    hourly_data = {
        "time": timestamps
    }

    for index, variable in enumerate(variables):
        hourly_data[variable] = (
            hourly.Variables(index).ValuesAsNumpy()
        )

    weather_df = pd.DataFrame(hourly_data)

    # Keep the next 24 hourly records from the current hour
    current_hour = pd.Timestamp.now(
        tz=TIMEZONE
    ).floor("h")

    next_24_hours = current_hour + pd.Timedelta(hours=24)

    weather_df = weather_df[
        (weather_df["time"] >= current_hour)
        & (weather_df["time"] < next_24_hours)
    ].reset_index(drop=True)

    return weather_df



def fetch_air_quality_forecast():
    """Fetch the next 24 hours of air quality forecast data."""

    url = "https://air-quality-api.open-meteo.com/v1/air-quality"

    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "hourly": [
            "us_aqi",
            "pm2_5",
            "pm10",
            "carbon_monoxide",
            "nitrogen_dioxide",
            "sulphur_dioxide",
            "ozone",
        ],
        "forecast_days": 2,
        "timezone": TIMEZONE,
    }

    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]

    # Generate a unique filename
    timestamp = datetime.now(
        ZoneInfo(TIMEZONE)
    ).strftime("%Y%m%d_%H%M%S_%f")

    raw_file = RAW_DATA_DIR / f"air_quality_forecast_{timestamp}.json"

    hourly = response.Hourly()

    variables = [
        "us_aqi",
        "pm2_5",
        "pm10",
        "carbon_monoxide",
        "nitrogen_dioxide",
        "sulphur_dioxide",
        "ozone",
    ]

    # Generate timezone-aware timestamps
    start_time = (
        pd.to_datetime(hourly.Time(), unit="s", utc=True)
        .tz_convert(TIMEZONE)
    )

    periods = hourly.Variables(0).ValuesAsNumpy().size

    timestamps = pd.date_range(
        start=start_time,
        periods=periods,
        freq=pd.Timedelta(seconds=hourly.Interval()),
    )

    # Save structured raw data
    raw_data = {
        "latitude": response.Latitude(),
        "longitude": response.Longitude(),
        "hourly": {
            "time": [
                time.isoformat() for time in timestamps
            ]
        },
    }

    for index, variable in enumerate(variables):
        values = hourly.Variables(index).ValuesAsNumpy()
        raw_data["hourly"][variable] = values.tolist()

    with open(raw_file, "w", encoding="utf-8") as file:
        json.dump(raw_data, file, indent=4)

    # Create DataFrame
    hourly_data = {
        "time": timestamps
    }

    for index, variable in enumerate(variables):
        hourly_data[variable] = (
            hourly.Variables(index).ValuesAsNumpy()
        )

    air_quality_df = pd.DataFrame(hourly_data)

    # Keep the next 24 hourly records
    current_hour = pd.Timestamp.now(
        tz=TIMEZONE
    ).floor("h")

    next_24_hours = current_hour + pd.Timedelta(hours=24)

    air_quality_df = air_quality_df[
        (air_quality_df["time"] >= current_hour)
        & (air_quality_df["time"] < next_24_hours)
    ].reset_index(drop=True)

    return air_quality_df



def fetch_combined_forecast():
    """Fetch and combine weather and air quality forecasts."""

    # Fetch weather data
    weather_df = fetch_weather_forecast()

    # Fetch air quality data
    air_quality_df = fetch_air_quality_forecast()

    # Merge both datasets using their common timestamps
    combined_df = pd.merge(
        weather_df,
        air_quality_df,
        on="time",
        how="inner"
    )

    # Sort records chronologically
    combined_df = combined_df.sort_values(
        by="time"
    ).reset_index(drop=True)

    # Save combined dataset
    output_file = PROCESSED_DATA_DIR / "combined_forecast.csv"

    combined_df.to_csv(output_file, index=False)

    print(f"Combined dataset saved to: {output_file}")

    return combined_df

