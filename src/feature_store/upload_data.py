
import os
import hopsworks

from dotenv import load_dotenv

from src.feature_store.prepare_data import (
    prepare_feature_store_data
)


# --------------------------------------------------
# 1. Load Hopsworks credentials
# --------------------------------------------------

load_dotenv()

api_key = os.getenv("HOPSWORKS_API_KEY")
project_name = os.getenv("HOPSWORKS_PROJECT_NAME")

if not api_key:
    raise ValueError("HOPSWORKS_API_KEY is missing from .env")

if not project_name:
    raise ValueError("HOPSWORKS_PROJECT_NAME is missing from .env")


# --------------------------------------------------
# 2. Connect to Hopsworks
# --------------------------------------------------

print("\nConnecting to Hopsworks...")

project = hopsworks.login(
    project=project_name,
    api_key_value=api_key
)

feature_store = project.get_feature_store()

print("Connected to:", project.name)
print("Feature Store accessed successfully.")


# --------------------------------------------------
# 3. Prepare datasets
# --------------------------------------------------

(
    weather_observations,
    weather_features,
    weather_targets
) = prepare_feature_store_data()


# --------------------------------------------------
# 4. Create Feature Groups
# --------------------------------------------------

print("\nCreating or retrieving Feature Groups...")

observations_fg = feature_store.get_or_create_feature_group(
    name="weather_observations",
    version=1,
    description="Raw hourly weather and air quality observations for Lahore.",
    primary_key=["time"],
    event_time="time",
    online_enabled=False,
    time_travel_format="ICEBERG"
)

features_fg = feature_store.get_or_create_feature_group(
    name="weather_features",
    version=1,
    description="Engineered hourly weather and air quality features for 24-hour forecasting.",
    primary_key=["time"],
    event_time="time",
    online_enabled=False,
    time_travel_format="ICEBERG"
)

targets_fg = feature_store.get_or_create_feature_group(
    name="weather_targets",
    version=1,
    description="Weather and air quality targets aligned 24 hours after each feature timestamp.",
    primary_key=["time"],
    event_time="time",
    online_enabled=False,
    time_travel_format="ICEBERG"
)

print("All three Feature Groups are ready.")


# --------------------------------------------------
# 5. Upload raw observations
# --------------------------------------------------

print("\nUploading raw observations...")

observations_fg.insert(
    weather_observations,
    wait=True
)

print("Raw observations uploaded.")


# --------------------------------------------------
# 6. Upload engineered features
# --------------------------------------------------

print("\nUploading engineered features...")

features_fg.insert(
    weather_features,
    wait=True
)

print("Engineered features uploaded.")


# --------------------------------------------------
# 7. Upload future targets
# --------------------------------------------------

print("\nUploading future targets...")

targets_fg.insert(
    weather_targets,
    wait=True
)

print("Future targets uploaded.")


# --------------------------------------------------
# 8. Completion
# --------------------------------------------------

print("\n" + "=" * 65)
print("FEATURE STORE UPLOAD COMPLETED")
print("=" * 65)

print("Weather observations:", len(weather_observations))
print("Weather features:", len(weather_features))
print("Weather targets:", len(weather_targets))
