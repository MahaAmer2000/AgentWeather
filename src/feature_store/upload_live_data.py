
import os

import hopsworks
import pandas as pd
from dotenv import load_dotenv


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

EXPECTED_COLUMNS = {
    "observations": 15,
    "features": 161,
    "targets": 10,
}

TARGET_TIME_COLUMN = "target_time"


# --------------------------------------------------
# 2. Validate live datasets
# --------------------------------------------------

def validate_live_datasets(
    weather_observations,
    weather_features,
    weather_targets,
):

    datasets = {
        "observations": weather_observations,
        "features": weather_features,
        "targets": weather_targets,
    }

    for name, dataframe in datasets.items():

        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError(
                f"{name} must be a pandas DataFrame."
            )

        if len(dataframe.columns) != EXPECTED_COLUMNS[name]:
            raise ValueError(
                f"{name}: expected "
                f"{EXPECTED_COLUMNS[name]} columns, "
                f"found {len(dataframe.columns)}."
            )

        if "time" not in dataframe.columns:
            raise ValueError(
                f"{name}: missing time column."
            )

        if dataframe.empty:

            if name == "targets":
                print(
                    "Targets dataset is empty. "
                    "Target upload will be skipped."
                )
                continue

            raise ValueError(
                f"{name} dataset cannot be empty."
            )

        if dataframe["time"].isna().any():
            raise ValueError(
                f"{name}: missing timestamps found."
            )

        if dataframe["time"].duplicated().any():
            raise ValueError(
                f"{name}: duplicate timestamps found."
            )

        # Missing values are allowed in observations
        # and engineered features.
        if name in ["observations", "features"]:

            missing_count = int(
                dataframe.isna().sum().sum()
            )

            print(
                f"{name}: {dataframe.shape} - "
                f"validation passed."
            )

            print(
                f"Missing values preserved: {missing_count}"
            )

        # Target labels must be complete.
        elif name == "targets":

            if TARGET_TIME_COLUMN not in dataframe.columns:
                raise ValueError(
                    "targets: missing target_time column."
                )

            target_columns = [
                column
                for column in dataframe.columns
                if column not in ["time", TARGET_TIME_COLUMN]
            ]

            if dataframe[target_columns].isna().any().any():
                raise ValueError(
                    "Targets contain missing values. "
                    "Incomplete target rows must be skipped."
                )

            if dataframe[TARGET_TIME_COLUMN].isna().any():
                raise ValueError(
                    "Targets contain missing target_time values."
                )

            expected_target_time = (
                dataframe["time"]
                + pd.Timedelta(hours=24)
            )

            if not dataframe[TARGET_TIME_COLUMN].equals(
                expected_target_time
            ):
                raise ValueError(
                    "Target timestamps are not exactly "
                    "24 hours after their feature timestamps."
                )

            print(
                f"targets: {dataframe.shape} - "
                "validation passed."
            )

    # --------------------------------------------------
    # Validate feature and target alignment
    # --------------------------------------------------

    if not weather_targets.empty:

        if not weather_targets["time"].isin(
            weather_features["time"]
        ).all():

            raise ValueError(
                "Some target timestamps do not have "
                "matching feature timestamps."
            )

    print("\nAll live datasets passed validation.")


# --------------------------------------------------
# 3. Convert timestamps to UTC
# --------------------------------------------------

def convert_timestamps_to_utc(
    weather_observations,
    weather_features,
    weather_targets,
):

    # Work on copies so the original prepared datasets
    # are not modified.
    weather_observations = weather_observations.copy()
    weather_features = weather_features.copy()
    weather_targets = weather_targets.copy()

    datasets = [
        weather_observations,
        weather_features,
        weather_targets,
    ]

    for dataframe in datasets:

        dataframe["time"] = pd.to_datetime(
            dataframe["time"],
            utc=True
        )

    if not weather_targets.empty:

        weather_targets["target_time"] = pd.to_datetime(
            weather_targets["target_time"],
            utc=True
        )

    print("\nAll timestamps converted to UTC.")

    return (
        weather_observations,
        weather_features,
        weather_targets,
    )


# --------------------------------------------------
# 4. Connect to Hopsworks
# --------------------------------------------------

def get_feature_store():

    load_dotenv()

    api_key = os.getenv("HOPSWORKS_API_KEY")
    project_name = os.getenv("HOPSWORKS_PROJECT_NAME")

    if not api_key:
        raise ValueError(
            "HOPSWORKS_API_KEY is missing from .env"
        )

    if not project_name:
        raise ValueError(
            "HOPSWORKS_PROJECT_NAME is missing from .env"
        )

    print("\nConnecting to Hopsworks...")

    project = hopsworks.login(
        project=project_name,
        api_key_value=api_key,
        engine="python"
    )

    feature_store = project.get_feature_store()

    print("Connected to:", project.name)

    return feature_store


# --------------------------------------------------
# 5. Upload live datasets
# --------------------------------------------------

def upload_live_data(
    weather_observations,
    weather_features,
    weather_targets,
):

    print("\n" + "=" * 65)
    print("VALIDATING LIVE DATASETS")
    print("=" * 65)

    # Convert timestamps before validation and upload.
    (
        weather_observations,
        weather_features,
        weather_targets,
    ) = convert_timestamps_to_utc(
        weather_observations,
        weather_features,
        weather_targets,
    )

    validate_live_datasets(
        weather_observations,
        weather_features,
        weather_targets,
    )

    # Connect only after local validation succeeds.
    feature_store = get_feature_store()

    # --------------------------------------------------
    # Retrieve existing Feature Groups
    # --------------------------------------------------

    print("\nRetrieving existing Feature Groups...")

    observations_fg = feature_store.get_feature_group(
        name="weather_observations",
        version=1
    )

    features_fg = feature_store.get_feature_group(
        name="weather_features",
        version=1
    )

    targets_fg = feature_store.get_feature_group(
        name="weather_targets",
        version=1
    )

    print("All three Feature Groups retrieved.")

    # --------------------------------------------------
    # Upload observations
    # --------------------------------------------------

    print("\nUpserting live observations...")

    observations_fg.insert(
        weather_observations,
        operation="upsert",
        wait=True
    )

    print("Live observations uploaded.")

    # --------------------------------------------------
    # Upload features
    # --------------------------------------------------

    print("\nUpserting live features...")

    features_fg.insert(
        weather_features,
        operation="upsert",
        wait=True
    )

    print("Live features uploaded.")

    # --------------------------------------------------
    # Upload targets
    # --------------------------------------------------

    if not weather_targets.empty:

        print("\nUpserting live targets...")

        targets_fg.insert(
            weather_targets,
            operation="upsert",
            wait=True
        )

        print("Live targets uploaded.")

    else:

        print(
            "\nNo complete target rows available. "
            "Target upload skipped."
        )

    # --------------------------------------------------
    # Completion
    # --------------------------------------------------

    print("\n" + "=" * 65)
    print("LIVE FEATURE STORE UPLOAD COMPLETED")
    print("=" * 65)

    print("Observations:", len(weather_observations))
    print("Features:", len(weather_features))
    print("Targets:", len(weather_targets))

    print("\nObservation UTC range:")
    print(weather_observations["time"].min())
    print(weather_observations["time"].max())

    print("\nFeature UTC range:")
    print(weather_features["time"].min())
    print(weather_features["time"].max())

    if not weather_targets.empty:

        print("\nTarget valid-time UTC range:")
        print(weather_targets["target_time"].min())
        print(weather_targets["target_time"].max())


# --------------------------------------------------
# 6. Run through the main pipeline
# --------------------------------------------------

if __name__ == "__main__":

    from src.ingestion.live import get_live_observations

    from src.feature_store.prepare_live_data import (
        prepare_live_feature_store_data
    )

    print("\nFETCHING LIVE WEATHER DATA")

    observations = get_live_observations()

    print("\nPREPARING FEATURE STORE DATA")

    (
        weather_observations,
        weather_features,
        weather_targets,
    ) = prepare_live_feature_store_data(observations)

    upload_live_data(
        weather_observations,
        weather_features,
        weather_targets,
    )
