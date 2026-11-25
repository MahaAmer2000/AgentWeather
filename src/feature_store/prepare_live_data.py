
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

from src.processing.imputation import impute_missing_values


# --------------------------------------------------
# 1. Prepare live Feature Store datasets
# --------------------------------------------------

def prepare_live_feature_store_data(observations):

    print("=" * 65)
    print("PREPARING LIVE FEATURE STORE DATA")
    print("=" * 65)

    # Validate and sort original observations.
    observations = validate_hourly_data(observations)

    if len(observations) != 72:
        raise ValueError(
            f"Expected exactly 72 hourly records, received {len(observations)}."
        )

    print("\nInput observations:", observations.shape)

    # --------------------------------------------------
    # 2. Preserve original observations
    # --------------------------------------------------

    weather_observations = observations.copy()

    # --------------------------------------------------
    # 3. Apply existing raw-data imputation
    # --------------------------------------------------

    print("\nApplying conservative raw-data imputation...")

    imputed_observations, imputation_report = (
        impute_missing_values(observations.copy())
    )

    # Remove temporary missingness indicators.
    flag_columns = [
        column for column in imputed_observations.columns
        if column.endswith("_was_missing")
    ]

    imputed_observations = imputed_observations.drop(
        columns=flag_columns
    )

    print(
        "\nTemporary missingness indicators excluded:",
        len(flag_columns)
    )

    # --------------------------------------------------
    # 4. Generate features from imputed observations
    # --------------------------------------------------

    features = imputed_observations.copy()

    features = create_time_features(features)
    features = encode_wind_direction(features)
    features = create_lag_features(features)
    features = create_rolling_features(features)
    features = create_rate_of_change_features(features)

    # Select the middle 24 forecast origins.
    features = features.iloc[24:48].copy()

    feature_columns = [
        column for column in features.columns
        if column not in ["time", "wind_direction_10m"]
    ]

    weather_features = features[
        ["time"] + feature_columns
    ].copy()

    # Remaining missing feature values are preserved.

    # --------------------------------------------------
    # 5. Generate targets from original observations
    # --------------------------------------------------

    # Target labels are never imputed.
    target_data = create_target_variables(observations)

    target_data = target_data.iloc[24:48].copy()

    target_columns = [
        f"{column}_target" for column in TARGET_COLUMNS
    ]

    weather_targets = target_data[
        ["time"] + target_columns
    ].copy()

    weather_targets["target_time"] = (
        weather_targets["time"] + pd.Timedelta(hours=24)
    )

    # Skip incomplete target rows.
    complete_target_mask = (
        weather_targets[target_columns].notna().all(axis=1)
    )

    skipped_targets = int((~complete_target_mask).sum())

    weather_targets = weather_targets.loc[
        complete_target_mask
    ].copy().reset_index(drop=True)

    if skipped_targets > 0:
        print(
            f"\nWARNING: Skipped {skipped_targets} rows "
            "with incomplete target labels."
        )

    # --------------------------------------------------
    # 6. Align numeric types with Hopsworks schemas
    # --------------------------------------------------

    integer_type_columns = {
        "weather_observations": (
            weather_observations,
            [
                "relative_humidity_2m",
                "wind_direction_10m",
                "us_aqi"
            ]
        ),
        "weather_features": (
            weather_features,
            [
                "relative_humidity_2m",
                "us_aqi"
            ]
        )
    }

    for group_name, (dataframe, columns) in integer_type_columns.items():

        for column in columns:

            values = pd.to_numeric(
                dataframe[column],
                errors="raise"
            )

            non_missing = values.dropna()

            if group_name == "weather_features":

                # Feature engineering can produce decimal values
                # through interpolation and rolling calculations.
                # Round to match the existing Hopsworks bigint schema.
                values = values.round()

            elif not non_missing.mod(1).eq(0).all():

                # Never silently alter original observations.
                raise ValueError(
                    f"{group_name}.{column} contains decimal values. "
                    "Original observations cannot be rounded."
                )

            # Nullable integer preserves NaN values.
            dataframe[column] = values.astype("Int64")

    # Double columns.
    weather_observations["precipitation_probability"] = (
        weather_observations["precipitation_probability"].astype("float64")
    )

    weather_features["precipitation_probability"] = (
        weather_features["precipitation_probability"].astype("float64")
    )

    # Feature Group calendar columns: int.
    calendar_int_columns = [
        "hour",
        "day_of_week",
        "day_of_month",
        "month",
        "day_of_year"
    ]

    for column in calendar_int_columns:
        weather_features[column] = (
            weather_features[column].astype("int32")
        )

    # Feature Group weekend column: bigint.
    weather_features["is_weekend"] = (
        weather_features["is_weekend"].astype("int64")
    )

    # Target variables must be complete and double type.
    for column in target_columns:
        weather_targets[column] = (
            weather_targets[column].astype("float64")
        )

    # --------------------------------------------------
    # 7. Validate output shapes and timestamp alignment
    # --------------------------------------------------

    assert weather_observations.shape == (72, 15)

    assert weather_features.shape == (24, 161)

    assert weather_targets.shape[1] == 10

    # Every retained target must have a corresponding feature row.
    assert weather_targets["time"].isin(
        weather_features["time"]
    ).all()

    # Feature timestamps must match the 24 forecast origins.
    assert weather_features["time"].reset_index(drop=True).equals(
        observations["time"].iloc[24:48].reset_index(drop=True)
    )

    # Every target must be exactly 24 hours after its origin.
    assert (
        weather_targets["target_time"] - weather_targets["time"]
    ).eq(pd.Timedelta(hours=24)).all()

    # Every target timestamp must match a future observation.
    expected_target_times = observations["time"].iloc[48:72]

    assert weather_targets["target_time"].isin(
        expected_target_times
    ).all()

    # --------------------------------------------------
    # 8. Display preparation results
    # --------------------------------------------------

    print("\nLIVE DATASET SHAPES")
    print("-" * 65)
    print("Weather observations:", weather_observations.shape)
    print("Weather features:", weather_features.shape)
    print("Weather targets:", weather_targets.shape)

    print("\nIMPUTATION SUMMARY")
    print("-" * 65)

    if not imputation_report.empty:
        print(imputation_report.to_string(index=False))

    print("\nMISSING VALUES AFTER IMPUTATION")
    print("-" * 65)
    print(
        "Original observations:",
        weather_observations.isna().sum().sum()
    )
    print(
        "Engineered features:",
        weather_features.isna().sum().sum()
    )
    print(
        "Target labels:",
        weather_targets.isna().sum().sum()
    )
    print("Incomplete target rows skipped:", skipped_targets)

    print("\nFEATURE-TARGET ALIGNMENT")
    print("-" * 65)
    print("Feature rows:", len(weather_features))
    print("Complete target rows:", len(weather_targets))
    print("Forecast horizon: 24 hours")

    print("\nPreparation completed successfully.")

    return (
        weather_observations,
        weather_features,
        weather_targets
    )
