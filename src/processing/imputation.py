
import numpy as np
import pandas as pd


# Maximum number of consecutive missing hours to interpolate.
MAX_GAP_HOURS = 3


# Variables approved for short-gap linear interpolation.
LINEAR_COLUMNS = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "us_aqi",
    "pm2_5",
    "pm10",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",
]


# Wind direction requires circular interpolation.
CIRCULAR_COLUMNS = ["wind_direction_10m"]


# These variables will not be automatically imputed.
PRESERVE_MISSING_COLUMNS = [
    "precipitation",
    "precipitation_probability",
    "uv_index",
]


def _interpolate_short_gaps(series, max_gap=MAX_GAP_HOURS):
    """
    Interpolate complete internal missing gaps only when
    their length is less than or equal to max_gap.

    Longer gaps and leading/trailing gaps remain missing.
    """

    result = series.copy()

    missing = result.isna()

    # Assign a group ID to each consecutive missing/non-missing run.
    groups = (missing != missing.shift()).cumsum()

    # Process each consecutive missing run independently.
    for _, group in result[missing].groupby(groups[missing]):
        gap_indices = group.index
        gap_length = len(gap_indices)

        # Only interpolate the entire gap if it is short enough.
        if gap_length > max_gap:
            continue

        start = gap_indices[0]
        end = gap_indices[-1]

        # Do not interpolate leading or trailing gaps.
        if start == 0 or end == len(result) - 1:
            continue

        # Both neighboring values must be available.
        if (
            pd.isna(result.iloc[start - 1])
            or pd.isna(result.iloc[end + 1])
        ):
            continue

        interpolated = result.interpolate(method="linear")

        result.iloc[gap_indices] = interpolated.iloc[gap_indices]

    return result


def _interpolate_wind_direction(series, max_gap=MAX_GAP_HOURS):
    """
    Interpolate wind direction using sine/cosine components
    to respect its circular nature.

    For example, 359° and 1° should interpolate near 0°,
    not 180°.
    """

    radians = np.radians(series)

    sin_component = pd.Series(
        np.sin(radians),
        index=series.index
    )

    cos_component = pd.Series(
        np.cos(radians),
        index=series.index
    )

    sin_filled = _interpolate_short_gaps(
        sin_component, max_gap
    )

    cos_filled = _interpolate_short_gaps(
        cos_component, max_gap
    )

    interpolated = np.degrees(
        np.arctan2(sin_filled, cos_filled)
    ) % 360

    # Preserve original values and fill only missing positions.
    return series.fillna(interpolated)


def impute_missing_values(df, max_gap=MAX_GAP_HOURS):
    """
    Apply conservative, variable-specific imputation
    to hourly raw weather observations.

    Rules:
    - Short internal gaps (up to max_gap hours) in approved
      continuous variables are interpolated.
    - Wind direction uses circular interpolation.
    - Precipitation, precipitation probability, and UV index
      are never automatically imputed.
    - Long gaps and leading/trailing gaps remain missing.
    - Missingness flags preserve information about original gaps.

    Returns:
        imputed_df: processed copy of the input DataFrame
        report: DataFrame summarizing imputation results
    """

    if "time" not in df.columns:
        raise ValueError("DataFrame must contain a 'time' column.")

    if not isinstance(max_gap, (int, np.integer)) or max_gap < 1:
        raise ValueError("max_gap must be a positive integer.")

    result = df.copy()

    result["time"] = pd.to_datetime(result["time"])

    result = result.sort_values("time").reset_index(drop=True)

    if result["time"].isna().any():
        raise ValueError("Missing timestamps found.")

    if result["time"].duplicated().any():
        raise ValueError("Duplicate timestamps found.")

    if not result["time"].diff().dropna().eq(
        pd.Timedelta(hours=1)
    ).all():
        raise ValueError(
            "Data must have continuous hourly timestamps."
        )

    original_missing = result.isna().sum()

    # Create missingness indicators before imputation.
    # A value of 1 means the original observation was missing.
    missing_columns = result.columns[
        result.isna().any()
    ].tolist()

    for column in missing_columns:
        if column != "time":
            flag_column = f"{column}_was_missing"

            if flag_column in result.columns:
                raise ValueError(
                    f"Missingness flag column already exists: {flag_column}"
                )

            result[flag_column] = (
                result[column].isna().astype("int8")
            )

    report_rows = []

    # Linear interpolation for approved continuous variables.
    for column in LINEAR_COLUMNS:
        if column not in result.columns:
            continue

        before = int(result[column].isna().sum())

        result[column] = pd.to_numeric(
            result[column], errors="coerce"
        )

        result[column] = _interpolate_short_gaps(
            result[column], max_gap
        )

        after = int(result[column].isna().sum())

        report_rows.append({
            "column": column,
            "method": "linear interpolation",
            "imputed": before - after,
            "remaining_missing": after,
        })

    # Circular interpolation for wind direction.
    for column in CIRCULAR_COLUMNS:
        if column not in result.columns:
            continue

        before = int(result[column].isna().sum())

        result[column] = pd.to_numeric(
            result[column], errors="coerce"
        )

        result[column] = _interpolate_wind_direction(
            result[column], max_gap
        )

        after = int(result[column].isna().sum())

        report_rows.append({
            "column": column,
            "method": "circular interpolation",
            "imputed": before - after,
            "remaining_missing": after,
        })

    # Preserve missing values in variables not approved for imputation.
    for column in PRESERVE_MISSING_COLUMNS:
        if column not in result.columns:
            continue

        report_rows.append({
            "column": column,
            "method": "preserve missing values",
            "imputed": 0,
            "remaining_missing": int(result[column].isna().sum()),
        })

    report = pd.DataFrame(report_rows)

    print("\nIMPUTATION REPORT")
    print("=" * 65)
    print(report.to_string(index=False))

    print("\nTOTAL MISSING VALUES")
    print("Before:", int(original_missing.sum()))
    print("After: ", int(result.isna().sum().sum()))

    print("\nMISSINGNESS FLAGS CREATED")
    if missing_columns:
        print([
            f"{column}_was_missing"
            for column in missing_columns
            if column != "time"
        ])
    else:
        print("No missingness flags were needed.")

    return result, report
