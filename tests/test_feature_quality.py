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


def test_feature_quality():

    # Load historical dataset
    file_path = HISTORICAL_DATA_DIR / "combined_historical.csv"
    df = pd.read_csv(file_path)

    df["time"] = pd.to_datetime(df["time"])

    print("Original shape:", df.shape)

    # Apply all feature engineering functions
    df = create_time_features(df)
    df = encode_wind_direction(df)
    df = create_lag_features(df)
    df = create_rolling_features(df)
    df = create_rate_of_change_features(df)

    print("\nEngineered shape:", df.shape)

    # 1. Check missing values
    missing_counts = df.isna().sum()

    missing_report = pd.DataFrame({
        "missing_count": missing_counts,
        "missing_percentage": (
            missing_counts / len(df) * 100
        )
    })

    missing_report = missing_report[
        missing_report["missing_count"] > 0
    ].sort_values(
        "missing_count",
        ascending=False
    )

    print("\nColumns with missing values:")
    print(missing_report.head(30))

    print(
        "\nTotal missing cells:",
        int(df.isna().sum().sum())
    )

    # 2. Check duplicate timestamps
    print(
        "\nDuplicate timestamps:",
        df["time"].duplicated().sum()
    )

    # 3. Check chronological ordering
    print(
        "Chronologically sorted:",
        df["time"].is_monotonic_increasing
    )

    # 4. Check infinite values
    numeric_df = df.select_dtypes(include=np.number)

    infinite_counts = np.isinf(numeric_df).sum()

    print(
        "\nTotal infinite values:",
        int(infinite_counts.sum())
    )

    print("\nColumns containing infinite values:")
    print(infinite_counts[infinite_counts > 0])

    print("\nFeature quality inspection completed.")


if __name__ == "__main__":
    test_feature_quality()