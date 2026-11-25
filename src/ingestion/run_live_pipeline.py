
from src.ingestion.live import get_live_observations

from src.feature_store.prepare_live_data import (
    prepare_live_feature_store_data
)

from src.feature_store.upload_live_data import (
    upload_live_data
)


def run_live_pipeline():

    print("\n" + "=" * 70)
    print("AGENTWEATHER - LIVE HOURLY PIPELINE")
    print("=" * 70)

    # 1. Fetch latest observations.
    observations = get_live_observations()

    # 2. Prepare Feature Store datasets.
    (
        weather_observations,
        weather_features,
        weather_targets
    ) = prepare_live_feature_store_data(observations)

    # 3. Upload all three datasets.
    upload_live_data(
        weather_observations,
        weather_features,
        weather_targets
    )

    print("\n" + "=" * 70)
    print("AGENTWEATHER LIVE PIPELINE FINISHED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    run_live_pipeline()
