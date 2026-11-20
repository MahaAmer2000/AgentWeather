
import pandas as pd

from src.processing.preprocessing import prepare_training_data

from src.models.xgboost_model import (
    train_xgboost_model,
    predict_xgboost_model,
    evaluate_xgboost_model
)

from src.models.baseline import naive_baseline

import numpy as np

from src.processing.preprocessing import (
    decode_wind_direction_prediction,
    calculate_wind_direction_errors
)

from src.models.baseline import wind_direction_baseline


# Load historical data
df = pd.read_csv("data/historical/combined_historical.csv")


# Prepare training datasets
prepared_datasets = prepare_training_data(df)


# Scalar targets (wind direction handled separately)
SCALAR_TARGETS = [
    "us_aqi",
    "temperature_2m",
    "precipitation",
    "precipitation_probability",
    "relative_humidity_2m",
    "uv_index",
    "wind_speed_10m"
]


# Store trained models and validation results
xgboost_models = {}
xgboost_results = {}


# Train and evaluate one XGBoost model per target
for target in SCALAR_TARGETS:

    print(f"\n{'=' * 60}")
    print(f"Training XGBoost model for: {target}")
    print(f"{'=' * 60}")

    # Select prepared dataset for current target
    target_data = prepared_datasets[target]

    # Train tuned XGBoost model
    model = train_xgboost_model(
        target_data["X_train_unscaled"],
        target_data["y_train"],
        target_data["X_val_unscaled"],
        target_data["y_val"]
    )

    # Generate validation predictions
    y_val_pred = predict_xgboost_model(
        model,
        target_data["X_val_unscaled"]
    )

    # Evaluate XGBoost model
    validation_results = evaluate_xgboost_model(
        target_data["y_val"],
        y_val_pred
    )

    # Evaluate naive baseline
    baseline_results = naive_baseline(
        target_data["X_val_unscaled"],
        target_data["y_val"],
        target
    )

    # Store trained model
    xgboost_models[target] = model

    # Store evaluation results
    xgboost_results[target] = {
        "XGBoost": validation_results,
        "Naive Baseline": baseline_results,
        "Best Iteration": model.best_iteration,
        "Best Validation RMSE": model.best_score
    }

    # Display results
    print("\nXGBoost Validation Results:")
    print(validation_results)

    print("\nNaive Baseline Validation Results:")
    print(baseline_results)

    print("\nBest iteration:", model.best_iteration)
    print("Best validation RMSE:", model.best_score)
    print("Number of trees used:", model.best_iteration + 1)


# Display summary of all seven models
print("\n\n" + "=" * 60)
print("XGBoost Validation Summary")
print("=" * 60)

for target, results in xgboost_results.items():

    print(f"\nTarget: {target}")

    print("XGBoost:", results["XGBoost"])
    print("Naive Baseline:", results["Naive Baseline"])
    print("Best Iteration:", results["Best Iteration"])
    print("Best Validation RMSE:", results["Best Validation RMSE"])


# Train XGBoost models for wind direction
print("\n" + "=" * 60)
print("Training XGBoost models for: wind_direction_10m")
print("=" * 60)

wind_data = prepared_datasets["wind_direction_10m"]

# Train sine model
wind_sin_model = train_xgboost_model(
    wind_data["X_train_unscaled"],
    wind_data["y_train"]["wind_direction_sin_target"],
    wind_data["X_val_unscaled"],
    wind_data["y_val"]["wind_direction_sin_target"]
)

# Train cosine model
wind_cos_model = train_xgboost_model(
    wind_data["X_train_unscaled"],
    wind_data["y_train"]["wind_direction_cos_target"],
    wind_data["X_val_unscaled"],
    wind_data["y_val"]["wind_direction_cos_target"]
)

# Predict sine and cosine values
sin_predictions = predict_xgboost_model(
    wind_sin_model,
    wind_data["X_val_unscaled"]
)

cos_predictions = predict_xgboost_model(
    wind_cos_model,
    wind_data["X_val_unscaled"]
)

# Combine predictions and convert them into degrees
wind_predictions = np.column_stack((
    sin_predictions,
    cos_predictions
))

wind_pred_degrees = decode_wind_direction_prediction(
    wind_predictions
)

# Convert actual encoded targets back into degrees
wind_true_encoded = np.column_stack((
    wind_data["y_val"]["wind_direction_sin_target"],
    wind_data["y_val"]["wind_direction_cos_target"]
))

wind_true_degrees = decode_wind_direction_prediction(
    wind_true_encoded
)

# Calculate circular errors
errors = calculate_wind_direction_errors(
    wind_true_degrees,
    wind_pred_degrees
)

wind_validation_results = {
    "Circular MAE (degrees)": np.mean(np.abs(errors)),
    "Circular RMSE (degrees)": np.sqrt(np.mean(errors ** 2))
}

# Calculate naive baseline using the direction 24 hours earlier
raw_wind = (
    df.assign(time=pd.to_datetime(df["time"]))
      .set_index("time")["wind_direction_10m"]
)

val_timestamps = pd.to_datetime(wind_data["val_timestamps"])

baseline_predictions = raw_wind.reindex(val_timestamps).to_numpy()

baseline_actuals = raw_wind.reindex(
    val_timestamps + pd.Timedelta(hours=24)
).to_numpy()

assert not pd.isna(baseline_predictions).any()
assert not pd.isna(baseline_actuals).any()

wind_baseline_results = wind_direction_baseline(
    baseline_predictions,
    baseline_actuals
)

# Store models and results
xgboost_models["wind_direction_sin"] = wind_sin_model
xgboost_models["wind_direction_cos"] = wind_cos_model

xgboost_results["wind_direction_10m"] = {
    "XGBoost": wind_validation_results,
    "Naive Baseline": wind_baseline_results
}

# Display results
print("\nWind Direction XGBoost Validation Results:")
print(wind_validation_results)

print("\nWind Direction Naive Baseline Results:")
print(wind_baseline_results)

