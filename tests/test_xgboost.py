
import pandas as pd
import numpy as np

from src.processing.preprocessing import (
    prepare_training_data,
    decode_wind_direction_prediction,
    calculate_wind_direction_errors
)

from src.models.xgboost_model import (
    train_xgboost_model,
    predict_xgboost_model,
    evaluate_xgboost_model
)

from src.models.baseline import (
    naive_baseline,
    wind_direction_baseline
)


# --------------------------------------------------
# 1. Load historical data
# --------------------------------------------------

df = pd.read_csv(
    "data/historical/combined_historical.csv"
)


# --------------------------------------------------
# 2. Prepare training datasets
# --------------------------------------------------

prepared_datasets = prepare_training_data(df)


# --------------------------------------------------
# 3. Configuration
# --------------------------------------------------

SCALAR_TARGETS = [
    "us_aqi",
    "temperature_2m",
    "precipitation",
    "precipitation_probability",
    "relative_humidity_2m",
    "uv_index",
    "wind_speed_10m"
]


# Store models and results
xgboost_models = {}
xgboost_results = {}


# --------------------------------------------------
# 4. Train and evaluate scalar targets
# --------------------------------------------------

for target in SCALAR_TARGETS:

    print("\n" + "=" * 60)
    print(f"Training XGBoost model for: {target}")
    print("=" * 60)

    target_data = prepared_datasets[target]

    # Train model
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

    # Evaluate model
    validation_metrics = evaluate_xgboost_model(
        target_data["y_val"],
        y_val_pred
    )

    # Evaluate naive baseline for reference
    baseline_metrics = naive_baseline(
        target_data["X_val_unscaled"],
        target_data["y_val"],
        target
    )

    # Store model
    xgboost_models[target] = model

    # Store metrics
    xgboost_results[target] = {
        "XGBoost": validation_metrics,
        "Naive Baseline": baseline_metrics,
        "Best Iteration": model.best_iteration,
        "Best Validation RMSE": model.best_score
    }

    # Display results
    print("\nXGBoost Validation Results:")
    print(validation_metrics)

    print("\nNaive Baseline Validation Results:")
    print(baseline_metrics)

    print("\nBest iteration:", model.best_iteration)
    print("Best validation RMSE:", model.best_score)


# --------------------------------------------------
# 5. Train and evaluate wind direction
# --------------------------------------------------

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

# Generate sine and cosine predictions
sin_predictions = predict_xgboost_model(
    wind_sin_model,
    wind_data["X_val_unscaled"]
)

cos_predictions = predict_xgboost_model(
    wind_cos_model,
    wind_data["X_val_unscaled"]
)

# Decode predictions into degrees
wind_predictions = np.column_stack((
    sin_predictions,
    cos_predictions
))

wind_pred_degrees = decode_wind_direction_prediction(
    wind_predictions
)

# Decode actual targets into degrees
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

wind_validation_metrics = {
    "MAE": np.mean(np.abs(errors)),
    "RMSE": np.sqrt(np.mean(errors ** 2)),
    "R2": None
}

# Calculate wind direction naive baseline
raw_wind = (
    df.assign(time=pd.to_datetime(df["time"]))
      .set_index("time")["wind_direction_10m"]
)

val_timestamps = pd.to_datetime(
    wind_data["val_timestamps"]
)

baseline_predictions = raw_wind.reindex(
    val_timestamps
).to_numpy()

baseline_actuals = raw_wind.reindex(
    val_timestamps + pd.Timedelta(hours=24)
).to_numpy()

assert not pd.isna(baseline_predictions).any()
assert not pd.isna(baseline_actuals).any()

wind_baseline_metrics = wind_direction_baseline(
    baseline_predictions,
    baseline_actuals
)

# Store wind models
xgboost_models["wind_direction_sin"] = wind_sin_model
xgboost_models["wind_direction_cos"] = wind_cos_model

# Store wind results
xgboost_results["wind_direction_10m"] = {
    "XGBoost": wind_validation_metrics,
    "Naive Baseline": wind_baseline_metrics,
    "Best Iteration": None
}

print("\nWind Direction XGBoost Validation Results:")
print(wind_validation_metrics)

print("\nWind Direction Naive Baseline Results:")
print(wind_baseline_metrics)


# --------------------------------------------------
# 6. Create consolidated XGBoost results
# --------------------------------------------------

results = []

for target, metrics in xgboost_results.items():

    xgb_metrics = metrics["XGBoost"]

    results.append({
        "Target": target,
        "MAE": xgb_metrics["MAE"],
        "RMSE": xgb_metrics["RMSE"],
        "R2": xgb_metrics["R2"],
        "Best Iteration": metrics["Best Iteration"]
    })


results_df = pd.DataFrame(results)


# --------------------------------------------------
# 7. Display final validation results
# --------------------------------------------------

print("\n" + "=" * 75)
print("FINAL XGBOOST VALIDATION RESULTS")
print("=" * 75)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# --------------------------------------------------
# 8. Save validation results
# --------------------------------------------------

results_df.to_csv(
    "results/xgboost_validation.csv",
    index=False
)

print("\nXGBoost validation results saved successfully.")
