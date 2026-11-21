
import pandas as pd
import numpy as np
import torch

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from src.processing.lstm_preprocessing import prepare_lstm_data
from src.models.lstm_model import (
    WeatherLSTM,
    train_lstm_model
)


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

TARGET_COLUMNS = [
    "us_aqi",
    "temperature_2m",
    "precipitation",
    "precipitation_probability",
    "relative_humidity_2m",
    "uv_index",
    "wind_speed_10m",
    "wind_direction_10m"
]

SEQUENCE_LENGTH = 24

HIDDEN_SIZE = 64
NUM_LAYERS = 1
DROPOUT = 0.2

LEARNING_RATE = 0.0001
BATCH_SIZE = 32
EPOCHS = 30
PATIENCE = 7


# --------------------------------------------------
# 2. Load historical data
# --------------------------------------------------

df = pd.read_csv(
    "data/historical/combined_historical.csv"
)


# --------------------------------------------------
# 3. Store results
# --------------------------------------------------

results = []


# --------------------------------------------------
# 4. Train and evaluate each target
# --------------------------------------------------

for target_column in TARGET_COLUMNS:

    print("\n" + "=" * 60)
    print(f"Training LSTM for: {target_column}")
    print("=" * 60)

    # Prepare target-specific dataset.
    dataset = prepare_lstm_data(
        df,
        target_column=target_column,
        sequence_length=SEQUENCE_LENGTH
    )

    X_train = dataset["X_train"]
    y_train = dataset["y_train"]

    X_val = dataset["X_val"]
    y_val = dataset["y_val"]

    print("\nDataset Shapes")
    print("----------------")
    print("Training:", X_train.shape, y_train.shape)
    print("Validation:", X_val.shape, y_val.shape)

    # Determine model output dimensions.
    output_size = (
        2 if target_column == "wind_direction_10m"
        else 1
    )

    # Set seeds before model initialization.
    torch.manual_seed(42)
    np.random.seed(42)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # Initialize a fresh model for this target.
    model = WeatherLSTM(
        input_size=X_train.shape[2],
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        output_size=output_size,
        dropout=DROPOUT
    )

    print("\nModel Parameters")
    print("----------------")
    print(
        "Total parameters:",
        sum(p.numel() for p in model.parameters())
    )

    # Train model.
    trained_model, history = train_lstm_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        patience=PATIENCE
    )

    # Verify training.
    assert len(history["train_loss"]) > 0
    assert len(history["train_loss"]) == len(history["val_loss"])

    assert np.isfinite(history["train_loss"]).all()
    assert np.isfinite(history["val_loss"]).all()

    best_epoch = int(np.argmin(history["val_loss"]) + 1)

    print("\nTraining Verification")
    print("---------------------")
    print("Completed epochs:", len(history["train_loss"]))
    print("Best epoch:", best_epoch)
    print("Best validation loss:", min(history["val_loss"]))

    # --------------------------------------------------
    # 5. Generate validation predictions
    # --------------------------------------------------

    trained_model.eval()

    device = next(trained_model.parameters()).device

    X_val_tensor = torch.tensor(
        X_val,
        dtype=torch.float32
    ).to(device)

    with torch.no_grad():
        predictions_scaled = (
            trained_model(X_val_tensor)
            .cpu()
            .numpy()
        )

    # --------------------------------------------------
    # 6. Calculate evaluation metrics
    # --------------------------------------------------

    if target_column == "wind_direction_10m":

        # Decode predicted sine/cosine into degrees.
        predicted_angles = np.degrees(
            np.arctan2(
                predictions_scaled[:, 0],
                predictions_scaled[:, 1]
            )
        ) % 360

        # Decode actual sine/cosine into degrees.
        actual_encoded = np.asarray(
            dataset["y_val_original"]
        )

        actual_angles = np.degrees(
            np.arctan2(
                actual_encoded[:, 0],
                actual_encoded[:, 1]
            )
        ) % 360

        # Calculate shortest signed angular difference.
        angular_errors = (
            predicted_angles - actual_angles + 180
        ) % 360 - 180

        mae = np.mean(np.abs(angular_errors))

        rmse = np.sqrt(
            np.mean(angular_errors ** 2)
        )

        results.append({
            "Target": target_column,
            "MAE": mae,
            "RMSE": rmse,
            "R2": None,
            "Best Epoch": best_epoch
        })

    else:

        # Convert predictions back to original units.
        val_predictions = (
            dataset["target_scaler"]
            .inverse_transform(predictions_scaled)
            .ravel()
        )

        actual_values = np.asarray(
            dataset["y_val_original"]
        ).ravel()

        mae = mean_absolute_error(
            actual_values,
            val_predictions
        )

        rmse = np.sqrt(
            mean_squared_error(
                actual_values,
                val_predictions
            )
        )

        r2 = r2_score(
            actual_values,
            val_predictions
        )

        results.append({
            "Target": target_column,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            "Best Epoch": best_epoch
        })

    print("\nValidation Results")
    print("------------------")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")

    if target_column != "wind_direction_10m":
        print(f"R2:   {r2:.4f}")
    else:
        print("Metrics calculated using circular angular errors.")


# --------------------------------------------------
# 7. Display final results
# --------------------------------------------------

results_df = pd.DataFrame(results)

print("\n" + "=" * 75)
print("FINAL LSTM VALIDATION RESULTS")
print("=" * 75)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)
# Save validation results.
results_df.to_csv(
    "results/lstm_validation.csv",
    index=False
)

print("\nLSTM validation results saved successfully.")
