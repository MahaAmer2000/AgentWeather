
import numpy as np
import copy

import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset


class WeatherLSTM(nn.Module):

    def __init__(
        self,
        input_size=160,
        hidden_size=64,
        num_layers=2,
        output_size=1,
        dropout=0.2
    ):
        super(WeatherLSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):

        # Pass sequences through LSTM.
        lstm_output, (hidden_state, cell_state) = self.lstm(x)

        # Get output from the final timestep.
        last_output = lstm_output[:, -1, :]

        # Generate prediction.
        last_output = self.dropout(last_output)
        return self.fc(last_output)


def train_lstm_model(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    learning_rate=0.001,
    batch_size=64,
    epochs=50,
    patience=7
):
    """
    Train an LSTM model with validation monitoring
    and early stopping.

    Supports:
    - Scalar targets: (samples,) or (samples, 1)
    - Multi-output targets: (samples, outputs)

    Returns:
    - Best model
    - Training history
    """

    # --------------------------------------------------
    # 1. Set random seeds
    # --------------------------------------------------

    torch.manual_seed(42)
    np.random.seed(42)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # --------------------------------------------------
    # 2. Select available device
    # --------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Training device: {device}")

    model = model.to(device)

    # --------------------------------------------------
    # 3. Prepare target dimensions
    # --------------------------------------------------

    y_train = np.asarray(y_train, dtype=np.float32)
    y_val = np.asarray(y_val, dtype=np.float32)

    # Convert scalar targets into 2D arrays.
    if y_train.ndim == 1:
        y_train = y_train.reshape(-1, 1)

    if y_val.ndim == 1:
        y_val = y_val.reshape(-1, 1)

    if y_train.ndim != 2 or y_val.ndim != 2:
        raise ValueError(
            "Targets must have one or two dimensions."
        )

    if y_train.shape[1] != y_val.shape[1]:
        raise ValueError(
            "Training and validation target dimensions must match."
        )

    # Ensure model output size matches target dimensions.
    if model.fc.out_features != y_train.shape[1]:
        raise ValueError(
            f"Model output size ({model.fc.out_features}) "
            f"does not match target dimensions ({y_train.shape[1]})."
        )

    # --------------------------------------------------
    # 4. Convert data into tensors
    # --------------------------------------------------

    X_train_tensor = torch.tensor(
        X_train,
        dtype=torch.float32
    )

    y_train_tensor = torch.tensor(
        y_train,
        dtype=torch.float32
    )

    X_val_tensor = torch.tensor(
        X_val,
        dtype=torch.float32
    ).to(device)

    y_val_tensor = torch.tensor(
        y_val,
        dtype=torch.float32
    ).to(device)

    # --------------------------------------------------
    # 5. Create training batches
    # --------------------------------------------------

    train_dataset = TensorDataset(
        X_train_tensor,
        y_train_tensor
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    # --------------------------------------------------
    # 6. Define loss and optimizer
    # --------------------------------------------------

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
    model.parameters(),
    lr=learning_rate
    )

    # --------------------------------------------------
    # 7. Track training progress
    # --------------------------------------------------

    history = {
        "train_loss": [],
        "val_loss": []
    }

    best_val_loss = float("inf")

    best_weights = copy.deepcopy(
        model.state_dict()
    )

    best_epoch = 0
    patience_counter = 0

    # --------------------------------------------------
    # 8. Training loop
    # --------------------------------------------------

    for epoch in range(epochs):

        model.train()

        total_train_loss = 0.0

        for X_batch, y_batch in train_loader:

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            # Reset gradients.
            optimizer.zero_grad()

            # Forward pass.
            predictions = model(X_batch)

            # Calculate loss.
            loss = criterion(
                predictions,
                y_batch
            )

            # Stop if loss becomes invalid.
            if not torch.isfinite(loss):
                raise RuntimeError(
                    "Training loss became NaN or infinite."
                )

            # Backpropagation.
            loss.backward()

            # Clip gradients to improve training stability.
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            # Update model parameters.
            optimizer.step()

            total_train_loss += (
                loss.item() * len(X_batch)
            )

        train_loss = (
            total_train_loss / len(train_dataset)
        )

        # --------------------------------------------------
        # 9. Validation
        # --------------------------------------------------

        model.eval()

        with torch.no_grad():

            val_predictions = model(X_val_tensor)

            val_loss = criterion(
                val_predictions,
                y_val_tensor
            ).item()

        if not np.isfinite(val_loss):
            raise RuntimeError(
                "Validation loss became NaN or infinite."
            )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Train Loss: {train_loss:.6f} "
            f"Val Loss: {val_loss:.6f}"
        )

        # --------------------------------------------------
        # 10. Check validation improvement
        # --------------------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            best_weights = copy.deepcopy(
                model.state_dict()
            )

            best_epoch = epoch + 1
            patience_counter = 0

        else:

            patience_counter += 1

        # --------------------------------------------------
        # 11. Early stopping
        # --------------------------------------------------

        if patience_counter >= patience:

            print(
                f"\nEarly stopping at epoch {epoch + 1}"
            )

            break

    # --------------------------------------------------
    # 12. Restore best model
    # --------------------------------------------------

    model.load_state_dict(best_weights)

    model.eval()

    print(f"\nBest epoch: {best_epoch}")
    print(f"Best validation loss: {best_val_loss:.6f}")

    return model, history
