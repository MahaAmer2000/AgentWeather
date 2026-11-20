import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)



def naive_baseline(X_test, y_test, target_column):

    # Get the current observed values as predictions
    y_pred = X_test[target_column].to_numpy()

    # Get the actual values 24 hours ahead
    y_true = y_test.to_numpy()

    # Calculate evaluation metrics
    mae = mean_absolute_error(y_true, y_pred)

    rmse = np.sqrt(
        mean_squared_error(y_true, y_pred)
    )

    r2 = r2_score(y_true, y_pred)

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }



def wind_direction_baseline(y_pred, y_true):

    # Convert inputs to NumPy arrays
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)

    # Calculate circular errors
    errors = ((y_pred - y_true + 180) % 360) - 180

    # Calculate circular MAE and RMSE
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors ** 2))

    return {
        "MAE": mae,
        "RMSE": rmse
    }


