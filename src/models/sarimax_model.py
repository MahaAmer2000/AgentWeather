
import numpy as np
import pandas as pd

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# --------------------------------------------------
# 1. Prepare target time series
# --------------------------------------------------

def prepare_sarimax_data(df, target_column):

    df = df.copy()

    # Sort observations chronologically
    df["time"] = pd.to_datetime(df["time"])

    df = df.sort_values("time").reset_index(drop=True)

    # Select the requested target
    target_series = df.set_index("time")[target_column]

    # Ensure hourly frequency
    target_series = target_series.asfreq("h")

    return target_series


# --------------------------------------------------
# 2. Train SARIMAX model
# --------------------------------------------------

def train_sarimax_model(
    train,
    order=(1, 0, 1),
    seasonal_order=(0, 0, 0, 0)
):

    model = SARIMAX(
        train,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    fitted_model = model.fit(disp=False)

    return fitted_model


# --------------------------------------------------
# 3. Generate forecasts
# --------------------------------------------------

def predict_sarimax_model(model, steps=24):

    predictions = model.get_forecast(steps=steps)

    return predictions


# --------------------------------------------------
# 4. Evaluate scalar target predictions
# --------------------------------------------------

def evaluate_sarimax_model(y_true, y_pred):

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


# --------------------------------------------------
# 5. Rolling-origin 24-hour-ahead predictions
# --------------------------------------------------

def rolling_sarimax_predictions(
    model,
    full_series,
    forecast_origins
):

    predictions = []

    last_observed_time = model.data.row_labels[-1]

    for timestamp in forecast_origins:

        # Get observations available since the last update
        new_observations = full_series.loc[
            last_observed_time + pd.Timedelta(hours=1):timestamp
        ]

        # Update model state without retraining parameters
        if len(new_observations) > 0:

            model = model.append(
                new_observations,
                refit=False
            )

        # Forecast the next 24 hours
        forecast = predict_sarimax_model(
            model,
            steps=24
        )

        # Keep only the 24th-hour prediction
        predictions.append(
            float(forecast.predicted_mean.iloc[-1])
        )

        last_observed_time = timestamp

    return np.array(predictions)


# --------------------------------------------------
# 6. Check stationarity
# --------------------------------------------------

def check_stationarity(series):

    result = adfuller(
        series.dropna(),
        result_object=True
    )

    adf_statistic = result.statistic

    p_value = result.pvalue

    print("ADF Statistic:", adf_statistic)
    print("p-value:", p_value)

    if p_value <= 0.05:

        print("Conclusion: The series is stationary.")

    else:

        print("Conclusion: The series is non-stationary.")
