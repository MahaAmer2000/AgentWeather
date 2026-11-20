import numpy as np
import pandas as pd

from xgboost import XGBRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)



def train_xgboost_model(X_train, y_train, X_val, y_val):

    model = XGBRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=1.0,
        min_child_weight=1,
        gamma=0,
        reg_lambda=5,
        objective="reg:squarederror",
        eval_metric="rmse",
        early_stopping_rounds=50,
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    return model


def predict_xgboost_model(model, X):
    predictions = model.predict(X)
    return predictions

def evaluate_xgboost_model(y_true, y_pred):

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