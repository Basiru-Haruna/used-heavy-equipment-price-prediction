"""Reproducible development benchmarks based on the original equipment notebook.

Historical windows have been examined during development; scores are NOT
untouched holdout or prospective model performance estimates.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from .preprocessing import CATEGORICAL, NUMERICAL, PREDICTORS, catboost_features

WINDOWS = [
    {"Experiment": "A", "TrainStart": 2001, "TrainEnd": 2005, "ValidStart": 2006, "ValidEnd": 2007},
    {"Experiment": "B", "TrainStart": 2003, "TrainEnd": 2007, "ValidStart": 2008, "ValidEnd": 2009},
    {"Experiment": "C", "TrainStart": 2005, "TrainEnd": 2009, "ValidStart": 2010, "ValidEnd": 2011},
]


def rf_pipeline() -> Pipeline:
    """Final five-feature Random Forest pipeline from the development notebook."""
    preprocessor = ColumnTransformer([
        ("numerical", SimpleImputer(strategy="median", add_indicator=True), NUMERICAL),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    regressor = RandomForestRegressor(
        n_estimators=50, max_depth=24, min_samples_leaf=5,
        max_features=0.5, random_state=42, n_jobs=-1,
    )
    return Pipeline([("preprocessor", preprocessor), ("model", regressor)])


def catboost_model() -> CatBoostRegressor:
    """Final CatBoost configuration from the original development notebook."""
    return CatBoostRegressor(
        iterations=500, depth=6, learning_rate=0.05,
        loss_function="RMSE", random_seed=42, task_type="CPU",
        verbose=False, allow_writing_files=False,
    )


def error_metrics(actual, predicted) -> dict:
    """Signed error = actual - predicted; positive indicates underestimation."""
    y = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    e = y - p
    ae = np.abs(e)
    valid_percent = y != 0
    return {
        "MAE": float(mean_absolute_error(y, p)),
        "MedianAE": float(np.median(ae)),
        "P90AE": float(np.percentile(ae, 90)),
        "MeanSignedError": float(e.mean()),
        "UnderestimationRatePercent": float(np.mean(e > 0) * 100),
        "ErrorAbove10000Percent": float(np.mean(ae > 10000) * 100),
        "MAPEPercent": float(np.mean(ae[valid_percent] / np.abs(y[valid_percent])) * 100) if valid_percent.any() else np.nan,
    }


def run_walkforward(cleaned: pd.DataFrame, *, include_predictions=False):
    """Train both benchmarks in each original five-year chronological window.

    Expensive on the full dataset. No validation-target fitting or selection occurs
    inside this function; preprocessing for Random Forest is fit to training rows.
    """
    results = []
    predictions = {}
    for window in WINDOWS:
        name = window["Experiment"]
        train = cleaned.loc[cleaned["SaleYear"].between(window["TrainStart"], window["TrainEnd"])].copy()
        valid = cleaned.loc[cleaned["SaleYear"].between(window["ValidStart"], window["ValidEnd"])].copy()
        if train.empty or valid.empty:
            raise ValueError(f"No records in training or validation for experiment {name}")
        if not (train["saledate"].max() < valid["saledate"].min()):
            raise ValueError(f"Chronological overlap in experiment {name}")
        x_train = catboost_features(train)
        x_valid = catboost_features(valid)
        y_train, y_valid = train["SalePrice"], valid["SalePrice"]
        for model_name in ("Random Forest", "CatBoost"):
            if model_name == "Random Forest":
                model = rf_pipeline()
                model.fit(x_train[PREDICTORS], y_train)
            else:
                model = catboost_model()
                model.fit(x_train[PREDICTORS], y_train, cat_features=CATEGORICAL)
            pred = np.asarray(model.predict(x_valid[PREDICTORS]))
            results.append({
                "Experiment": name, "Model": model_name,
                "TrainingPeriod": f'{window["TrainStart"]}-{window["TrainEnd"]}',
                "ValidationPeriod": f'{window["ValidStart"]}-{window["ValidEnd"]}',
                "TrainingTransactions": len(train), "ValidationTransactions": len(valid),
                **error_metrics(y_valid, pred),
            })
            if include_predictions:
                predictions[(name, model_name)] = pd.DataFrame({
                    "OriginalIndex": valid.index,
                    "ActualPrice": y_valid.to_numpy(), "PredictedPrice": pred,
                    "SignedError": y_valid.to_numpy() - pred,
                    "ProductGroup": valid["ProductGroup"].to_numpy(),
                    "EquipmentAge": valid["EquipmentAge"].to_numpy(),
                })
    table = pd.DataFrame(results)
    return (table, predictions) if include_predictions else table
