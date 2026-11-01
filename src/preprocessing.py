"""Deterministic data preparation for historical equipment auction transactions.

Cleaning logic is based on the original Blue Book for Bulldozers project notebook.
No statistics are learned from future validation data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PREDICTORS = [
    "EquipmentAge", "MachineHours_Clean", "ProductGroup", "ModelID", "ProductSize"
]
CATEGORICAL = ["ProductGroup", "ModelID", "ProductSize"]
NUMERICAL = ["EquipmentAge", "MachineHours_Clean"]
REQUIRED = ["saledate", "YearMade", "MachineHoursCurrentMeter", "SalePrice", *CATEGORICAL]


def clean_transactions(raw: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy, preserving originals and keeping all transactions.

    Rules reproduced from the original analysis:
    * Manufacturing year 1000 and manufacturing years after sale are invalid.
    * Equipment age is sale year minus valid manufacturing year.
    * Operating hours above (equipment age + 1) * 365.25 * 24 are invalid.
    * Unknown-age transactions cannot be verified by this physical-hours rule.
    * Missing and zero operating hours stay distinct.
    """
    missing_columns = set(REQUIRED) - set(raw.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    data = raw.copy()
    data["saledate"] = pd.to_datetime(data["saledate"], errors="coerce")
    data["SaleYear"] = data["saledate"].dt.year
    data["YearMade_Clean"] = pd.to_numeric(data["YearMade"], errors="coerce").astype(float)
    invalid_year = (data["YearMade_Clean"] == 1000) | (
        data["YearMade_Clean"] > data["SaleYear"]
    )
    data["YearMade_Invalid_Flag"] = invalid_year.fillna(False).astype("int8")
    data.loc[invalid_year, "YearMade_Clean"] = np.nan
    data["EquipmentAge"] = data["SaleYear"] - data["YearMade_Clean"]

    original_hours = pd.to_numeric(data["MachineHoursCurrentMeter"], errors="coerce")
    data["MaxPossibleHours"] = (data["EquipmentAge"] + 1) * 365.25 * 24
    invalid_hours = original_hours > data["MaxPossibleHours"]
    data["Hours_PhysicalRule_Flag"] = invalid_hours.fillna(False).astype("int8")
    data["MachineHours_Clean"] = original_hours.mask(invalid_hours)
    data["HoursStatus"] = np.select(
        [original_hours.isna(), original_hours.eq(0), original_hours.gt(0)],
        ["Missing", "Zero", "Positive"], default="Other"
    )
    return data


def catboost_features(cleaned: pd.DataFrame) -> pd.DataFrame:
    """Create model-ready predictors without looking at validation targets.

    The CatBoost model used missing categorical labels and preserved numerical NaNs.
    """
    x = cleaned[PREDICTORS].copy()
    for col in CATEGORICAL:
        x[col] = x[col].fillna("Missing").astype(str)
    for col in NUMERICAL:
        x[col] = pd.to_numeric(x[col], errors="coerce")
    return x
