"""Small, synthetic-data unit checks; not a substitute for full dataset validation."""
import numpy as np
import pandas as pd
from src.preprocessing import clean_transactions, catboost_features
from src.modeling import error_metrics


def sample():
    return pd.DataFrame({
        'saledate': ['2010-01-15','2010-01-15','2010-01-15','2010-01-15'],
        'YearMade': [1000,2012,2005,2009],
        'MachineHoursCurrentMeter': [100,50,0,1_000_000],
        'SalePrice': [10000,20000,12000,15000],
        'ProductGroup': ['MG','BL','SSL','WL'],
        'ModelID': [10,11,12,13],
        'ProductSize': [None,'Small',None,'Large'],
    })


def test_quality_rules():
    cleaned = clean_transactions(sample())
    assert len(cleaned) == 4
    assert cleaned.EquipmentAge.isna().sum() == 2
    assert cleaned.MachineHours_Clean.iloc[2] == 0
    assert np.isnan(cleaned.MachineHours_Clean.iloc[3])
    # Accept pandas 3 string dtype as well as legacy object storage.
    model_ids = catboost_features(cleaned)["ModelID"]
    assert model_ids.map(lambda value: isinstance(value, str)).all()
    assert catboost_features(cleaned).ProductSize.iloc[0] == 'Missing'


def test_prediction_error_direction():
    report = error_metrics([100, 200], [80, 220])
    assert report['MeanSignedError'] == 0
    assert report['MAE'] == 20
