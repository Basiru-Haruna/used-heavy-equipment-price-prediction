"""Reproducible reporting helpers for model evaluation."""
import pandas as pd


def comparison_table(walkforward: pd.DataFrame) -> pd.DataFrame:
    """Derive, rather than manually enter, across-period MAE summaries."""
    return (walkforward.groupby('Model', as_index=False)
            .agg(WalkForwardAverageMAE=('MAE', 'mean'),
                 WalkForwardMAEStd=('MAE', 'std'),
                 Periods=('Experiment', 'nunique'))
            .sort_values('WalkForwardAverageMAE'))


def by_period_table(walkforward: pd.DataFrame) -> pd.DataFrame:
    return walkforward.pivot(index='Experiment', columns='Model', values='MAE')


def missing_age_error_report(predictions: pd.DataFrame) -> pd.DataFrame:
    """Descriptive comparison; not a causal effect of missingness."""
    data = predictions.copy()
    data['AbsoluteError'] = (data.ActualPrice - data.PredictedPrice).abs()
    data['AgeStatus'] = data.EquipmentAge.isna().map({True: 'Age Missing', False: 'Age Available'})
    summary = data.groupby('AgeStatus').agg(
        Transactions=('ActualPrice', 'size'),
        MeanActualPrice=('ActualPrice', 'mean'),
        MAE=('AbsoluteError', 'mean'),
    )
    summary['MeanNormalizedMAEPercent'] = 100 * summary['MAE'] / summary['MeanActualPrice']
    return summary
