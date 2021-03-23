import numpy as np
import pandas as pd
import optuna as op
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def get_sample_weights(df: pd.DataFrame, method: str = 'proportional', **kwargs) -> np.array:
    if method == 'uniform':
        sw = np.ones_like(df.index)
    elif method == 'splits':
        dates = {
            df['date'].iloc[0].strftime('%Y-%m-%d'): 1,
            (df['date'].iloc[-1] + pd.DateOffset(days=1)).date().strftime('%Y-%m-%d'): 1
        }
        dates = dates if kwargs.get('dates') is None else {**dates, **kwargs['dates']}
        index = sorted(dates.keys())
        index = [pd.Series(dates[sd], pd.date_range(sd, ed, closed=None)) for sd, ed in zip(index[:-1], index[1:])]
        sw = pd.concat(index).values
    elif method == 'proportional':
        column = 'n_severe' if kwargs.get('column') is None else kwargs['column']
        sample_weight = (df[column] - df[column].mean()) / df[column].std()
        sample_weight = sample_weight - sample_weight.min() + 1
        sw = sample_weight.values
    else:
        raise ValueError(f'{method} is not a supported method')
    return sw


def get_custom_estimator(loss: str = 'mse', sample_weight: np.array = None, normalize: bool = True):
    if loss.lower() == 'mse':
        loss = mean_squared_error
    elif loss.lower() == 'mae':
        loss = mean_absolute_error
    elif loss.lower() == 'r2':
        loss = lambda y_true, y_pred, sample_weight: 1 - r2_score(y_true, y_pred, sample_weight=sample_weight)
    else:
        raise ValueError(f'{loss} is not a supported loss')

    def func(y_true, y_pred):
        if normalize:
            factor = np.array(y_true).max()
            y_true = np.array(y_true) / factor
            y_pred = np.array(y_pred) / factor
        return loss(y_true, y_pred, sample_weight=sample_weight)

    return func


def compute_violation(pars: dict, orderings: list) -> float:
    violation = 1.
    for ordering in orderings:
        ordering = [pars[attribute] for attribute in ordering]
        for greater, lower in zip(ordering[:-1], ordering[1:]):
            violation *= max(1, lower / greater)
    return violation


def fixed_param(trial: op.Trial, name: str, value: float = 0.) -> float:
    return trial.suggest_float(name, value, value)
