"""Utility functions for vangja time series models.

This module provides helper functions for data processing and evaluation
of time series models.

Functions
---------
get_group_definition
    Assign group codes to different time series based on pooling type.
metrics
    Calculate evaluation metrics for time series predictions.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    root_mean_squared_error,
)

from vangja.types import PoolType


def get_group_definition(
    data: pd.DataFrame, pool_type: PoolType
) -> tuple[np.ndarray, int, dict[int, str]]:
    """Assign group codes to different series based on pooling type.

    This function processes a multi-series dataframe and assigns integer codes
    to each unique series. The behavior depends on the pool_type parameter:

    - "complete": All series share a single group (code 0)
    - "partial" or "individual": Each unique series gets its own code

    Parameters
    ----------
    data : pd.DataFrame
        A pandas dataframe that must at least have columns ds (predictor), y
        (target) and series (name of time series).
    pool_type : PoolType
        Type of pooling performed when sampling. One of "complete", "partial",
        or "individual".

    Returns
    -------
    group : np.ndarray
        Array of integer group codes, one for each row in data.
    n_groups : int
        Number of unique groups.
    group_mapping : dict[int, str]
        Dictionary mapping group codes (int) to series names (str).

    Examples
    --------
    >>> import pandas as pd
    >>> data = pd.DataFrame({
    ...     'ds': pd.date_range('2020-01-01', periods=6),
    ...     'y': [1, 2, 3, 4, 5, 6],
    ...     'series': ['A', 'A', 'A', 'B', 'B', 'B']
    ... })
    >>> group, n_groups, mapping = get_group_definition(data, 'partial')
    >>> print(n_groups)
    2
    >>> print(mapping)
    {0: 'A', 1: 'B'}
    """
    pool_cols = "series"
    if pool_type == "complete":
        group = np.zeros(len(data), dtype="int")
        group_mapping = {0: data.iloc[0][pool_cols]}
        n_groups = 1
    else:
        data[pool_cols] = pd.Categorical(data[pool_cols])
        group = data[pool_cols].cat.codes.values
        group_mapping = dict(enumerate(data[pool_cols].cat.categories))
        n_groups = data[pool_cols].nunique()

    return group, n_groups, group_mapping


def metrics(
    y_true: pd.DataFrame, future: pd.DataFrame, pool_type: PoolType
) -> pd.DataFrame:
    """Calculate evaluation metrics for time series predictions.

    Computes Mean Squared Error (MSE), Root Mean Squared Error (RMSE),
    Mean Absolute Error (MAE), and Mean Absolute Percentage Error (MAPE)
    for each time series in the dataset.

    Parameters
    ----------
    y_true : pd.DataFrame
        A pandas dataframe containing the true values for the inference period
        that must at least have columns ds (predictor), y (target) and series
        (name of time series).
    future : pd.DataFrame
        Pandas dataframe containing the timestamps and predictions. Must have
        columns named 'yhat_{group_code}' for each group.
    pool_type : PoolType
        Type of pooling performed when sampling. Used to determine group
        assignments in y_true.

    Returns
    -------
    pd.DataFrame
        A dataframe with series names as index and columns for each metric:
        'mse', 'rmse', 'mae', 'mape'.

    Examples
    --------
    >>> from vangja import LinearTrend
    >>> from vangja.utils import metrics
    >>> model = LinearTrend()
    >>> model.fit(train_data)
    >>> future = model.predict(horizon=30)
    >>> evaluation = metrics(test_data, future, pool_type="complete")
    >>> print(evaluation)
              mse     rmse      mae     mape
    series1  25.3    5.03    4.21    0.082

    Notes
    -----
    The predictions in `future` are matched to the last `len(y_true)` rows
    for each group's prediction column. This assumes the prediction horizon
    covers the test period.
    """
    metrics_dict = {"mse": {}, "rmse": {}, "mae": {}, "mape": {}}
    test_group, _, test_groups_ = get_group_definition(y_true, pool_type)
    for group_code, group_name in test_groups_.items():
        group_idx = test_group == group_code
        y = y_true["y"][group_idx]
        yhat = future[f"yhat_{group_code}"][-len(y) :]
        metrics_dict["mse"][group_name] = mean_squared_error(y, yhat)
        metrics_dict["rmse"][group_name] = root_mean_squared_error(y, yhat)
        metrics_dict["mae"][group_name] = mean_absolute_error(y, yhat)
        metrics_dict["mape"][group_name] = mean_absolute_percentage_error(y, yhat)

    return pd.DataFrame(metrics_dict)
