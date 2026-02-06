"""Utility functions for vangja time series models.

This module provides helper functions for data processing and evaluation
of time series models.

Functions
---------
remove_random_gaps
    Remove random contiguous intervals from a time series to simulate missing data.
get_group_definition
    Assign group codes to different time series based on pooling type.
filter_predictions_by_series
    Filter predictions to dates relevant to a specific series.
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


def remove_random_gaps(
    df: pd.DataFrame, n_gaps: int = 4, gap_fraction: float = 0.2
) -> pd.DataFrame:
    """Remove random continuous intervals (gaps) from a time series DataFrame.

    Creates realistic missing-data scenarios by removing ``n_gaps``
    non-overlapping contiguous blocks from the data. Each block removes
    approximately ``gap_fraction`` of the total data points.

    Parameters
    ----------
    df : pd.DataFrame
        A time series DataFrame. Must have at least a ``ds`` column.
    n_gaps : int, default 4
        Number of contiguous intervals to remove.
    gap_fraction : float, default 0.2
        Fraction of total data points removed per gap.

    Returns
    -------
    pd.DataFrame
        A copy of the input DataFrame with the specified gaps removed,
        index reset.

    Raises
    ------
    ValueError
        If the total number of points to remove exceeds the length of the
        DataFrame.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'ds': pd.date_range('2020-01-01', periods=100),
    ...     'y': range(100),
    ... })
    >>> df_with_gaps = remove_random_gaps(df, n_gaps=2, gap_fraction=0.1)
    >>> len(df_with_gaps) < len(df)
    True
    """
    n = len(df)
    gap_size = int(n * gap_fraction)
    total_gap_size = n_gaps * gap_size

    if total_gap_size >= n:
        raise ValueError(
            f"Cannot remove {n_gaps} gaps of {gap_fraction*100}% each from data"
        )

    # Generate non-overlapping gap start positions
    available_indices = list(range(n - gap_size))
    gap_starts = []

    for i in range(n_gaps):
        if not available_indices:
            break
        start = np.random.choice(available_indices)
        gap_starts.append(start)
        # Remove indices that would overlap with this gap
        available_indices = [
            idx
            for idx in available_indices
            if idx >= start + gap_size or idx + gap_size <= start
        ]

    # Create mask for rows to keep
    keep_mask = np.ones(n, dtype=bool)
    for start in gap_starts:
        keep_mask[start : start + gap_size] = False

    return df[keep_mask].reset_index(drop=True)


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


def filter_predictions_by_series(
    future: pd.DataFrame,
    series_data: pd.DataFrame,
    yhat_col: str = "yhat_0",
    horizon: int = 0,
) -> pd.DataFrame:
    """Filter predictions to only include dates relevant to a specific series.

    When fitting multiple series simultaneously with different date ranges,
    the predict() method generates predictions for the entire combined time
    range. This function filters predictions to only include dates within a
    specific series' range, which is essential for correct metric calculation
    and plotting.

    Parameters
    ----------
    future : pd.DataFrame
        Predictions dataframe from model.predict() containing 'ds' and yhat columns.
    series_data : pd.DataFrame
        The original data for a specific series (train + test combined, or just
        the portion you want to filter to). Must have 'ds' column.
    yhat_col : str, default "yhat_0"
        The name of the prediction column to include in the output.
    horizon : int, default 0
        Additional days beyond the series' max date to include (for forecast period).

    Returns
    -------
    pd.DataFrame
        Filtered predictions with columns ['ds', 'yhat_0'] containing only dates
        within the series' range plus the specified horizon.

    Examples
    --------
    >>> # After fitting a multi-series model
    >>> future_combined = model.predict(horizon=365)
    >>> # Filter to only Air Passengers' relevant dates
    >>> future_passengers = filter_predictions_by_series(
    ...     future_combined,
    ...     air_passengers,  # full dataset (train + test)
    ...     yhat_col=f"yhat_{passengers_group}",
    ...     horizon=365
    ... )
    """
    date_min = series_data["ds"].min()
    date_max = series_data["ds"].max()

    filtered = future[["ds", yhat_col]].copy()
    filtered.columns = ["ds", "yhat_0"]
    filtered = filtered[
        (filtered["ds"] >= date_min)
        & (filtered["ds"] <= date_max + pd.Timedelta(days=horizon))
    ]
    return filtered.reset_index(drop=True)


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
        columns named 'yhat_{group_code}' for each group. The 'ds' column is
        used to match predictions to test data by date.
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
    Predictions are matched to test data by merging on the 'ds' column. This
    correctly handles cases where predictions are at a different frequency
    than the test data (e.g., daily predictions vs monthly test data).
    """
    # Copy y_true and add a 'series' column if not present
    processed_y_true = y_true.copy()
    if "series" not in processed_y_true.columns:
        processed_y_true["series"] = "series"

    # Ensure ds columns are datetime for proper merging
    processed_y_true["ds"] = pd.to_datetime(processed_y_true["ds"])
    future = future.copy()
    future["ds"] = pd.to_datetime(future["ds"])

    metrics_dict = {"mse": {}, "rmse": {}, "mae": {}, "mape": {}}
    test_group, _, test_groups_ = get_group_definition(processed_y_true, pool_type)
    for group_code, group_name in test_groups_.items():
        group_idx = test_group == group_code
        y_true_group = processed_y_true[group_idx][["ds", "y"]]

        # Merge predictions with test data on ds to correctly align dates
        merged = y_true_group.merge(
            future[["ds", f"yhat_{group_code}"]],
            on="ds",
            how="inner",
        )

        if len(merged) == 0:
            raise ValueError(
                f"No matching dates found between test data and predictions for "
                f"series '{group_name}'. Ensure predictions cover the test period."
            )

        y = merged["y"]
        yhat = merged[f"yhat_{group_code}"]
        metrics_dict["mse"][group_name] = mean_squared_error(y, yhat)
        metrics_dict["rmse"][group_name] = root_mean_squared_error(y, yhat)
        metrics_dict["mae"][group_name] = mean_absolute_error(y, yhat)
        metrics_dict["mape"][group_name] = mean_absolute_percentage_error(y, yhat)

    return pd.DataFrame(metrics_dict)
