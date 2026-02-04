"""Functions for loading real-world time series datasets.

This module provides convenience functions for loading commonly used
time series datasets in the format expected by vangja (columns: ds, y).
"""

import pandas as pd


def load_air_passengers() -> pd.DataFrame:
    """Load the Air Passengers dataset.

    The Air Passengers dataset is a classic time series dataset containing
    monthly totals of international airline passengers from January 1949 to
    December 1960 (144 observations).

    This dataset exhibits:
    - Clear upward trend
    - Strong yearly seasonality
    - Multiplicative seasonality (variance increases with level)

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - `ds`: datetime, monthly timestamps from 1949-01 to 1960-12
        - `y`: float, number of passengers (in thousands)

    Examples
    --------
    >>> from vangja.datasets import load_air_passengers
    >>> df = load_air_passengers()
    >>> print(f"Shape: {df.shape}")
    Shape: (144, 2)
    >>> print(f"Date range: {df['ds'].min()} to {df['ds'].max()}")
    Date range: 1949-01-01 to 1960-12-01

    Notes
    -----
    Data is downloaded from the Prophet examples repository on GitHub.
    Original source: Box, G. E. P., Jenkins, G. M. and Reinsel, G. C. (1976)
    Time Series Analysis, Forecasting and Control. Third Edition.
    """
    url = "https://raw.githubusercontent.com/facebook/prophet/main/examples/example_air_passengers.csv"
    df = pd.read_csv(url)
    df["ds"] = pd.to_datetime(df["ds"])
    return df


def load_peyton_manning() -> pd.DataFrame:
    """Load the Peyton Manning Wikipedia page views dataset.

    This dataset contains daily log-transformed Wikipedia page views for
    Peyton Manning from December 2007 to January 2016 (2905 observations).

    This dataset exhibits:
    - Multiple trend changes (career events)
    - Strong yearly seasonality (NFL season)
    - Weekly seasonality (game days)
    - Holiday effects (Super Bowl, playoffs)

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - `ds`: datetime, daily timestamps from 2007-12-10 to 2016-01-20
        - `y`: float, log-transformed page views

    Examples
    --------
    >>> from vangja.datasets import load_peyton_manning
    >>> df = load_peyton_manning()
    >>> print(f"Shape: {df.shape}")
    Shape: (2905, 2)
    >>> print(f"Date range: {df['ds'].min().date()} to {df['ds'].max().date()}")
    Date range: 2007-12-10 to 2016-01-20

    Notes
    -----
    Data is downloaded from the Prophet examples repository on GitHub.
    This is the same dataset used in the original Prophet paper and tutorials.
    """
    url = "https://raw.githubusercontent.com/facebook/prophet/main/examples/example_wp_log_peyton_manning.csv"
    df = pd.read_csv(url)
    df["ds"] = pd.to_datetime(df["ds"])
    return df
