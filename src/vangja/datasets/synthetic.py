"""Functions for generating synthetic time series datasets.

This module provides functions to generate synthetic time series data
with known parameters for testing and demonstrating vangja's capabilities.
"""

from typing import Any

import numpy as np
import pandas as pd


def _fourier_series(t: np.ndarray, period: float, n_components: int) -> np.ndarray:
    """Generate Fourier series basis functions.

    Parameters
    ----------
    t : np.ndarray
        Time values (normalized or raw)
    period : float
        Period of the seasonal component
    n_components : int
        Number of Fourier terms (sin/cos pairs)

    Returns
    -------
    np.ndarray
        Matrix of shape (len(t), 2 * n_components) with cos and sin terms
    """
    x = 2 * np.pi * (np.arange(n_components) + 1) * t[:, None] / period
    return np.concatenate((np.cos(x), np.sin(x)), axis=1)


def generate_multi_store_data(
    start_date: str = "2015-01-01",
    end_date: str = "2019-12-31",
    freq: str = "D",
    seed: int | None = 42,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Generate synthetic multi-store time series data.

    Creates 5 synthetic time series representing different stores, all sharing
    the same date range. Each series has:
    - Linear trend with different slopes and intercepts
    - Yearly seasonality with different amplitudes
    - Weekly seasonality
    - Random noise

    This dataset is ideal for demonstrating:
    - Simultaneous vs sequential fitting
    - Individual pooling across multiple series
    - Vectorized multi-series forecasting

    Parameters
    ----------
    start_date : str, default "2015-01-01"
        Start date for the time series
    end_date : str, default "2019-12-31"
        End date for the time series
    freq : str, default "D"
        Frequency of the time series (e.g., "D" for daily)
    seed : int or None, default 42
        Random seed for reproducibility. Set to None for random data.

    Returns
    -------
    df : pd.DataFrame
        Combined DataFrame with columns:
        - `ds`: datetime timestamps
        - `y`: target values
        - `series`: store name (e.g., "store_north")
    params : list of dict
        List of parameter dictionaries for each store, containing:
        - `name`: store name
        - `trend_slope`, `trend_intercept`: trend parameters
        - `yearly_amplitude`, `weekly_amplitude`: seasonality amplitudes
        - `noise_std`: noise standard deviation

    Examples
    --------
    >>> from vangja.datasets import generate_multi_store_data
    >>> df, params = generate_multi_store_data(seed=42)
    >>> print(f"Total samples: {len(df)}")
    >>> print(f"Number of stores: {df['series'].nunique()}")
    >>> print(f"Stores: {df['series'].unique().tolist()}")
    """
    if seed is not None:
        np.random.seed(seed)

    dates = pd.date_range(start=start_date, end=end_date, freq=freq)

    # Store parameters
    store_params = [
        {
            "name": "store_north",
            "trend_slope": 20,
            "trend_intercept": 100,
            "yearly_amplitude": 30,
            "weekly_amplitude": 5,
            "noise_std": 8,
        },
        {
            "name": "store_south",
            "trend_slope": 15,
            "trend_intercept": 80,
            "yearly_amplitude": 25,
            "weekly_amplitude": 8,
            "noise_std": 6,
        },
        {
            "name": "store_east",
            "trend_slope": 25,
            "trend_intercept": 120,
            "yearly_amplitude": 40,
            "weekly_amplitude": 6,
            "noise_std": 10,
        },
        {
            "name": "store_west",
            "trend_slope": 10,
            "trend_intercept": 90,
            "yearly_amplitude": 20,
            "weekly_amplitude": 4,
            "noise_std": 5,
        },
        {
            "name": "store_central",
            "trend_slope": 30,
            "trend_intercept": 150,
            "yearly_amplitude": 50,
            "weekly_amplitude": 10,
            "noise_std": 12,
        },
    ]

    all_series = []
    for params in store_params:
        n = len(dates)
        t = np.arange(n)

        # Linear trend
        trend = params["trend_intercept"] + params["trend_slope"] * t / 365

        # Yearly seasonality (period = 365.25 days)
        yearly = params["yearly_amplitude"] * np.sin(2 * np.pi * t / 365.25)

        # Weekly seasonality (period = 7 days)
        weekly = params["weekly_amplitude"] * np.sin(2 * np.pi * t / 7)

        # Random noise
        noise = np.random.randn(n) * params["noise_std"]

        y = trend + yearly + weekly + noise

        series_df = pd.DataFrame({"ds": dates, "y": y, "series": params["name"]})
        all_series.append(series_df)

    df = pd.concat(all_series, ignore_index=True)
    return df, store_params


def generate_hierarchical_products(
    start_date: str = "2018-01-01",
    end_date: str = "2019-12-31",
    freq: str = "D",
    n_changepoints: int = 8,
    seed: int | None = 42,
    include_all_year: bool = False,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    """Generate synthetic hierarchical product time series data.

    Creates synthetic time series representing products that belong to
    groups with different seasonal patterns:

    - **Summer products** (3 series): Peak sales in summer months
    - **Winter products** (2 series): Peak sales in winter months (opposite)
    - **All-year products** (1 series, optional): Minimal seasonality

    Each series has:

    - Piecewise linear trend with changepoints (Prophet-style)
    - Yearly seasonality from Fourier series (group-specific pattern)
    - Weekly seasonality from Fourier series
    - Random noise

    This dataset is ideal for demonstrating:

    - Hierarchical Bayesian modeling with partial pooling
    - Using UniformConstant(-1, 1) to handle opposite seasonality directions
    - Group-level parameter sharing
    - Shrinkage effects

    Parameters
    ----------
    start_date : str, default "2018-01-01"
        Start date for the time series
    end_date : str, default "2019-12-31"
        End date for the time series
    freq : str, default "D"
        Frequency of the time series (e.g., "D" for daily)
    n_changepoints : int, default 8
        Number of potential changepoints in the trend
    seed : int or None, default 42
        Random seed for reproducibility. Set to None for random data.
    include_all_year : bool, default False
        If True, include an "all_year" product with minimal seasonality.
        This creates 6 series total instead of 5.

    Returns
    -------
    df : pd.DataFrame
        Combined DataFrame with columns:

        - `ds`: datetime timestamps
        - `y`: target values (sales)
        - `series`: product name (e.g., "summer_1", "winter_2", "all_year")

    params : dict
        Dictionary mapping product names to their parameters:

        - `k`: initial slope
        - `m`: initial intercept (base level)
        - `delta`: slope changes at changepoints
        - `yearly_beta`: Fourier coefficients for yearly seasonality
        - `weekly_beta`: Fourier coefficients for weekly seasonality
        - `noise_std`: noise standard deviation
        - `group`: "summer", "winter", or "all_year"

    Examples
    --------
    >>> from vangja.datasets import generate_hierarchical_products
    >>> df, params = generate_hierarchical_products(seed=42)
    >>> print(f"Total samples: {len(df)}")
    >>> print(f"Products: {list(params.keys())}")
    >>> print(f"Summer products: {[k for k, v in params.items() if v['group'] == 'summer']}")
    >>> print(f"Winter products: {[k for k, v in params.items() if v['group'] == 'winter']}")

    Including the all-year product:

    >>> df, params = generate_hierarchical_products(seed=42, include_all_year=True)
    >>> print(f"All-year products: {[k for k, v in params.items() if v['group'] == 'all_year']}")

    Notes
    -----
    The data generation follows the Prophet/timeseers formulation:

    y = g(t) + s_yearly(t) + s_weekly(t) + noise

    where:

    - g(t) is a piecewise linear trend with changepoints
    - s(t) are Fourier series seasonality components
    - Summer products have positive yearly seasonality (peak in summer)
    - Winter products have negative yearly seasonality (peak in winter)
    - All-year products have minimal seasonality (nearly flat)

    To model products with opposite seasonality directions, use
    UniformConstant(-1, 1) as a scaling factor in the model composition.
    """
    if seed is not None:
        np.random.seed(seed)

    dates = pd.date_range(start=start_date, end=end_date, freq=freq)
    n_days = len(dates)
    t_normalized = np.linspace(0, 1, n_days)

    # Shared changepoint times (evenly spaced in first 80% of data)
    changepoint_times = np.linspace(0, 0.8, n_changepoints)

    # Number of Fourier components
    n_yearly_components = 3
    n_weekly_components = 2

    # Define seasonal patterns for summer and winter product groups
    # Summer products peak in summer (positive first Fourier component)
    yearly_summer_beta = np.array([2.0, 0.5, 0.2, 0.8, 0.3, 0.1]) * 1000
    # Winter products have opposite seasonality
    yearly_winter_beta = -yearly_summer_beta

    # Shared weekly pattern base
    weekly_base_beta = np.array([0.3, 0.1, 0.15, 0.05]) * 200

    # Define parameters for each product
    product_params = {
        # Summer products - similar yearly pattern, different trends
        "summer_1": {
            "k": 0.5,
            "m": 5000,
            "delta": np.random.laplace(0, 0.3, n_changepoints) * 2,
            "yearly_beta": yearly_summer_beta * (1.0 + np.random.randn(6) * 0.1),
            "weekly_beta": weekly_base_beta * 1.2,
            "noise_std": 150,
            "group": "summer",
        },
        "summer_2": {
            "k": 0.3,
            "m": 3000,
            "delta": np.random.laplace(0, 0.3, n_changepoints) * 1.5,
            "yearly_beta": yearly_summer_beta * (1.1 + np.random.randn(6) * 0.1),
            "weekly_beta": weekly_base_beta * 0.8,
            "noise_std": 100,
            "group": "summer",
        },
        "summer_3": {
            "k": 0.8,
            "m": 7000,
            "delta": np.random.laplace(0, 0.3, n_changepoints) * 2.5,
            "yearly_beta": yearly_summer_beta * (0.9 + np.random.randn(6) * 0.1),
            "weekly_beta": weekly_base_beta * 1.0,
            "noise_std": 200,
            "group": "summer",
        },
        # Winter products - opposite yearly pattern
        "winter_1": {
            "k": 0.4,
            "m": 4000,
            "delta": np.random.laplace(0, 0.3, n_changepoints) * 1.8,
            "yearly_beta": yearly_winter_beta * (1.0 + np.random.randn(6) * 0.1),
            "weekly_beta": weekly_base_beta * 0.9,
            "noise_std": 120,
            "group": "winter",
        },
        "winter_2": {
            "k": 0.6,
            "m": 6000,
            "delta": np.random.laplace(0, 0.3, n_changepoints) * 2.2,
            "yearly_beta": yearly_winter_beta * (0.95 + np.random.randn(6) * 0.1),
            "weekly_beta": weekly_base_beta * 1.1,
            "noise_std": 180,
            "group": "winter",
        },
    }

    # Optionally add all-year product with minimal seasonality
    if include_all_year:
        product_params["all_year"] = {
            "k": 0.5,
            "m": 4500,
            "delta": np.random.laplace(0, 0.3, n_changepoints) * 1.5,
            "yearly_beta": yearly_summer_beta * 0.1,  # Minimal yearly seasonality
            "weekly_beta": weekly_base_beta * 1.0,
            "noise_std": 100,
            "group": "all_year",
        }

    def _generate_single_series(
        name: str,
        dates: pd.DatetimeIndex,
        k: float,
        m: float,
        delta: np.ndarray,
        yearly_beta: np.ndarray,
        weekly_beta: np.ndarray,
        noise_std: float,
        **kwargs,
    ) -> pd.DataFrame:
        """Generate a single time series with trend, seasonality, and noise."""
        n = len(dates)
        t = np.linspace(0, 1, n)

        # Piecewise linear trend with changepoints
        A = (t[:, None] > changepoint_times) * 1
        growth = k + A @ delta
        gamma = -changepoint_times * delta
        offset = m + A @ gamma
        trend = growth * t + offset

        # Yearly seasonality (using Fourier series)
        n_yearly = len(yearly_beta) // 2
        yearly_fourier = _fourier_series(t, 365.25 / n, n_yearly)
        yearly_seasonality = yearly_fourier @ yearly_beta

        # Weekly seasonality (using Fourier series)
        n_weekly = len(weekly_beta) // 2
        weekly_fourier = _fourier_series(t, 7 / n, n_weekly)
        weekly_seasonality = weekly_fourier @ weekly_beta

        # Combine components
        y = (
            trend
            + yearly_seasonality
            + weekly_seasonality
            + np.random.randn(n) * noise_std
        )

        return pd.DataFrame({"ds": dates, "y": y, "series": name})

    # Generate all series
    all_series = []
    for name, params in product_params.items():
        series = _generate_single_series(name=name, dates=dates, **params)
        all_series.append(series)

    df = pd.concat(all_series, ignore_index=True)
    return df, product_params
