"""Pytest fixtures for vangja tests."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_data():
    """Create sample time series data for testing."""
    np.random.seed(42)
    dates = pd.date_range(start="2020-01-01", periods=100, freq="D")
    values = (
        np.sin(np.arange(100) * 2 * np.pi / 30) * 10 + 50 + np.random.randn(100) * 2
    )

    return pd.DataFrame({"ds": dates, "y": values, "series": "test_series"})


@pytest.fixture
def multi_series_data():
    """Create multi-series time series data for testing."""
    np.random.seed(42)
    dates = pd.date_range(start="2020-01-01", periods=100, freq="D")

    series1_values = (
        np.sin(np.arange(100) * 2 * np.pi / 30) * 10 + 50 + np.random.randn(100) * 2
    )
    series2_values = (
        np.cos(np.arange(100) * 2 * np.pi / 30) * 15 + 60 + np.random.randn(100) * 3
    )

    df1 = pd.DataFrame({"ds": dates, "y": series1_values, "series": "series_a"})

    df2 = pd.DataFrame({"ds": dates, "y": series2_values, "series": "series_b"})

    return pd.concat([df1, df2], ignore_index=True)


@pytest.fixture
def linear_data():
    """Create linear trend data for testing."""
    np.random.seed(42)
    dates = pd.date_range(start="2020-01-01", periods=100, freq="D")
    t = np.arange(100)
    values = 2 * t + 10 + np.random.randn(100) * 5  # slope=2, intercept=10

    return pd.DataFrame({"ds": dates, "y": values, "series": "linear_series"})


@pytest.fixture
def seasonal_data():
    """Create seasonal data for testing Fourier seasonality."""
    np.random.seed(42)
    dates = pd.date_range(start="2020-01-01", periods=365, freq="D")
    t = np.arange(365)
    # Weekly seasonality
    weekly = 5 * np.sin(2 * np.pi * t / 7)
    # Yearly seasonality
    yearly = 20 * np.sin(2 * np.pi * t / 365)
    values = 100 + weekly + yearly + np.random.randn(365) * 2

    return pd.DataFrame({"ds": dates, "y": values, "series": "seasonal_series"})
