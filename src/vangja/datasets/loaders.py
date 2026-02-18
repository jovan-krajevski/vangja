"""Functions for loading real-world time series datasets.

This module provides convenience functions for loading commonly used
time series datasets in the format expected by vangja (columns: ds, y).
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from urllib.request import urlopen

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


def load_citi_bike_sales() -> pd.DataFrame:
    """Load the Citi Bike station 360 sales dataset.

    This dataset contains daily bike ride counts from Citi Bike station 360
    in New York City (2013-07-01 to 2014-10-31). It is used to demonstrate
    forecasting short time series with transfer learning.

    The dataset exhibits:

    - Strong weekly seasonality (weekday vs weekend patterns)
    - Yearly seasonality correlated with temperature/weather
    - Approximately 3 months of initial data used for training (~106 days)

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:

        - ``ds``: datetime, daily timestamps from 2013-07-01 to 2014-10-31
        - ``y``: float, number of bike rides

    Examples
    --------
    >>> from vangja.datasets import load_citi_bike_sales
    >>> df = load_citi_bike_sales()
    >>> print(f"Shape: {df.shape}")  # doctest: +SKIP
    Shape: (488, 2)

    Notes
    -----
    This dataset is from Tim Radtke's blog post "Modeling Short Time Series
    with Prior Knowledge". The vangja library was partially inspired by this
    work and Juan Orduz's PyMC implementation.

    Requires the ``pyreadr`` package (install with ``pip install vangja[datasets]``).

    References
    ----------
    .. [1] Radtke, T. (2019). Modeling Short Time Series with Prior Knowledge.
       https://minimizeregret.com/short-time-series-prior-knowledge
    .. [2] Orduz, J. (2022). Modeling Short Time Series with Prior Knowledge in PyMC.
       https://juanitorduz.github.io/short_time_series_pymc/
    """
    try:
        import pyreadr
    except ImportError as e:
        raise ImportError(
            "pyreadr is required to load Citi Bike data. "
            "Install with: pip install vangja[datasets]"
        ) from e

    url = "https://github.com/timradtke/short-time-series/raw/master/citi_bike_360.Rds"
    with tempfile.NamedTemporaryFile(suffix=".Rds") as tmp:
        with urlopen(url) as resp:
            tmp.write(resp.read())
            tmp.flush()
        rds_result = pyreadr.read_r(tmp.name)

    df = rds_result[None]  # RDS files have a single dataframe with key None
    df = df.rename(columns={"date": "ds", "rides": "y"})
    df["ds"] = pd.to_datetime(df["ds"])
    # Remove rows with missing values
    df = df.dropna(subset=["y"])
    # Keep data before 2015-10-01
    df = df[df["ds"] < "2015-10-01"]
    return df[["ds", "y"]]


def load_nyc_temperature(return_daily_average: bool = True) -> pd.DataFrame:
    """Load New York City historical daily temperature data.

    This dataset contains daily maximum temperatures (Fahrenheit) for
    New York City from 2012-10-01 to 2017-11-29. It is used to learn
    yearly seasonality patterns that can be transferred to short time series.

    The dataset exhibits:

    - Strong yearly seasonality (summer highs, winter lows)
    - Consistent periodic pattern across years

    Parameters
    ----------
    return_daily_average : bool, default True
        If True, return daily average temperatures. If False, return raw hourly data.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:

        - ``ds``: datetime, daily timestamps from 2012-10-01 to 2017-11-29
        - ``y``: float, maximum daily temperature in Fahrenheit

    Examples
    --------
    >>> from vangja.datasets import load_nyc_temperature
    >>> df = load_nyc_temperature()
    >>> print(f"Shape: {df.shape}")  # doctest: +SKIP
    Shape: (1886, 2)

    Notes
    -----
    This dataset is from Tim Radtke's blog post "Modeling Short Time Series
    with Prior Knowledge". The temperature seasonality can be used as prior
    information for forecasting related short time series (e.g., bike sales).

    References
    ----------
    .. [1] Radtke, T. (2019). Modeling Short Time Series with Prior Knowledge.
       https://minimizeregret.com/short-time-series-prior-knowledge
    .. [2] Original data from Kaggle historical hourly weather data.
       https://www.kaggle.com/selfishgene/historical-hourly-weather-data
    """
    url = "https://raw.githubusercontent.com/timradtke/short-time-series/master/temperature.csv"
    df = pd.read_csv(url)
    df = df.rename(columns={"datetime": "ds", "New York": "y"})
    df["ds"] = pd.to_datetime(df["ds"])
    if return_daily_average:
        df = df.resample("D", on="ds").mean().reset_index()

    return df[["ds", "y"]]


def load_stock_data(
    tickers: list[str],
    split_date: str | pd.Timestamp,
    window_size: int,
    horizon_size: int,
    cache_path: Path | None = None,
    interpolate: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load historical stock data split into training and test sets.

    Downloads daily OHLCV data for the specified tickers using Yahoo
    Finance and computes the typical price as
    ``(Open + High + Low + Close) / 4``. The data is split into a
    training window and a test horizon around ``split_date``.

    Parameters
    ----------
    tickers : list[str]
        List of ticker symbols to download (e.g., ``["AAPL", "MSFT"]``).
    split_date : str or pd.Timestamp
        The date separating training and test data. Training data
        covers ``[split_date - window_size, split_date)`` and test
        data covers ``[split_date, split_date + horizon_size]``.
    window_size : int
        Number of calendar days for the training window (before
        ``split_date``).
    horizon_size : int
        Number of calendar days for the test horizon (from
        ``split_date`` onwards).
    cache_path : Path or None, default None
        Directory for caching downloaded data. Each ticker is stored
        as a CSV file. If None, data is downloaded without caching.
        If provided, parent directories are created if they do not
        exist.
    interpolate : bool, default False
        If True, missing days (weekends, holidays) within each series
        are filled using linear interpolation after reindexing to a
        daily calendar.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        ``(train_df, test_df)`` — DataFrames with columns:

        - ``ds``: datetime
        - ``y``: float, typical price
        - ``series``: str, ticker symbol

    Examples
    --------
    >>> from vangja.datasets import load_stock_data
    >>> train, test = load_stock_data(
    ...     ["AAPL"], "2024-01-01", window_size=365, horizon_size=30
    ... )  # doctest: +SKIP
    >>> print(train.columns.tolist())  # doctest: +SKIP
    ['ds', 'y', 'series']

    Notes
    -----
    Requires the ``yfinance`` package (install with
    ``pip install vangja[datasets]``).
    """
    from vangja.datasets.stocks import _download_stock_data

    split = pd.Timestamp(split_date)
    start = split - pd.Timedelta(days=window_size - 1)
    end = split + pd.Timedelta(days=horizon_size)

    extended_start = start - pd.Timedelta(days=5)
    extended_end = end + pd.Timedelta(days=5)

    data = _download_stock_data(tickers, cache_path=cache_path)

    if data.empty:
        empty: pd.DataFrame = pd.DataFrame(columns=["ds", "y", "series"])
        return empty, empty.copy()

    # Build output DataFrame
    result = data[["ds", "ticker", "typical_price"]].rename(
        columns={"ticker": "series", "typical_price": "y"},
    )

    # Filter to requested date range
    result = result[
        (result["ds"] >= extended_start) & (result["ds"] <= extended_end)
    ].copy()

    if interpolate:
        interpolated: list[pd.DataFrame] = []
        for ticker in result["series"].unique():
            ticker_data = result[result["series"] == ticker].copy()
            full_range = pd.date_range(start=extended_start, end=extended_end, freq="D")
            ticker_data = ticker_data.set_index("ds").reindex(full_range)
            ticker_data["y"] = ticker_data["y"].interpolate(method="linear")
            ticker_data["series"] = ticker
            ticker_data = ticker_data.reset_index().rename(
                columns={"index": "ds"},
            )
            # Drop edges where forward/backward fill didn't reach
            ticker_data = ticker_data.dropna(subset=["y"])
            interpolated.append(ticker_data)
        result = pd.concat(interpolated, ignore_index=True)

    # Split into train and test
    train_df = (
        result[(result["ds"] >= start) & (result["ds"] <= split)]
        .copy()
        .reset_index(drop=True)
    )
    test_df = (
        result[(result["ds"] > split) & (result["ds"] <= end)]
        .copy()
        .reset_index(drop=True)
    )

    return train_df, test_df


def _ensure_kagglehub() -> None:
    """Import kagglehub or raise a helpful ImportError."""
    try:
        import kagglehub  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "kagglehub is required to download Kaggle datasets. "
            "Install with: pip install vangja[datasets]"
        ) from e


# Valid cities in the Kaggle historical-hourly-weather-data temperature.csv
KAGGLE_TEMPERATURE_CITIES: list[str] = [
    "Portland",
    "San Francisco",
    "Seattle",
    "Los Angeles",
    "San Diego",
    "Las Vegas",
    "Phoenix",
    "Albuquerque",
    "Denver",
    "San Antonio",
    "Dallas",
    "Houston",
    "Kansas City",
    "Minneapolis",
    "Saint Louis",
    "Chicago",
    "Nashville",
    "Indianapolis",
    "Atlanta",
    "Detroit",
    "Jacksonville",
    "Charlotte",
    "Miami",
    "Pittsburgh",
    "Philadelphia",
    "New York",
    "Boston",
    "Vancouver",
    "Toronto",
    "Montreal",
    "Beersheba",
    "Tel Aviv District",
    "Eilat",
    "Haifa",
    "Nahariyya",
    "Jerusalem",
]

# Valid appliance / total columns in the Kaggle smart-home HomeC.csv
SMART_HOME_COLUMNS: list[str] = [
    "use [kW]",
    "gen [kW]",
    "House overall [kW]",
    "Dishwasher [kW]",
    "Furnace 1 [kW]",
    "Furnace 2 [kW]",
    "Home office [kW]",
    "Fridge [kW]",
    "Wine cellar [kW]",
    "Garage door [kW]",
    "Kitchen 12 [kW]",
    "Kitchen 14 [kW]",
    "Kitchen 38 [kW]",
    "Barn [kW]",
    "Well [kW]",
    "Microwave [kW]",
    "Living room [kW]",
    "Solar [kW]",
]


def load_kaggle_temperature(
    city: str = "New York",
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
    freq: str = "D",
) -> pd.DataFrame:
    """Load historical hourly temperature data from Kaggle.

    Downloads the ``temperature.csv`` file from the
    `Historical Hourly Weather Data
    <https://www.kaggle.com/datasets/selfishgene/historical-hourly-weather-data>`_
    dataset. Returns data for the requested city, filtered to the given
    date range and aggregated to the specified frequency.

    The raw data contains hourly observations in **Kelvin**. Values are
    converted to **Celsius** before returning.

    Parameters
    ----------
    city : str, default "New York"
        City column to extract. Must be one of the 36 cities in the
        dataset (see ``KAGGLE_TEMPERATURE_CITIES``).
    start_date : str, pd.Timestamp, or None, default None
        Start of the date range (inclusive). If None, the earliest
        available date is used (~2012-10-01).
    end_date : str, pd.Timestamp, or None, default None
        End of the date range (inclusive). If None, the latest
        available date is used (~2017-11-30).
    freq : str, default "D"
        Pandas offset alias for temporal aggregation (e.g. ``"D"`` for
        daily mean, ``"W"`` for weekly mean, ``"h"`` for hourly — no
        aggregation). The aggregation function is ``mean``.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:

        - ``ds``: datetime
        - ``y``: float, temperature in degrees Celsius

    Raises
    ------
    ImportError
        If ``kagglehub`` is not installed.
    ValueError
        If ``city`` is not a valid column in the dataset.

    Examples
    --------
    >>> from vangja.datasets import load_kaggle_temperature
    >>> df = load_kaggle_temperature("New York", "2015-01-01", "2015-12-31")  # doctest: +SKIP
    >>> print(df.columns.tolist())  # doctest: +SKIP
    ['ds', 'y']

    Notes
    -----
    Requires the ``kagglehub`` package (install with
    ``pip install vangja[datasets]``).

    Data is downloaded and cached locally by ``kagglehub``. A valid
    Kaggle API token is required (see
    `Kaggle API docs <https://github.com/Kaggle/kaggle-api#api-credentials>`_).

    References
    ----------
    .. [1] Historical Hourly Weather Data.
       https://www.kaggle.com/datasets/selfishgene/historical-hourly-weather-data
    """
    if city not in KAGGLE_TEMPERATURE_CITIES:
        raise ValueError(
            f"Unknown city {city!r}. Must be one of: "
            f"{', '.join(KAGGLE_TEMPERATURE_CITIES)}"
        )

    _ensure_kagglehub()
    import kagglehub

    path = kagglehub.dataset_download(
        "selfishgene/historical-hourly-weather-data",
    )
    csv_path = Path(path) / "temperature.csv"

    df = pd.read_csv(csv_path, usecols=["datetime", city])
    df = df.rename(columns={"datetime": "ds", city: "y"})
    df["ds"] = pd.to_datetime(df["ds"])

    # Convert Kelvin → Celsius
    df["y"] = df["y"] - 273.15

    # Drop missing values
    df = df.dropna(subset=["y"])

    # Filter date range
    if start_date is not None:
        df = df[df["ds"] >= pd.Timestamp(start_date)]
    if end_date is not None:
        df = df[df["ds"] <= pd.Timestamp(end_date)]

    # Aggregate to requested frequency
    df = df.resample(freq, on="ds").mean(numeric_only=True).reset_index()
    df = df.dropna(subset=["y"])

    return df[["ds", "y"]]


def load_smart_home_readings(
    column: str = "use [kW]",
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
    freq: str | None = None,
) -> pd.DataFrame:
    """Load smart home energy readings from Kaggle.

    Downloads the ``HomeC.csv`` file from the
    `Smart Home Dataset with Weather Information
    <https://www.kaggle.com/datasets/taranvee/smart-home-dataset-with-weather-information>`_
    dataset. Returns data for the requested appliance or total column,
    filtered to the given date range and aggregated to the specified
    frequency.

    The raw data has 1-minute resolution and covers roughly
    2016-01-01 to 2016-12-16. Each column is in **kW**.

    Parameters
    ----------
    column : str, default "use [kW]"
        The appliance or total column to extract. Must be one of the
        energy columns in the dataset (see ``SMART_HOME_COLUMNS``).
        Common choices:

        - ``"use [kW]"`` — total energy use
        - ``"gen [kW]"`` — total energy generation
        - ``"House overall [kW]"`` — house overall consumption
        - ``"Dishwasher [kW]"``, ``"Fridge [kW]"``, etc. — individual
          appliances
    start_date : str, pd.Timestamp, or None, default None
        Start of the date range (inclusive). If None, the earliest
        available date is used (~2016-01-01).
    end_date : str, pd.Timestamp, or None, default None
        End of the date range (inclusive). If None, the latest
        available date is used (~2016-12-16).
    freq : str or None, default "D"
        Pandas offset alias for temporal aggregation (e.g. ``"D"`` for
        daily mean, ``"h"`` for hourly mean, ``"W"`` for weekly mean).
        The aggregation function is ``mean``. If None, no aggregation
        is performed and the original 1-minute data is returned.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:

        - ``ds``: datetime
        - ``y``: float, energy reading in kW

    Raises
    ------
    ImportError
        If ``kagglehub`` is not installed.
    ValueError
        If ``column`` is not a valid column in the dataset.

    Examples
    --------
    >>> from vangja.datasets import load_smart_home_readings
    >>> df = load_smart_home_readings("Fridge [kW]", "2016-03-01", "2016-06-30")  # doctest: +SKIP
    >>> print(df.columns.tolist())  # doctest: +SKIP
    ['ds', 'y']

    Notes
    -----
    Requires the ``kagglehub`` package (install with
    ``pip install vangja[datasets]``).

    Data is downloaded and cached locally by ``kagglehub``. A valid
    Kaggle API token is required (see
    `Kaggle API docs <https://github.com/Kaggle/kaggle-api#api-credentials>`_).

    The raw ``time`` column contains Unix timestamps. The last row of
    the CSV may contain malformed data and is automatically dropped.

    References
    ----------
    .. [1] Smart Home Dataset with Weather Information.
       https://www.kaggle.com/datasets/taranvee/smart-home-dataset-with-weather-information
    """
    if column not in SMART_HOME_COLUMNS:
        raise ValueError(
            f"Unknown column {column!r}. Must be one of: "
            f"{', '.join(SMART_HOME_COLUMNS)}"
        )

    _ensure_kagglehub()
    import kagglehub

    path = kagglehub.dataset_download(
        "taranvee/smart-home-dataset-with-weather-information",
    )
    csv_path = Path(path) / "HomeC.csv"

    df = pd.read_csv(csv_path, usecols=[column])
    df = df.rename(columns={column: "y"})

    # Fix timestamps
    df["ds"] = pd.date_range("2016-01-01 05:00", periods=len(df), freq="min")

    # Coerce y to numeric (some rows may contain header strings)
    df["y"] = pd.to_numeric(df["y"], errors="coerce")

    # Filter date range
    if start_date is not None:
        df = df[df["ds"] >= pd.Timestamp(start_date)]
    if end_date is not None:
        df = df[df["ds"] <= pd.Timestamp(end_date)]

    # Aggregate to requested frequency
    if freq is not None:
        df = df.resample(freq, on="ds").mean(numeric_only=True).reset_index()

    df = df.dropna(subset=["ds", "y"])

    return df[["ds", "y"]]
