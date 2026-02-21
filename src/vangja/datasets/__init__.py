"""Dataset generation and loading utilities for vangja examples.

This module provides functions to load real-world datasets and generate
synthetic time series data for testing and demonstrating vangja's capabilities.

Real-world Datasets
-------------------
- `load_air_passengers()` — Classic monthly airline passenger data (1949-1960)
- `load_peyton_manning()` — Daily Wikipedia page views (2007-2016)
- `load_citi_bike_sales()` — Daily bike rides from NYC station (2013-2014)
- `load_nyc_temperature()` — Daily max temperature for NYC (2012-2017)
- `load_kaggle_temperature()` — Hourly temperature for 36 cities (2012-2017)
- `load_smart_home_readings()` — Smart home appliance energy readings (2016)

Synthetic Datasets
------------------
- `generate_multi_store_data()` — Multiple store time series with trend + seasonality
- `generate_hierarchical_products()` — Products with group-level seasonal patterns

Examples
--------
>>> from vangja.datasets import load_air_passengers, generate_multi_store_data
>>>
>>> # Load real data
>>> air_passengers = load_air_passengers()
>>>
>>> # Generate synthetic data
>>> stores_df, store_params = generate_multi_store_data(seed=42)
"""

from vangja.datasets.loaders import (
    KaggleTemperatureCity,
    SmartHomeColumn,
    load_air_passengers,
    load_citi_bike_sales,
    load_kaggle_temperature,
    load_nyc_temperature,
    load_peyton_manning,
    load_smart_home_readings,
    load_stock_data,
)
from vangja.datasets.stocks import get_sp500_tickers_for_range
from vangja.datasets.synthetic import (
    generate_hierarchical_products,
    generate_multi_store_data,
)

__all__ = [
    "load_air_passengers",
    "load_peyton_manning",
    "load_citi_bike_sales",
    "load_nyc_temperature",
    "load_kaggle_temperature",
    "load_smart_home_readings",
    "load_stock_data",
    "KaggleTemperatureCity",
    "SmartHomeColumn",
    "get_sp500_tickers_for_range",
    "generate_multi_store_data",
    "generate_hierarchical_products",
]
