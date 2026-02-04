"""Dataset generation and loading utilities for vangja examples.

This module provides functions to load real-world datasets and generate
synthetic time series data for testing and demonstrating vangja's capabilities.

Real-world Datasets
-------------------
- `load_air_passengers()` — Classic monthly airline passenger data (1949-1960)
- `load_peyton_manning()` — Daily Wikipedia page views (2007-2016)

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

from vangja.datasets.loaders import load_air_passengers, load_peyton_manning
from vangja.datasets.synthetic import (
    generate_hierarchical_products,
    generate_multi_store_data,
)

__all__ = [
    "load_air_passengers",
    "load_peyton_manning",
    "generate_multi_store_data",
    "generate_hierarchical_products",
]
