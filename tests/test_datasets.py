"""Tests for vangja.datasets module.

Tests for synthetic data generators and the _fourier_series helper.
Real-data loaders (load_air_passengers, load_peyton_manning, etc.) are NOT
tested here because they download from external URLs.
"""

import numpy as np
import pandas as pd
import pytest

from vangja.datasets.synthetic import (
    _fourier_series,
    generate_hierarchical_products,
    generate_multi_store_data,
)


# ---------------------------------------------------------------------------
# _fourier_series
# ---------------------------------------------------------------------------


class TestFourierSeries:
    """Tests for the _fourier_series helper function."""

    def test_output_shape(self):
        """Test that output has shape (len(t), 2 * n_components)."""
        t = np.linspace(0, 1, 100)
        result = _fourier_series(t, period=365.25, n_components=5)

        assert result.shape == (100, 10)

    def test_single_component(self):
        """Test with a single Fourier component (one sin + one cos)."""
        t = np.linspace(0, 1, 50)
        result = _fourier_series(t, period=7, n_components=1)

        assert result.shape == (50, 2)

    def test_values_bounded(self):
        """Test that output values are in [-1, 1]."""
        t = np.linspace(0, 10, 500)
        result = _fourier_series(t, period=365.25, n_components=10)

        assert result.min() >= -1.0
        assert result.max() <= 1.0

    def test_known_values_at_zero(self):
        """Test cos/sin values at t=0."""
        t = np.array([0.0])
        result = _fourier_series(t, period=1.0, n_components=2)

        # At t=0: cos(0)=1, cos(0)=1, sin(0)=0, sin(0)=0
        assert result.shape == (1, 4)
        np.testing.assert_allclose(result[0, 0], 1.0, atol=1e-10)  # cos(0)
        np.testing.assert_allclose(result[0, 1], 1.0, atol=1e-10)  # cos(0)
        np.testing.assert_allclose(result[0, 2], 0.0, atol=1e-10)  # sin(0)
        np.testing.assert_allclose(result[0, 3], 0.0, atol=1e-10)  # sin(0)

    def test_periodicity(self):
        """Test that the function is periodic."""
        period = 7.0
        t1 = np.array([0.0])
        t2 = np.array([period])
        r1 = _fourier_series(t1, period=period, n_components=3)
        r2 = _fourier_series(t2, period=period, n_components=3)

        np.testing.assert_allclose(r1, r2, atol=1e-10)

    def test_empty_array(self):
        """Test with empty input array."""
        t = np.array([])
        result = _fourier_series(t, period=7, n_components=3)

        assert result.shape == (0, 6)

    def test_large_n_components(self):
        """Test with a large number of components."""
        t = np.linspace(0, 1, 10)
        result = _fourier_series(t, period=365.25, n_components=20)

        assert result.shape == (10, 40)


# ---------------------------------------------------------------------------
# generate_multi_store_data
# ---------------------------------------------------------------------------


class TestGenerateMultiStoreData:
    """Tests for generate_multi_store_data function."""

    def test_returns_tuple(self):
        """Test that the function returns a tuple of (DataFrame, list)."""
        result = generate_multi_store_data(seed=42)

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], pd.DataFrame)
        assert isinstance(result[1], list)

    def test_dataframe_columns(self):
        """Test that the DataFrame has the expected columns."""
        df, _ = generate_multi_store_data(seed=42)

        assert "ds" in df.columns
        assert "y" in df.columns
        assert "series" in df.columns

    def test_five_stores(self):
        """Test that 5 stores are generated."""
        df, params = generate_multi_store_data(seed=42)

        assert df["series"].nunique() == 5
        assert len(params) == 5

    def test_store_names(self):
        """Test the expected store names."""
        df, _ = generate_multi_store_data(seed=42)
        expected_names = {
            "store_north",
            "store_south",
            "store_east",
            "store_west",
            "store_central",
        }

        assert set(df["series"].unique()) == expected_names

    def test_date_range(self):
        """Test that date range matches the specified parameters."""
        df, _ = generate_multi_store_data(
            start_date="2020-01-01", end_date="2020-12-31", seed=42
        )

        assert df["ds"].min() == pd.Timestamp("2020-01-01")
        assert df["ds"].max() == pd.Timestamp("2020-12-31")

    def test_all_stores_same_date_range(self):
        """Test that all stores share the same date range."""
        df, _ = generate_multi_store_data(seed=42)

        for store in df["series"].unique():
            store_data = df[df["series"] == store]
            assert store_data["ds"].min() == df["ds"].min()
            assert store_data["ds"].max() == df["ds"].max()

    def test_all_stores_same_length(self):
        """Test that all stores have the same number of observations."""
        df, _ = generate_multi_store_data(seed=42)
        lengths = df.groupby("series").size()

        assert lengths.nunique() == 1

    def test_deterministic_with_seed(self):
        """Test that results are deterministic with the same seed."""
        df1, _ = generate_multi_store_data(seed=123)
        df2, _ = generate_multi_store_data(seed=123)

        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_give_different_results(self):
        """Test that different seeds produce different data."""
        df1, _ = generate_multi_store_data(seed=1)
        df2, _ = generate_multi_store_data(seed=2)

        # y values should differ
        assert not np.allclose(df1["y"].values, df2["y"].values)

    def test_params_structure(self):
        """Test that params list has expected keys."""
        _, params = generate_multi_store_data(seed=42)

        for p in params:
            assert "name" in p
            assert "trend_slope" in p
            assert "trend_intercept" in p
            assert "yearly_amplitude" in p
            assert "weekly_amplitude" in p
            assert "noise_std" in p

    def test_custom_frequency(self):
        """Test with weekly frequency."""
        df, _ = generate_multi_store_data(
            start_date="2020-01-01", end_date="2020-12-31", freq="W", seed=42
        )

        # Weekly frequency should give ~52 weeks per store
        per_store = len(df) // 5
        assert 50 <= per_store <= 55

    def test_ds_column_is_datetime(self):
        """Test that ds column is datetime type."""
        df, _ = generate_multi_store_data(seed=42)

        assert pd.api.types.is_datetime64_any_dtype(df["ds"])

    def test_y_values_are_numeric(self):
        """Test that y values are numeric (float)."""
        df, _ = generate_multi_store_data(seed=42)

        assert pd.api.types.is_float_dtype(df["y"])


# ---------------------------------------------------------------------------
# generate_hierarchical_products
# ---------------------------------------------------------------------------


class TestGenerateHierarchicalProducts:
    """Tests for generate_hierarchical_products function."""

    def test_returns_tuple(self):
        """Test that the function returns a tuple of (DataFrame, dict)."""
        result = generate_hierarchical_products(seed=42)

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], pd.DataFrame)
        assert isinstance(result[1], dict)

    def test_dataframe_columns(self):
        """Test that the DataFrame has the expected columns."""
        df, _ = generate_hierarchical_products(seed=42)

        assert "ds" in df.columns
        assert "y" in df.columns
        assert "series" in df.columns

    def test_five_products_default(self):
        """Test that 5 products are generated by default."""
        df, params = generate_hierarchical_products(seed=42)

        assert df["series"].nunique() == 5
        assert len(params) == 5

    def test_six_products_with_all_year(self):
        """Test that 6 products are generated with include_all_year=True."""
        df, params = generate_hierarchical_products(
            seed=42, include_all_year=True
        )

        assert df["series"].nunique() == 6
        assert len(params) == 6
        assert "all_year" in params

    def test_product_names(self):
        """Test the expected product names without all_year."""
        df, params = generate_hierarchical_products(seed=42)
        expected = {"summer_1", "summer_2", "summer_3", "winter_1", "winter_2"}

        assert set(df["series"].unique()) == expected
        assert set(params.keys()) == expected

    def test_product_groups(self):
        """Test that products belong to the correct groups."""
        _, params = generate_hierarchical_products(seed=42)

        summer_count = sum(1 for p in params.values() if p["group"] == "summer")
        winter_count = sum(1 for p in params.values() if p["group"] == "winter")

        assert summer_count == 3
        assert winter_count == 2

    def test_all_year_group(self):
        """Test the all_year product group."""
        _, params = generate_hierarchical_products(
            seed=42, include_all_year=True
        )

        assert params["all_year"]["group"] == "all_year"

    def test_params_structure(self):
        """Test that each product's params have expected keys."""
        _, params = generate_hierarchical_products(seed=42)

        for name, p in params.items():
            assert "k" in p, f"Missing 'k' for {name}"
            assert "m" in p, f"Missing 'm' for {name}"
            assert "delta" in p, f"Missing 'delta' for {name}"
            assert "yearly_beta" in p, f"Missing 'yearly_beta' for {name}"
            assert "weekly_beta" in p, f"Missing 'weekly_beta' for {name}"
            assert "noise_std" in p, f"Missing 'noise_std' for {name}"
            assert "group" in p, f"Missing 'group' for {name}"

    def test_date_range(self):
        """Test that date range matches defaults."""
        df, _ = generate_hierarchical_products(seed=42)

        assert df["ds"].min() == pd.Timestamp("2018-01-01")
        assert df["ds"].max() == pd.Timestamp("2019-12-31")

    def test_custom_date_range(self):
        """Test with custom date range."""
        df, _ = generate_hierarchical_products(
            start_date="2020-06-01", end_date="2021-05-31", seed=42
        )

        assert df["ds"].min() == pd.Timestamp("2020-06-01")
        assert df["ds"].max() == pd.Timestamp("2021-05-31")

    def test_deterministic_with_seed(self):
        """Test that the same seed produces the same results."""
        df1, _ = generate_hierarchical_products(seed=99)
        df2, _ = generate_hierarchical_products(seed=99)

        pd.testing.assert_frame_equal(df1, df2)

    def test_opposite_seasonality_direction(self):
        """Test that summer and winter products have opposite yearly betas."""
        _, params = generate_hierarchical_products(seed=42)

        summer_beta = params["summer_1"]["yearly_beta"]
        winter_beta = params["winter_1"]["yearly_beta"]

        # They should have opposite signs (approximately)
        # The correlation should be strongly negative
        correlation = np.corrcoef(summer_beta, winter_beta)[0, 1]
        assert correlation < -0.5

    def test_all_products_same_date_range(self):
        """Test that all products share the same date range."""
        df, _ = generate_hierarchical_products(seed=42)

        for product in df["series"].unique():
            product_data = df[df["series"] == product]
            assert product_data["ds"].min() == df["ds"].min()
            assert product_data["ds"].max() == df["ds"].max()

    def test_changepoint_count(self):
        """Test that delta arrays have correct length."""
        _, params = generate_hierarchical_products(
            n_changepoints=12, seed=42
        )

        for name, p in params.items():
            assert len(p["delta"]) == 12, f"Wrong delta length for {name}"

    def test_ds_column_is_datetime(self):
        """Test that ds column is datetime type."""
        df, _ = generate_hierarchical_products(seed=42)

        assert pd.api.types.is_datetime64_any_dtype(df["ds"])
