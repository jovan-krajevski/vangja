"""Tests for Kaggle dataset loaders.

These tests mock kagglehub and CSV reads to stay offline and deterministic.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from vangja.datasets.loaders import (
    KAGGLE_TEMPERATURE_CITIES,
    SMART_HOME_COLUMNS,
    load_kaggle_temperature,
    load_smart_home_readings,
)


# ---------------------------------------------------------------------------
# Helpers — fake CSV data
# ---------------------------------------------------------------------------


def _make_temperature_csv_df(city: str = "New York") -> pd.DataFrame:
    """Create a small DataFrame mimicking temperature.csv structure."""
    dates = pd.date_range("2015-01-01", periods=48, freq="h")
    # Temperatures in Kelvin (roughly 270–285 K)
    rng = np.random.default_rng(42)
    temps = 273.15 + 5 + rng.standard_normal(len(dates)) * 3
    return pd.DataFrame({"datetime": dates.astype(str), city: temps})


def _make_homec_csv_df(column: str = "use [kW]") -> pd.DataFrame:
    """Create a small DataFrame mimicking HomeC.csv structure."""
    start_ts = int(pd.Timestamp("2016-03-01").timestamp())
    # 1-minute resolution, 1440 rows = 1 day
    timestamps = [start_ts + i * 60 for i in range(1440)]
    rng = np.random.default_rng(42)
    values = rng.uniform(0.1, 2.0, size=len(timestamps))
    return pd.DataFrame({"time": timestamps, column: values})


@pytest.fixture()
def mock_kagglehub():
    """Insert a mock kagglehub module into sys.modules for the test."""
    mock = MagicMock()
    with patch.dict(sys.modules, {"kagglehub": mock}):
        yield mock


# ---------------------------------------------------------------------------
# load_kaggle_temperature
# ---------------------------------------------------------------------------


class TestLoadKaggleTemperature:
    """Tests for load_kaggle_temperature."""

    def test_returns_ds_y_columns(self, mock_kagglehub, tmp_path):
        mock_kagglehub.dataset_download.return_value = str(tmp_path)
        with patch("vangja.datasets.loaders.pd.read_csv") as mock_csv:
            mock_csv.return_value = _make_temperature_csv_df("New York")
            df = load_kaggle_temperature("New York")

        assert list(df.columns) == ["ds", "y"]

    def test_kelvin_to_celsius(self, mock_kagglehub, tmp_path):
        """Temperature should be converted from Kelvin to Celsius."""
        mock_kagglehub.dataset_download.return_value = str(tmp_path)
        with patch("vangja.datasets.loaders.pd.read_csv") as mock_csv:
            mock_csv.return_value = _make_temperature_csv_df("New York")
            df = load_kaggle_temperature("New York")

        # All Celsius values should be roughly -10..+20 for our test data
        assert df["y"].max() < 50
        assert df["y"].min() > -50

    def test_date_filtering(self, mock_kagglehub, tmp_path):
        mock_kagglehub.dataset_download.return_value = str(tmp_path)
        with patch("vangja.datasets.loaders.pd.read_csv") as mock_csv:
            mock_csv.return_value = _make_temperature_csv_df("New York")
            df = load_kaggle_temperature(
                "New York",
                start_date="2015-01-01 12:00:00",
                end_date="2015-01-01 23:00:00",
                freq="h",
            )

        assert df["ds"].min() >= pd.Timestamp("2015-01-01 12:00:00")
        assert df["ds"].max() <= pd.Timestamp("2015-01-01 23:00:00")

    def test_daily_aggregation(self, mock_kagglehub, tmp_path):
        mock_kagglehub.dataset_download.return_value = str(tmp_path)
        with patch("vangja.datasets.loaders.pd.read_csv") as mock_csv:
            mock_csv.return_value = _make_temperature_csv_df("New York")
            df = load_kaggle_temperature("New York", freq="D")

        # 48 hourly rows over 2 days → should produce 2 daily rows
        assert len(df) == 2

    def test_hourly_no_aggregation(self, mock_kagglehub, tmp_path):
        mock_kagglehub.dataset_download.return_value = str(tmp_path)
        with patch("vangja.datasets.loaders.pd.read_csv") as mock_csv:
            mock_csv.return_value = _make_temperature_csv_df("New York")
            df = load_kaggle_temperature("New York", freq="h")

        assert len(df) == 48

    def test_invalid_city_raises(self):
        with pytest.raises(ValueError, match="Unknown city"):
            load_kaggle_temperature("Atlantis")

    def test_valid_cities_list(self):
        """Sanity check: the cities list has the expected length."""
        assert len(KAGGLE_TEMPERATURE_CITIES) == 36
        assert "New York" in KAGGLE_TEMPERATURE_CITIES
        assert "Jerusalem" in KAGGLE_TEMPERATURE_CITIES


# ---------------------------------------------------------------------------
# load_smart_home_readings
# ---------------------------------------------------------------------------


class TestLoadSmartHomeReadings:
    """Tests for load_smart_home_readings."""

    def test_returns_ds_y_columns(self, mock_kagglehub, tmp_path):
        mock_kagglehub.dataset_download.return_value = str(tmp_path)
        with patch("vangja.datasets.loaders.pd.read_csv") as mock_csv:
            mock_csv.return_value = _make_homec_csv_df("use [kW]")
            df = load_smart_home_readings("use [kW]")

        assert list(df.columns) == ["ds", "y"]

    def test_daily_aggregation(self, mock_kagglehub, tmp_path):
        mock_kagglehub.dataset_download.return_value = str(tmp_path)
        with patch("vangja.datasets.loaders.pd.read_csv") as mock_csv:
            mock_csv.return_value = _make_homec_csv_df("use [kW]")
            df = load_smart_home_readings("use [kW]", freq="D")

        # 1440 one-minute rows all within a single day → 1 daily row
        assert len(df) == 1

    def test_hourly_aggregation(self, mock_kagglehub, tmp_path):
        mock_kagglehub.dataset_download.return_value = str(tmp_path)
        with patch("vangja.datasets.loaders.pd.read_csv") as mock_csv:
            mock_csv.return_value = _make_homec_csv_df("use [kW]")
            df = load_smart_home_readings("use [kW]", freq="h")

        # 1440 minutes in 1 day → 24 hourly bins
        assert len(df) == 24

    def test_date_filtering(self, mock_kagglehub, tmp_path):
        mock_kagglehub.dataset_download.return_value = str(tmp_path)
        # Create 2 days of data
        start_ts = int(pd.Timestamp("2016-03-01").timestamp())
        timestamps = [start_ts + i * 60 for i in range(2880)]  # 2 days
        rng = np.random.default_rng(7)
        raw = pd.DataFrame({
            "time": timestamps,
            "use [kW]": rng.uniform(0.1, 2.0, size=len(timestamps)),
        })
        with patch("vangja.datasets.loaders.pd.read_csv") as mock_csv:
            mock_csv.return_value = raw
            df = load_smart_home_readings(
                "use [kW]",
                start_date="2016-03-02",
                end_date="2016-03-02",
                freq="D",
            )

        assert len(df) == 1
        assert df["ds"].iloc[0] == pd.Timestamp("2016-03-02")

    def test_invalid_column_raises(self):
        with pytest.raises(ValueError, match="Unknown column"):
            load_smart_home_readings("Toaster [kW]")

    def test_malformed_rows_dropped(self, mock_kagglehub, tmp_path):
        """Rows with non-numeric values in y should be dropped."""
        mock_kagglehub.dataset_download.return_value = str(tmp_path)
        raw = _make_homec_csv_df("use [kW]")
        # Inject a malformed row (as if the CSV has a trailing backslash)
        bad_row = pd.DataFrame({"time": ["\\"], "use [kW]": ["\\"]})
        raw = pd.concat([raw, bad_row], ignore_index=True)
        with patch("vangja.datasets.loaders.pd.read_csv") as mock_csv:
            mock_csv.return_value = raw
            df = load_smart_home_readings("use [kW]", freq="h")

        # Should still work, with the bad row silently dropped
        assert len(df) == 24
        assert df["y"].isna().sum() == 0

    def test_valid_columns_list(self):
        """Sanity check: the columns list has the expected length."""
        assert len(SMART_HOME_COLUMNS) == 18
        assert "use [kW]" in SMART_HOME_COLUMNS
        assert "Fridge [kW]" in SMART_HOME_COLUMNS
