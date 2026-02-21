"""Tests for Kaggle dataset loaders.

These tests mock kagglehub and CSV reads to stay offline and deterministic.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from vangja.datasets.loaders import (
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


def _make_homec_csv_df(*columns: str) -> pd.DataFrame:
    """Create a small DataFrame mimicking HomeC.csv structure.

    Supports one or more columns.
    """
    n_rows = 1440  # 1 day at 1-minute resolution
    rng = np.random.default_rng(42)
    data: dict[str, object] = {}
    for col in columns:
        data[col] = rng.uniform(0.1, 2.0, size=n_rows)
    return pd.DataFrame(data)


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


# ---------------------------------------------------------------------------
# load_smart_home_readings — single column
# ---------------------------------------------------------------------------


class TestLoadSmartHomeReadingsSingle:
    """Tests for load_smart_home_readings with a single column."""

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

        # 1440 minutes starting at 05:00 spans 2 calendar days
        assert len(df) == 2

    def test_hourly_aggregation(self, mock_kagglehub, tmp_path):
        mock_kagglehub.dataset_download.return_value = str(tmp_path)
        with patch("vangja.datasets.loaders.pd.read_csv") as mock_csv:
            mock_csv.return_value = _make_homec_csv_df("use [kW]")
            df = load_smart_home_readings("use [kW]", freq="h")

        # 1440 minutes in 1 day → 24 hourly bins
        assert len(df) == 24

    def test_no_series_column_for_single(self, mock_kagglehub, tmp_path):
        """A single column should NOT produce a 'series' column."""
        mock_kagglehub.dataset_download.return_value = str(tmp_path)
        with patch("vangja.datasets.loaders.pd.read_csv") as mock_csv:
            mock_csv.return_value = _make_homec_csv_df("Fridge [kW]")
            df = load_smart_home_readings("Fridge [kW]")

        assert "series" not in df.columns

    def test_malformed_rows_dropped(self, mock_kagglehub, tmp_path):
        """Rows with non-numeric values in y should be dropped."""
        mock_kagglehub.dataset_download.return_value = str(tmp_path)
        raw = _make_homec_csv_df("use [kW]")
        # Inject a malformed row (as if the CSV has a trailing backslash)
        bad_row = pd.DataFrame({"use [kW]": ["\\"]})
        raw = pd.concat([raw, bad_row], ignore_index=True)
        with patch("vangja.datasets.loaders.pd.read_csv") as mock_csv:
            mock_csv.return_value = raw
            df = load_smart_home_readings("use [kW]", freq="h")

        # Should still work, with the bad row silently dropped
        assert len(df) == 24
        assert df["y"].isna().sum() == 0


# ---------------------------------------------------------------------------
# load_smart_home_readings — multiple columns
# ---------------------------------------------------------------------------


class TestLoadSmartHomeReadingsMulti:
    """Tests for load_smart_home_readings with a list of columns."""

    def test_returns_ds_y_series_columns(self, mock_kagglehub, tmp_path):
        mock_kagglehub.dataset_download.return_value = str(tmp_path)
        with patch("vangja.datasets.loaders.pd.read_csv") as mock_csv:
            mock_csv.return_value = _make_homec_csv_df(
                "Fridge [kW]", "Microwave [kW]"
            )
            df = load_smart_home_readings(["Fridge [kW]", "Microwave [kW]"])

        assert list(df.columns) == ["ds", "y", "series"]

    def test_series_values(self, mock_kagglehub, tmp_path):
        """The series column should contain the original column names."""
        mock_kagglehub.dataset_download.return_value = str(tmp_path)
        cols = ["Fridge [kW]", "Microwave [kW]"]
        with patch("vangja.datasets.loaders.pd.read_csv") as mock_csv:
            mock_csv.return_value = _make_homec_csv_df(*cols)
            df = load_smart_home_readings(cols)

        assert set(df["series"].unique()) == set(cols)

    def test_multi_column_daily_aggregation(self, mock_kagglehub, tmp_path):
        mock_kagglehub.dataset_download.return_value = str(tmp_path)
        cols = ["Fridge [kW]", "Microwave [kW]"]
        with patch("vangja.datasets.loaders.pd.read_csv") as mock_csv:
            mock_csv.return_value = _make_homec_csv_df(*cols)
            df = load_smart_home_readings(cols, freq="D")

        # 2 columns × 2 calendar days = 4 rows
        assert len(df) == 4
        assert df["series"].nunique() == 2

    def test_multi_column_hourly_aggregation(self, mock_kagglehub, tmp_path):
        mock_kagglehub.dataset_download.return_value = str(tmp_path)
        cols = ["use [kW]", "gen [kW]", "Fridge [kW]"]
        with patch("vangja.datasets.loaders.pd.read_csv") as mock_csv:
            mock_csv.return_value = _make_homec_csv_df(*cols)
            df = load_smart_home_readings(cols, freq="h")

        # 3 columns × 24 hours = 72 rows
        assert len(df) == 72

    def test_multi_column_no_aggregation(self, mock_kagglehub, tmp_path):
        mock_kagglehub.dataset_download.return_value = str(tmp_path)
        cols = ["Fridge [kW]", "Microwave [kW]"]
        with patch("vangja.datasets.loaders.pd.read_csv") as mock_csv:
            mock_csv.return_value = _make_homec_csv_df(*cols)
            df = load_smart_home_readings(cols)

        # 2 columns × 1440 minute-rows = 2880
        assert len(df) == 2880
