"""Extended tests for vangja.utils module.

Tests for remove_random_gaps, filter_predictions_by_series, metrics,
and plotting utilities that were not covered in the original test suite.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from vangja.utils import (
    filter_predictions_by_series,
    metrics,
    remove_random_gaps,
)


# ---------------------------------------------------------------------------
# remove_random_gaps
# ---------------------------------------------------------------------------


class TestRemoveRandomGaps:
    """Tests for remove_random_gaps function."""

    def _make_df(self, n: int = 200) -> pd.DataFrame:
        dates = pd.date_range("2020-01-01", periods=n, freq="D")
        return pd.DataFrame({"ds": dates, "y": np.arange(n, dtype=float)})

    def test_basic_gap_removal(self):
        """Test that gaps are actually removed."""
        df = self._make_df(200)
        result = remove_random_gaps(df, n_gaps=2, gap_fraction=0.1)

        assert len(result) < len(df)

    def test_returns_dataframe(self):
        """Test that the return value is a DataFrame."""
        df = self._make_df(200)
        result = remove_random_gaps(df, n_gaps=1, gap_fraction=0.1)

        assert isinstance(result, pd.DataFrame)

    def test_preserves_columns(self):
        """Test that all columns are preserved."""
        df = self._make_df(200)
        df["series"] = "test"
        result = remove_random_gaps(df, n_gaps=1, gap_fraction=0.1)

        assert list(result.columns) == list(df.columns)

    def test_index_is_reset(self):
        """Test that the index is reset after gap removal."""
        df = self._make_df(200)
        result = remove_random_gaps(df, n_gaps=2, gap_fraction=0.1)

        assert result.index[0] == 0
        assert result.index[-1] == len(result) - 1
        assert list(result.index) == list(range(len(result)))

    def test_approximate_gap_fraction(self):
        """Test that the number of removed points is close to expectations."""
        np.random.seed(42)
        df = self._make_df(1000)
        n_gaps = 4
        gap_fraction = 0.1
        result = remove_random_gaps(df, n_gaps=n_gaps, gap_fraction=gap_fraction)

        expected_removed = int(1000 * gap_fraction) * n_gaps
        actual_removed = len(df) - len(result)

        # Allow some tolerance — gaps may overlap with edges
        assert actual_removed <= expected_removed
        assert actual_removed > 0

    def test_n_gaps_one(self):
        """Test with a single gap."""
        np.random.seed(42)
        df = self._make_df(100)
        result = remove_random_gaps(df, n_gaps=1, gap_fraction=0.2)

        assert len(result) < len(df)
        assert len(result) >= 80  # At most 20% removed

    def test_raises_on_excessive_gap_removal(self):
        """Test that ValueError is raised when gaps exceed data length."""
        df = self._make_df(100)

        with pytest.raises(ValueError, match="Cannot remove"):
            remove_random_gaps(df, n_gaps=10, gap_fraction=0.2)

    def test_deterministic_with_seed(self):
        """Test that results are deterministic with the same random seed."""
        df = self._make_df(200)

        np.random.seed(123)
        result1 = remove_random_gaps(df, n_gaps=2, gap_fraction=0.15)

        np.random.seed(123)
        result2 = remove_random_gaps(df, n_gaps=2, gap_fraction=0.15)

        pd.testing.assert_frame_equal(result1, result2)

    def test_does_not_modify_original(self):
        """Test that the original DataFrame is not modified."""
        df = self._make_df(200)
        original_len = len(df)
        _ = remove_random_gaps(df, n_gaps=2, gap_fraction=0.1)

        assert len(df) == original_len

    def test_small_dataset(self):
        """Test with a very small dataset."""
        df = self._make_df(20)
        result = remove_random_gaps(df, n_gaps=1, gap_fraction=0.1)

        assert len(result) < 20
        assert len(result) > 0


# ---------------------------------------------------------------------------
# filter_predictions_by_series
# ---------------------------------------------------------------------------


class TestFilterPredictionsBySeries:
    """Tests for filter_predictions_by_series function."""

    def test_basic_filtering(self):
        """Test basic date-range filtering."""
        future = pd.DataFrame(
            {
                "ds": pd.date_range("2020-01-01", periods=365, freq="D"),
                "yhat_0": np.random.randn(365),
                "yhat_1": np.random.randn(365),
            }
        )
        series_data = pd.DataFrame(
            {
                "ds": pd.date_range("2020-03-01", periods=30, freq="D"),
                "y": np.random.randn(30),
            }
        )

        result = filter_predictions_by_series(future, series_data, yhat_col="yhat_0")

        assert result["ds"].min() >= series_data["ds"].min()
        assert result["ds"].max() <= series_data["ds"].max()

    def test_column_renaming(self):
        """Test that the yhat column is renamed to yhat_0."""
        future = pd.DataFrame(
            {
                "ds": pd.date_range("2020-01-01", periods=100, freq="D"),
                "yhat_5": np.arange(100, dtype=float),
            }
        )
        series_data = pd.DataFrame(
            {
                "ds": pd.date_range("2020-01-10", periods=20, freq="D"),
                "y": np.zeros(20),
            }
        )

        result = filter_predictions_by_series(
            future, series_data, yhat_col="yhat_5"
        )

        assert "yhat_0" in result.columns
        assert "yhat_5" not in result.columns

    def test_with_horizon(self):
        """Test filtering with a forecast horizon."""
        future = pd.DataFrame(
            {
                "ds": pd.date_range("2020-01-01", periods=200, freq="D"),
                "yhat_0": np.arange(200, dtype=float),
            }
        )
        series_data = pd.DataFrame(
            {
                "ds": pd.date_range("2020-01-01", periods=100, freq="D"),
                "y": np.zeros(100),
            }
        )

        result = filter_predictions_by_series(
            future, series_data, yhat_col="yhat_0", horizon=50
        )

        expected_max = series_data["ds"].max() + pd.Timedelta(days=50)
        assert result["ds"].max() <= expected_max
        assert len(result) > 100  # Should include some horizon dates

    def test_index_is_reset(self):
        """Test that the index is reset."""
        future = pd.DataFrame(
            {
                "ds": pd.date_range("2020-01-01", periods=100, freq="D"),
                "yhat_0": np.zeros(100),
            }
        )
        series_data = pd.DataFrame(
            {
                "ds": pd.date_range("2020-02-01", periods=20, freq="D"),
                "y": np.zeros(20),
            }
        )

        result = filter_predictions_by_series(future, series_data)

        assert result.index[0] == 0

    def test_output_has_only_ds_and_yhat(self):
        """Test that output only contains ds and yhat_0 columns."""
        future = pd.DataFrame(
            {
                "ds": pd.date_range("2020-01-01", periods=50, freq="D"),
                "yhat_0": np.zeros(50),
                "yhat_1": np.ones(50),
                "extra_col": np.zeros(50),
            }
        )
        series_data = pd.DataFrame(
            {
                "ds": pd.date_range("2020-01-01", periods=30, freq="D"),
                "y": np.zeros(30),
            }
        )

        result = filter_predictions_by_series(future, series_data)

        assert list(result.columns) == ["ds", "yhat_0"]

    def test_empty_result_when_no_overlap(self):
        """Test that no rows are returned when date ranges don't overlap."""
        future = pd.DataFrame(
            {
                "ds": pd.date_range("2020-01-01", periods=30, freq="D"),
                "yhat_0": np.zeros(30),
            }
        )
        series_data = pd.DataFrame(
            {
                "ds": pd.date_range("2021-01-01", periods=30, freq="D"),
                "y": np.zeros(30),
            }
        )

        result = filter_predictions_by_series(future, series_data)

        assert len(result) == 0


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


class TestMetrics:
    """Tests for metrics function."""

    def test_basic_metrics(self):
        """Test that metrics returns all expected metric columns."""
        dates = pd.date_range("2020-01-01", periods=50, freq="D")
        y_true = pd.DataFrame(
            {"ds": dates, "y": np.ones(50) * 100, "series": "test"}
        )
        future = pd.DataFrame(
            {"ds": dates, "yhat_0": np.ones(50) * 110}
        )

        result = metrics(y_true, future, "complete")

        assert "mse" in result.columns
        assert "rmse" in result.columns
        assert "mae" in result.columns
        assert "mape" in result.columns

    def test_perfect_prediction(self):
        """Test that all metrics are zero for perfect predictions."""
        dates = pd.date_range("2020-01-01", periods=30, freq="D")
        values = np.random.randn(30) * 10 + 50
        y_true = pd.DataFrame({"ds": dates, "y": values, "series": "test"})
        future = pd.DataFrame({"ds": dates, "yhat_0": values})

        result = metrics(y_true, future, "complete")

        assert result["mse"].values[0] == pytest.approx(0.0, abs=1e-10)
        assert result["rmse"].values[0] == pytest.approx(0.0, abs=1e-10)
        assert result["mae"].values[0] == pytest.approx(0.0, abs=1e-10)
        assert result["mape"].values[0] == pytest.approx(0.0, abs=1e-10)

    def test_known_error(self):
        """Test metrics with known constant error."""
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        y_true = pd.DataFrame(
            {"ds": dates, "y": np.ones(10) * 100, "series": "test"}
        )
        future = pd.DataFrame({"ds": dates, "yhat_0": np.ones(10) * 110})

        result = metrics(y_true, future, "complete")

        assert result["mse"].values[0] == pytest.approx(100.0)
        assert result["rmse"].values[0] == pytest.approx(10.0)
        assert result["mae"].values[0] == pytest.approx(10.0)
        assert result["mape"].values[0] == pytest.approx(0.1)

    def test_without_series_column(self):
        """Test that metrics works when y_true has no 'series' column."""
        dates = pd.date_range("2020-01-01", periods=20, freq="D")
        y_true = pd.DataFrame({"ds": dates, "y": np.ones(20) * 50})
        future = pd.DataFrame({"ds": dates, "yhat_0": np.ones(20) * 55})

        result = metrics(y_true, future, "complete")

        assert len(result) == 1
        assert "mse" in result.columns

    def test_no_matching_dates_raises(self):
        """Test that ValueError is raised when no dates match."""
        y_true = pd.DataFrame(
            {
                "ds": pd.date_range("2020-01-01", periods=10, freq="D"),
                "y": np.ones(10),
                "series": "test",
            }
        )
        future = pd.DataFrame(
            {
                "ds": pd.date_range("2021-01-01", periods=10, freq="D"),
                "yhat_0": np.ones(10),
            }
        )

        with pytest.raises(ValueError, match="No matching dates"):
            metrics(y_true, future, "complete")

    def test_multi_series_metrics(self):
        """Test metrics with multiple series."""
        dates = pd.date_range("2020-01-01", periods=20, freq="D")
        y_true = pd.DataFrame(
            {
                "ds": list(dates) * 2,
                "y": list(np.ones(20) * 100) + list(np.ones(20) * 200),
                "series": ["series_a"] * 20 + ["series_b"] * 20,
            }
        )
        future = pd.DataFrame(
            {
                "ds": dates,
                "yhat_0": np.ones(20) * 105,
                "yhat_1": np.ones(20) * 195,
            }
        )

        result = metrics(y_true, future, "partial")

        assert len(result) == 2
        assert "series_a" in result.index
        assert "series_b" in result.index

    def test_partial_date_overlap(self):
        """Test metrics when predictions only partially overlap with test data."""
        dates_true = pd.date_range("2020-01-01", periods=30, freq="D")
        dates_pred = pd.date_range("2020-01-15", periods=30, freq="D")

        y_true = pd.DataFrame(
            {"ds": dates_true, "y": np.ones(30) * 100, "series": "test"}
        )
        future = pd.DataFrame({"ds": dates_pred, "yhat_0": np.ones(30) * 110})

        result = metrics(y_true, future, "complete")

        # Should compute metrics on the overlapping dates only
        assert result["mae"].values[0] == pytest.approx(10.0)

    def test_returns_dataframe(self):
        """Test that metrics returns a pandas DataFrame."""
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        y_true = pd.DataFrame(
            {"ds": dates, "y": np.ones(10), "series": "test"}
        )
        future = pd.DataFrame({"ds": dates, "yhat_0": np.ones(10) * 2})

        result = metrics(y_true, future, "complete")

        assert isinstance(result, pd.DataFrame)
