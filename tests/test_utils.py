"""Tests for vangja.utils module."""

import numpy as np
import pandas as pd
import pytest

from vangja.utils import get_group_definition


class TestGetGroupDefinition:
    """Tests for get_group_definition function."""

    def test_complete_pooling_single_series(self, sample_data):
        """Test complete pooling with a single series."""
        group, n_groups, group_mapping = get_group_definition(sample_data, "complete")

        assert n_groups == 1
        assert len(group) == len(sample_data)
        assert all(g == 0 for g in group)
        assert 0 in group_mapping
        assert group_mapping[0] == "test_series"

    def test_complete_pooling_multi_series(self, multi_series_data):
        """Test complete pooling with multiple series (should treat as one group)."""
        group, n_groups, group_mapping = get_group_definition(
            multi_series_data, "complete"
        )

        assert n_groups == 1
        assert all(g == 0 for g in group)

    def test_partial_pooling_single_series(self, sample_data):
        """Test partial pooling with a single series."""
        group, n_groups, group_mapping = get_group_definition(sample_data, "partial")

        assert n_groups == 1
        assert len(group) == len(sample_data)

    def test_partial_pooling_multi_series(self, multi_series_data):
        """Test partial pooling with multiple series."""
        group, n_groups, group_mapping = get_group_definition(
            multi_series_data, "partial"
        )

        assert n_groups == 2
        assert len(group) == len(multi_series_data)
        # Check that we have two different group codes
        unique_groups = np.unique(group)
        assert len(unique_groups) == 2

    def test_group_mapping_keys_match_unique_groups(self, multi_series_data):
        """Test that group_mapping keys match unique group codes."""
        group, n_groups, group_mapping = get_group_definition(
            multi_series_data, "partial"
        )

        unique_groups = np.unique(group)
        assert set(unique_groups) == set(group_mapping.keys())

    def test_group_mapping_values_are_series_names(self, multi_series_data):
        """Test that group_mapping values are original series names."""
        group, n_groups, group_mapping = get_group_definition(
            multi_series_data, "partial"
        )

        series_names = set(multi_series_data["series"].unique())
        mapping_values = set(group_mapping.values())
        assert mapping_values == series_names

    def test_group_array_dtype(self, sample_data):
        """Test that group array is integer type."""
        group, _, _ = get_group_definition(sample_data, "complete")
        assert group.dtype in [np.int64, np.int32, int]

    def test_group_array_length_matches_data(self, sample_data):
        """Test that group array length matches data length."""
        group, _, _ = get_group_definition(sample_data, "partial")
        assert len(group) == len(sample_data)


class TestGetGroupDefinitionEdgeCases:
    """Edge case tests for get_group_definition."""

    def test_empty_dataframe(self):
        """Test with empty dataframe - should handle gracefully or raise."""
        empty_df = pd.DataFrame({"ds": [], "y": [], "series": []})

        # Empty dataframe with complete pooling should raise IndexError
        # when trying to get the first row for mapping
        with pytest.raises(IndexError):
            get_group_definition(empty_df, "complete")

    def test_single_row_dataframe(self):
        """Test with single row dataframe."""
        single_row = pd.DataFrame(
            {"ds": [pd.Timestamp("2020-01-01")], "y": [100.0], "series": ["single"]}
        )
        group, n_groups, group_mapping = get_group_definition(single_row, "complete")

        assert len(group) == 1
        assert n_groups == 1

    def test_many_series(self):
        """Test with many different series."""
        np.random.seed(42)
        n_series = 10
        dfs = []
        for i in range(n_series):
            dates = pd.date_range(start="2020-01-01", periods=50, freq="D")
            dfs.append(
                pd.DataFrame(
                    {"ds": dates, "y": np.random.randn(50), "series": f"series_{i}"}
                )
            )
        multi_df = pd.concat(dfs, ignore_index=True)

        group, n_groups, group_mapping = get_group_definition(multi_df, "partial")

        assert n_groups == n_series
        assert len(group_mapping) == n_series


class TestRelativeMAE:
    """Tests for the Relative-MAE-against-persistence primary metric."""

    def _unit(self, y_true, y_pred, y_pers):
        from vangja.utils import relative_mae

        return relative_mae(y_true, y_pred, y_pers)

    def test_beats_persistence_below_one(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pers = np.array([0.0, 0.0, 0.0, 0.0])  # MAE = 2.5
        y_pred = np.array([1.0, 2.0, 3.0, 4.0])  # MAE = 0
        assert self._unit(y_true, y_pred, y_pers) == pytest.approx(0.0)

    def test_worse_than_persistence_above_one(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pers = np.array([1.0, 1.0, 1.0, 1.0])  # MAE = 1.5
        y_pred = np.array([10.0, 10.0, 10.0, 10.0])  # MAE = 7.5
        assert self._unit(y_true, y_pred, y_pers) == pytest.approx(5.0)

    def test_epsilon_excludes_near_zero_denominator(self):
        from vangja.utils import relative_mae

        # Exact-zero denominator: excluded (nan), never inf
        y_true = np.zeros(3)
        y_pers = np.zeros(3)
        y_pred = np.zeros(3)
        assert np.isnan(relative_mae(y_true, y_pred, y_pers))
        # Below-threshold (but nonzero) denominator: excluded by epsilon
        y_true = np.array([1e-6, 0.0, 0.0])
        y_pers = np.zeros(3)
        assert np.isnan(relative_mae(y_true, y_pred, y_pers, epsilon=1e-3))

    def test_epsilon_threshold(self):
        from vangja.utils import relative_mae

        y_true = np.array([0.001, 0.0])
        y_pers = np.array([0.0, 0.0])  # MAE = 0.0005
        y_pred = np.array([0.0015, 0.0])  # MAE = 0.00025
        assert np.isnan(relative_mae(y_true, y_pred, y_pers, epsilon=1e-3))
        assert relative_mae(y_true, y_pred, y_pers, epsilon=1e-4) == pytest.approx(0.5)

    def test_scale_invariant(self):
        from vangja.utils import relative_mae

        y_true = np.array([1.0, 2.0, 3.0])
        y_pers = np.array([1.0, 1.0, 1.0])
        y_pred = np.array([1.5, 2.5, 3.5])
        assert relative_mae(y_true, y_pred, y_pers) == pytest.approx(
            relative_mae(10 * y_true, 10 * y_pred, 10 * y_pers)
        )

    def test_empty_returns_nan(self):
        from vangja.utils import relative_mae

        assert np.isnan(relative_mae([], [], []))


class TestPersistenceForecast:
    """Tests for the persistence (random-walk) forecast helper."""

    def test_last_value_carried_forward(self):
        from vangja.utils import persistence_forecast

        train = pd.DataFrame(
            {
                "ds": pd.date_range("2020-01-01", periods=3),
                "y": [10.0, 12.0, 15.0],
                "series": "A",
            }
        )
        test = pd.DataFrame(
            {
                "ds": pd.date_range("2020-01-04", periods=4),
                "y": [1.0, 2.0, 3.0, 4.0],
                "series": "A",
            }
        )
        pers = persistence_forecast(train, test)
        assert list(pers.columns) == ["ds", "series", "yhat"]
        assert (pers["yhat"] == 15.0).all()
        assert (pers["ds"] == test["ds"]).all()

    def test_multi_series(self):
        from vangja.utils import persistence_forecast

        train = pd.DataFrame(
            {
                "ds": list(pd.date_range("2020-01-01", periods=3)) * 2,
                "y": [10.0, 12.0, 15.0, 5.0, 6.0, 4.0],
                "series": ["A", "A", "A", "B", "B", "B"],
            }
        )
        test = pd.DataFrame(
            {
                "ds": list(pd.date_range("2020-01-04", periods=2)) * 2,
                "y": [0.0] * 4,
                "series": ["A", "A", "B", "B"],
            }
        )
        pers = persistence_forecast(train, test)
        a = pers[pers["series"] == "A"]["yhat"]
        b = pers[pers["series"] == "B"]["yhat"]
        assert (a == 15.0).all()
        assert (b == 4.0).all()

    def test_series_missing_from_train_skipped(self):
        from vangja.utils import persistence_forecast

        train = pd.DataFrame(
            {
                "ds": pd.date_range("2020-01-01", periods=2),
                "y": [1.0, 2.0],
                "series": "A",
            }
        )
        test = pd.DataFrame(
            {
                "ds": pd.date_range("2020-01-03", periods=2),
                "y": [0.0, 0.0],
                "series": "B",
            }
        )
        pers = persistence_forecast(train, test)
        assert pers.empty
